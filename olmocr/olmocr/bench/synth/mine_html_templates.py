import argparse
import asyncio
import concurrent.futures
import json
import os
import random
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import pypdf
from anthropic import Anthropic
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from syntok.segmenter import process
from tqdm import tqdm

from olmocr.bench.tests import (
    TestType,
)
from olmocr.data.renderpdf import (
    get_png_dimensions_from_base64,
    render_pdf_to_base64png,
)


def download_s3_pdf(s3_path, local_path):
    """Download a PDF from S3 to a local path."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    result = subprocess.run(["aws", "s3", "cp", s3_path, local_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.returncode == 0


def generate_html_from_image(client, image_base64):
    """Call Claude API to generate HTML from an image using a multi-step prompting strategy."""
    png_width, png_height = get_png_dimensions_from_base64(image_base64)

    try:
        # Step 1: Initial analysis and column detection
        analysis_response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=2000,
            temperature=0.1,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_base64}},
                        {
                            "type": "text",
                            "text": "Analyze this document and provide a detailed assessment of its structure. Focus specifically on:\n"
                            "1. How many columns does the document have? Is it single-column, two-column, three-column, or a mixed layout?\n"
                            "2. What are the main sections and content types (headings, paragraphs, lists, tables, images, etc.)?\n"
                            "3. Does it have headers, footers, page numbers, or other special elements?\n"
                            "4. Is there any complex formatting that would be challenging to reproduce in HTML?\n\n"
                            "Please be very precise about the number of columns and how they're arranged.",
                        },
                    ],
                }
            ],
        )

        analysis_text = ""
        for content in analysis_response.content:
            if content.type == "text":
                analysis_text += content.text

        # Step 2: Initial HTML generation with detailed layout instructions
        initial_response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=6000,
            temperature=0.2,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_base64}},
                        {
                            "type": "text",
                            "text": "Render this document as clean, semantic HTML. Here's my analysis of the document structure:\n\n"
                            f"{analysis_text}\n\n"
                            "Important requirements:\n"
                            "1. Use appropriate HTML tags for elements like headings, paragraphs, lists, tables, etc.\n"
                            "2. Use the <header> and <footer> tags to represent content at the top/bottom which would not normally be part of the main content, such as page numbers, etc.\n"
                            "3. Use a placeholder <div> tag with class 'image' which will render as a grey box with black outline to make sure images have their original size, shape, and position on the page.\n"
                            "4. CRITICAL: If the document has a multi-column layout, you MUST preserve the exact same number of columns in your HTML. Use CSS flexbox or grid to create the columns.\n"
                            "5. Focus on creating valid, accessible HTML that preserves the appearance and formatting of the original page as closely as possible.\n"
                            f"6. The webpage will be viewed with a fixed viewport size of {png_width // 2} pixels wide by {png_height // 2} pixels tall.\n\n"
                            "7. For multi-column layouts, use explicit CSS. The most important aspect is preserving the column structure of the original document - this is critical.\n\n"
                            "Enclose your HTML in a ```html code block.",
                        },
                    ],
                }
            ],
        )

        # Extract initial HTML
        initial_html = ""
        for content in initial_response.content:
            if content.type == "text":
                initial_html += content.text

        # Extract code block
        if "```html" in initial_html:
            start = initial_html.find("```html") + 7
            end = initial_html.rfind("```")
            if end > start:
                initial_html = initial_html[start:end].strip()
            else:
                initial_html = initial_html[start:].strip()
        elif "```" in initial_html:
            start = initial_html.find("```") + 3
            end = initial_html.rfind("```")
            if end > start:
                initial_html = initial_html[start:end].strip()
            else:
                initial_html = initial_html[start:].strip()

        return initial_html
    except Exception as e:
        print(f"Error calling Claude API: {e}")
        return None


def extract_page_from_pdf(input_path, output_path, page_num):
    """
    Extract a specific page from a PDF and save it as a new PDF.

    Args:
        input_path: Path to the input PDF
        output_path: Path to save the extracted page
        page_num: The page number to extract (1-indexed, converted to 0-indexed for pypdf)

    Returns:
        bool: True if extraction was successful, False otherwise
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Read the input PDF
        reader = pypdf.PdfReader(input_path)

        # Convert to 0-indexed for pypdf
        zero_idx_page = page_num - 1

        # Check if page number is valid
        if zero_idx_page >= len(reader.pages) or zero_idx_page < 0:
            print(f"Page number {page_num} out of range for {input_path} with {len(reader.pages)} pages")
            return False

        # Create a new PDF with just the selected page
        writer = pypdf.PdfWriter()
        writer.add_page(reader.pages[zero_idx_page])

        # Write the output PDF
        with open(output_path, "wb") as output_file:
            writer.write(output_file)

        return True
    except Exception as e:
        print(f"Error extracting page {page_num} from {input_path}: {str(e)}")
        return False


async def render_pdf_with_playwright(html_content, output_pdf_path, png_width, png_height):
    """
    Render HTML content using Playwright and save it as PDF.
    Try different scale factors if needed to ensure the output is exactly one page.

    Args:
        html_content: HTML content to render
        output_pdf_path: Path to save the rendered PDF
        png_width: Width of the viewport
        png_height: Height of the viewport

    Returns:
        bool: True if rendering was successful with exactly one page, False otherwise
    """
    scale_factors = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]  # Try these scale factors in order

    for scale in scale_factors:
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page(viewport={"width": int(png_width // 2 * scale), "height": int(png_height // 2 * scale)})

                # Set the HTML content
                await page.set_content(html_content)

                # Save as PDF with formatting options
                await page.pdf(
                    path=output_pdf_path,
                    scale=scale,
                    print_background=True,
                )

                await browser.close()

                # Check if the output PDF has exactly one page
                try:
                    reader = pypdf.PdfReader(output_pdf_path)
                    if len(reader.pages) == 1:
                        print(f"Successfully rendered as a single page PDF with scale factor {scale}")
                        return True
                    else:
                        print(f"PDF has {len(reader.pages)} pages with scale factor {scale}, trying a smaller scale...")
                        # Continue to the next scale factor
                except Exception as pdf_check_error:
                    print(f"Error checking PDF page count: {pdf_check_error}")
                    return False

        except Exception as e:
            print(f"Error rendering PDF with Playwright at scale {scale}: {str(e)}")
            # Try the next scale factor

    print("Failed to render PDF as a single page with any scale factor")
    return False


def generate_tests_from_html(html_content: str, pdf_id: str, page_num: int) -> List[Dict]:
    """
    Generate tests from HTML content parsed from the PDF.

    Args:
        html_content: The HTML content of the page
        pdf_id: The unique identifier for the PDF
        page_num: The page number

    Returns:
        A list of test dictionaries that can be saved as JSONL
    """
    tests = []
    pdf_filename = f"{pdf_id}_page{page_num}.pdf"
    soup = BeautifulSoup(html_content, "html.parser")

    # Step 1: Process headers, footers, and page numbers for TextAbsenceTests
    headers = soup.find_all("header")
    footers = soup.find_all("footer")
    page_numbers = soup.find_all("div", class_="page-number")

    # Function to create absence tests from text elements
    def create_absence_tests_from_elements(parent_element, element_type):
        # Find all text-containing leaf elements within the parent
        text_elements = []

        # Get all target elements
        target_tags = parent_element.find_all(["span", "div", "p", "h1", "h2", "h3", "h4", "h5", "h6"])
        
        # Filter to only include leaf nodes (elements that don't contain other target elements)
        for tag in target_tags:
            # Check if this element has no children from our target tags
            is_leaf = not tag.find(["span", "div", "p", "h1", "h2", "h3", "h4", "h5", "h6"])
            
            if is_leaf:
                text = tag.get_text().strip()
                if text:
                    text_elements.append(text)

        # If no elements found, use the parent's text as a fallback
        if not text_elements:
            parent_text = parent_element.get_text().strip()
            if parent_text:
                text_elements.append(parent_text)

        # Create tests for each text element
        for text in text_elements:
            if len(text) > 3:  # Only create tests for meaningful text
                tests.append(
                    {
                        "pdf": pdf_filename,
                        "page": page_num,
                        "id": f"{pdf_id}_{element_type}_{uuid.uuid4().hex[:8]}",
                        "type": TestType.ABSENT.value,
                        "text": text,
                        "max_diffs": 0,
                    }
                )

    # Create TextAbsenceTests for headers
    for header in headers:
        create_absence_tests_from_elements(header, "header")

    # Create TextAbsenceTests for footers
    for footer in footers:
        create_absence_tests_from_elements(footer, "footer")

    # Create TextAbsenceTests for page numbers
    for page_number in page_numbers:
        page_number_text = page_number.get_text().strip()
        if page_number_text:
            tests.append(
                {
                    "pdf": pdf_filename,
                    "page": page_num,
                    "id": f"{pdf_id}_page_number_{uuid.uuid4().hex[:8]}",
                    "type": TestType.ABSENT.value,
                    "text": page_number_text,
                    "max_diffs": 0,
                }
            )

    # Step 2: Generate tests from tables
    tables = soup.find_all("table")
    for table_idx, table in enumerate(tables):
        # Get all cells in the table
        cells = table.find_all(["td", "th"])

        # Skip empty tables or tables with very few cells
        if len(cells) < 4:
            continue

        # Generate tests for some randomly selected cells
        sampled_cells = random.sample(cells, min(3, len(cells)))

        for cell in sampled_cells:
            cell_text = cell.get_text().strip()
            if not cell_text or len(cell_text) < 3:
                continue

            # Find position of this cell in the table
            row = cell.find_parent("tr")
            rows = table.find_all("tr")
            row_idx = rows.index(row)

            # Find cells in this row
            row_cells = row.find_all(["td", "th"])
            col_idx = row_cells.index(cell)

            # Create a TableTest with relevant relationships
            test_data = {
                "pdf": pdf_filename,
                "page": page_num,
                "id": f"{pdf_id}_table{table_idx}_{uuid.uuid4().hex[:8]}",
                "type": TestType.TABLE.value,
                "cell": cell_text,
                "max_diffs": 5,
            }

            # Check cell up
            if row_idx > 0:
                prev_row = rows[row_idx - 1]
                prev_row_cells = prev_row.find_all(["td", "th"])
                if col_idx < len(prev_row_cells):
                    up_text = prev_row_cells[col_idx].get_text().strip()
                    if up_text:
                        test_data["up"] = up_text

            # Check cell down
            if row_idx < len(rows) - 1:
                next_row = rows[row_idx + 1]
                next_row_cells = next_row.find_all(["td", "th"])
                if col_idx < len(next_row_cells):
                    down_text = next_row_cells[col_idx].get_text().strip()
                    if down_text:
                        test_data["down"] = down_text

            # Check cell left
            if col_idx > 0:
                left_text = row_cells[col_idx - 1].get_text().strip()
                if left_text:
                    test_data["left"] = left_text

            # Check cell right
            if col_idx < len(row_cells) - 1:
                right_text = row_cells[col_idx + 1].get_text().strip()
                if right_text:
                    test_data["right"] = right_text

            # Check top heading (first row in the table or a header row)
            if row_idx > 0:
                header_row = rows[0]
                header_cells = header_row.find_all(["td", "th"])
                if col_idx < len(header_cells):
                    top_heading = header_cells[col_idx].get_text().strip()
                    if top_heading:
                        test_data["top_heading"] = top_heading

            # Check left heading (first column in the table)
            if col_idx > 0:
                left_heading = row_cells[0].get_text().strip()
                if left_heading:
                    test_data["left_heading"] = left_heading

            # Only add the test if we have at least one relation
            if len(test_data) > 6:  # 6 is the number of required fields
                tests.append(test_data)

    # Step 3: Generate TextPresenceTests for main body content
    # Make a copy of the soup for the main content
    main_soup = BeautifulSoup(str(soup), "html.parser")

    # Remove headers, footers, and tables from the main_soup
    for element in main_soup.find_all(["header", "footer", "table", "head"]):
        element.extract()

    # Get all paragraphs and headings in the main content
    paragraphs = main_soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])

    full_text = main_soup.get_text().strip()

    sentences = []
    for paragraph in process(full_text):
        for sentence in paragraph:
            # Convert token sequence to string and clean it
            sentence_str = ""
            for token in sentence:
                sentence_str += token.spacing + token.value

            sentence_str = sentence_str.strip()

            if sentence_str:
                sentences.append(sentence_str)

    # Add a few random ordering tests
    all_indexes = list(range(len(sentences)))
    
    # Ex. pick N pairs of indexes from all_indexes
    random_pairs = set()
    for _ in range(10):
        idx1, idx2 = random.sample(all_indexes, 2)
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        random_pairs.add((idx1, idx2))

    for i, j in random_pairs:
        first_sentence = sentences[i]
        second_sentence = sentences[j]

        if len(first_sentence) < 10 or len(second_sentence) < 10:
            continue

        tests.append(
            {
                "pdf": pdf_filename,
                "page": page_num,
                "id": f"{pdf_id}_order_{uuid.uuid4().hex[:8]}",
                "type": TestType.ORDER.value,
                "before": first_sentence,
                "after": second_sentence,
                "max_diffs": round(max(len(first_sentence), len(second_sentence)) * 0.05),
            }
        )

    return tests


def process_pdf(pdf_info, args, client):
    """Process a single PDF, render a random page, and create an HTML template."""
    s3_path, index = pdf_info

    # Create a unique folder for each PDF in the temp directory
    pdf_id = f"pdf_{index:05d}"
    temp_pdf_dir = os.path.join(args.temp_dir, pdf_id)
    os.makedirs(temp_pdf_dir, exist_ok=True)

    # Download PDF to local temp directory
    local_pdf_path = os.path.join(temp_pdf_dir, "document.pdf")
    if not download_s3_pdf(s3_path, local_pdf_path):
        print(f"Failed to download PDF from {s3_path}")
        return None

    try:
        # Get page count using pypdf
        reader = pypdf.PdfReader(local_pdf_path)
        num_pages = len(reader.pages)

        if num_pages == 0:
            print(f"PDF has no pages: {s3_path}")
            return None

        # Select a random page
        page_num = random.randint(1, num_pages)

        # Render the page as a base64 PNG
        image_base64 = render_pdf_to_base64png(local_pdf_path, page_num, target_longest_image_dim=2048)

        # Generate HTML from the image
        html_content = generate_html_from_image(client, image_base64)
        if not html_content:
            print(f"Failed to generate HTML for {s3_path}, page {page_num}")
            return None

        # Create output directories
        html_dir = os.path.join(args.output_dir, "html")
        pdfs_dir = os.path.join(args.output_dir, "pdfs")
        os.makedirs(html_dir, exist_ok=True)
        os.makedirs(pdfs_dir, exist_ok=True)

        # Save HTML to output directory
        html_path = os.path.join(html_dir, f"{pdf_id}_page{page_num}.html")
        with open(html_path, "w") as f:
            f.write(html_content)

        # Extract the page and save as PDF
        original_pdf_path = os.path.join(pdfs_dir, f"{pdf_id}_page{page_num}_original.pdf")
        if not extract_page_from_pdf(local_pdf_path, original_pdf_path, page_num):
            print(f"Failed to extract page {page_num} from {local_pdf_path}")

        # Render PDF using Playwright if not skipped
        playwright_pdf_path = None
        render_success = False
        playwright_pdf_filename = f"{pdf_id}_page{page_num}.pdf"  # This will be used in the tests

        if not args.skip_playwright:
            playwright_pdf_path = os.path.join(pdfs_dir, playwright_pdf_filename)

            try:
                # Get PNG dimensions
                png_width, png_height = get_png_dimensions_from_base64(image_base64)

                # Run the async function in the synchronous context
                render_success = asyncio.run(render_pdf_with_playwright(html_content, playwright_pdf_path, png_width, png_height))

                if render_success:
                    print(f"Successfully rendered with Playwright: {playwright_pdf_path}")
                else:
                    print(f"Failed to render as a single page PDF: {playwright_pdf_path}")
                    playwright_pdf_path = None
            except Exception as e:
                print(f"Failed to render with Playwright: {e}")
                playwright_pdf_path = None
                render_success = False

        # If playwright rendering failed and was required, return None to skip this test
        if not args.skip_playwright and not render_success:
            return None
            
        # Generate tests from the HTML content
        # Use the playwright rendered PDF path for tests
        tests = generate_tests_from_html(html_content, pdf_id, page_num)
        
        # Update the PDF path in all tests to use the playwright rendered PDF
        for test in tests:
            test["pdf"] = playwright_pdf_filename
                
        return {
            "pdf_id": pdf_id,
            "s3_path": s3_path,
            "page_number": page_num,
            "html_path": html_path,
            "original_pdf_path": original_pdf_path,
            "playwright_pdf_path": playwright_pdf_path,
            "tests": tests,
            "num_tests": len(tests),
        }
    except Exception as e:
        print(f"Error processing {s3_path}: {e}")
        return None
    finally:
        # Clean up temp directory for this PDF
        if os.path.exists(temp_pdf_dir):
            subprocess.run(["rm", "-rf", temp_pdf_dir])


def main():
    parser = argparse.ArgumentParser(description="Convert PDFs to HTML templates and render with Playwright")
    parser.add_argument("--input_list", required=True, help="Path to a file containing S3 paths to PDFs")
    parser.add_argument("--output_dir", required=True, help="Directory to store extracted pages and tests")
    parser.add_argument("--temp_dir", default="/tmp/mine_tables", help="Directory for temporary files")
    parser.add_argument("--max_tests", type=int, default=100, help="Maximum number of tests to generate")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel threads to use")
    parser.add_argument("--api_key", help="Claude API key (or set ANTHROPIC_API_KEY environment variable)")
    parser.add_argument("--skip_playwright", action="store_true", help="Skip Playwright PDF rendering")
    args = parser.parse_args()

    # Ensure output and temp directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)

    # Get API key
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: API key not provided. Use --api_key or set ANTHROPIC_API_KEY environment variable.")
        return

    # Initialize Claude client
    client = Anthropic(api_key=api_key)

    # Reservoir sampling implementation
    s3_paths = []
    with open(args.input_list, "r") as f:
        for i, line in enumerate(tqdm(f)):
            line = line.strip()
            if not line:
                continue

            if i < 100000:
                s3_paths.append(line)
            else:
                # Randomly replace elements with decreasing probability
                j = random.randint(0, i)
                if j < 100000:
                    s3_paths[j] = line

    print(f"Found {len(s3_paths)} PDF paths in input list")

    # Shuffle and limit to max_tests
    random.shuffle(s3_paths)
    s3_paths = s3_paths[: args.max_tests]
    
    # Initialize synthetic.json as a JSONL file (empty initially)
    synthetic_json_path = os.path.join(args.output_dir, "synthetic.jsonl")
    open(synthetic_json_path, "w").close()  # Create empty file
    
    # Counter for test statistics
    test_counter = 0
    test_types = {"present": 0, "absent": 0, "table": 0, "order": 0}
    results = []
    
    # Initialize a threading lock for file access
    import threading
    file_lock = threading.Lock()

    # Process PDFs in parallel
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        # Submit all tasks
        futures = {executor.submit(process_pdf, (s3_path, i), args, client): s3_path for i, s3_path in enumerate(s3_paths)}

        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing PDFs"):
            s3_path = futures[future]
            try:
                result = future.result()
                if result and result.get("tests"):
                    results.append(result)
                    
                    # Append tests to synthetic.json as they're created (JSONL format)
                    with file_lock:
                        # Append each test as a separate JSON line
                        with open(synthetic_json_path, "a") as f:
                            for test in result["tests"]:
                                f.write(json.dumps(test) + "\n")
                        
                        # Update counters
                        test_counter += len(result["tests"])
                        for test in result["tests"]:
                            test_type = test.get("type", "")
                            if test_type in test_types:
                                test_types[test_type] += 1
                                
                        print(f"Added {len(result['tests'])} tests from {result['pdf_id']}, total: {test_counter}")
            except Exception as e:
                print(f"Error processing {s3_path}: {e}")

    print(f"Generated {len(results)} HTML templates")

    # Print summary of Playwright rendering results
    playwright_success = sum(1 for r in results if r and r.get("playwright_pdf_path"))
    if not args.skip_playwright:
        print(f"Playwright PDF rendering: {playwright_success}/{len(results)} successful")
    
    print(f"Saved {test_counter} tests to {synthetic_json_path}")
    
    # Print summary of generated tests
    print(f"Generated a total of {test_counter} tests across {len(results)} templates")

    # Print test type distribution
    if test_counter > 0:
        print("Test type distribution:")
        for test_type, count in test_types.items():
            print(f"  - {test_type}: {count} tests")


if __name__ == "__main__":
    main()
