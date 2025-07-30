import base64
import os
import random
import re
import unittest
from io import BytesIO
from unittest.mock import patch

import numpy as np
import pytest
import requests
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

from olmocr.train.core.config import DataConfig, SourceConfig, TrainConfig
from olmocr.train.dataloader import build_finetuning_dataset
from olmocr.train.dataprep import (
    batch_prepare_data_for_molmo_training,
    build_finetuning_prompt,
    prepare_data_for_molmo_training,
    prepare_data_for_qwen2_training,
)
from olmocr.train.utils import make_dataset


@pytest.mark.nonci
class TestDataprep(unittest.TestCase):
    def testFullDataloader(self):
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        config = TrainConfig(
            train_data=DataConfig(
                seed=42,
                sources=[
                    SourceConfig(
                        name="eval_test",
                        target_longest_image_dim=1024,
                        target_anchor_text_len=6000,
                        response_glob_path="s3://ai2-oe-data/jakep/pdfdata/openai_batch_done_v5_1_eval/*.json",
                    )
                ],
            ),
            valid_data=DataConfig(
                seed=42,
                sources=[
                    SourceConfig(
                        name="eval_test",
                        target_longest_image_dim=1024,
                        target_anchor_text_len=6000,
                        response_glob_path="s3://ai2-oe-data/jakep/pdfdata/openai_batch_done_v5_1_eval/*.json",
                    )
                ],
            ),
        )
        train_dataset, valid_dataset = make_dataset(config, processor)

        im_end_token_ids = processor.tokenizer("<|im_end|>\n", add_special_tokens=False)["input_ids"]

        # train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=False)
        for entry in train_dataset:
            print({x: (y.shape, y.dtype) for (x, y) in entry.items()})

            self.assertEqual(entry["input_ids"].dtype, np.int64)
            self.assertEqual(entry["attention_mask"].dtype, np.int64)
            self.assertEqual(entry["labels"].dtype, np.int64)
            self.assertEqual(entry["pixel_values"].dtype, np.float32)
            self.assertEqual(entry["image_grid_thw"].dtype, np.int64)

            # Extract input_ids and labels
            input_ids = entry["input_ids"]
            labels = entry["labels"]

            # 1. Verify that the last token is the end token
            # Ensure input_ids is long enough
            self.assertTrue(len(input_ids) >= len(im_end_token_ids), "Input IDs are shorter than the end token sequence.")

            # Compare the last tokens of input_ids with im_end_token_ids
            self.assertEqual(
                input_ids[-len(im_end_token_ids) :].tolist(), im_end_token_ids, "The last tokens of input_ids do not match the end token sequence."
            )

            # 2. Ensure labels are masked correctly and match input_ids after the mask
            # Find where labels start being non-masked (-100 is the mask value)
            label_indices = np.where(labels != -100)[0]

            # There should be at least one label that is not masked
            self.assertTrue(len(label_indices) > 0, "No unmasked labels found in labels array.")

            first_label_index = label_indices[0]

            # Ensure the masked portion is at least 10 tokens long
            self.assertTrue(first_label_index >= 10, "Masked portion of labels is less than 10 tokens.")

            # Check that all values before first_label_index are -100
            self.assertTrue(np.all(labels[:first_label_index] == -100), "Labels before the first unmasked token are not all -100.")

            # Check that the unmasked labels match the corresponding input_ids
            self.assertTrue(
                np.array_equal(labels[first_label_index:], input_ids[first_label_index:]), "Unmasked labels do not match the corresponding input_ids."
            )

            # Optionally, verify that the last unmasked tokens in labels match the end token IDs
            unmasked_labels = labels[labels != -100]
            self.assertEqual(
                unmasked_labels[-len(im_end_token_ids) :].tolist(), im_end_token_ids, "The last unmasked tokens in labels do not match the end token sequence."
            )

    def testListTargetAnchorLength(self):
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        config = TrainConfig(
            train_data=DataConfig(
                seed=42,
                sources=[
                    SourceConfig(
                        name="eval_test",
                        target_longest_image_dim=1024,
                        target_anchor_text_len=[0, 6000],  # Only 0 and 6000
                        response_glob_path="s3://ai2-oe-data/jakep/pdfdata/openai_batch_done_v5_1_eval/*.json",
                    )
                ],
            ),
            valid_data=DataConfig(
                seed=42,
                sources=[
                    SourceConfig(
                        name="eval_test",
                        target_longest_image_dim=1024,
                        target_anchor_text_len=[0, 6000],  # Only 0 and 6000
                        response_glob_path="s3://ai2-oe-data/jakep/pdfdata/openai_batch_done_v5_1_eval/*.json",
                    )
                ],
            ),
        )

        # Set a fixed seed for reproducibility
        random.seed(42)
        train_dataset, valid_dataset = make_dataset(config, processor)

        zero_count = 0
        full_count = 0
        num_iterations = 100

        for i in range(num_iterations):
            entry = train_dataset[0]  # Get the first entry repeatedly

            # Basic type checks
            self.assertEqual(entry["input_ids"].dtype, np.int64)
            self.assertEqual(entry["attention_mask"].dtype, np.int64)
            self.assertEqual(entry["labels"].dtype, np.int64)
            self.assertEqual(entry["pixel_values"].dtype, np.float32)
            self.assertEqual(entry["image_grid_thw"].dtype, np.int64)

            # Get the input text before the response
            # Find where labels start being non-masked (-100 is the mask value)
            label_indices = np.where(entry["labels"] != -100)[0]
            first_label_index = label_indices[0] if len(label_indices) > 0 else len(entry["input_ids"])

            # Decode the input portion to check the prompt
            input_text = processor.tokenizer.decode(entry["input_ids"][:first_label_index])

            pattern = r"RAW_TEXT_START\nPage dimensions: (\d+\.?\d*)x(\d+\.?\d*)\s+RAW_TEXT_END"

            match = re.search(pattern, input_text, flags=re.MULTILINE)
            if match:
                zero_count += 1
            else:
                full_count += 1

        # Verify the distribution: should be roughly 10% zero-length, 90% full-length
        zero_ratio = zero_count / num_iterations
        full_ratio = full_count / num_iterations

        print(zero_count, full_count)

        self.assertTrue(0.45 <= zero_ratio <= 0.55, f"Expected zero-length ratio around 0.5, got {zero_ratio:.2f}")
        self.assertTrue(0.45 <= full_ratio <= 0.55, f"Expected full-length ratio around 0.5, got {full_ratio:.2f}")

        # Verify total adds up to 100%
        self.assertEqual(zero_count + full_count, num_iterations, "Total count should equal number of iterations")


@pytest.mark.nonci
class TestMolmoDataPrep(unittest.TestCase):
    def testMolmoDefaultSetup(self):
        processor = AutoProcessor.from_pretrained("allenai/Molmo-7B-O-0924", trust_remote_code=True, torch_dtype="auto", device_map="auto")

        # process the image and text
        inputs = processor.process(images=[Image.open(requests.get("https://picsum.photos/id/237/536/354", stream=True).raw)], text="Describe this image.")

        print(inputs.keys())
        print(inputs["input_ids"])
        print(processor.tokenizer.batch_decode(inputs["input_ids"]))

        labels = processor.tokenizer("This is a page of the pdf that's the text", return_tensors="np")

        print(labels)
        print(processor.tokenizer.batch_decode(labels["input_ids"]))

    def testMolmoDataPrep(self):
        # Initialize the processor for Molmo
        processor = AutoProcessor.from_pretrained("allenai/Molmo-7B-O-0924", trust_remote_code=True, torch_dtype="auto", device_map="auto")

        # Create a mock example
        example = {
            "local_pdf_path": os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "edgar.pdf"),
            "page_num": 1,
            "response": "This is the response text.",
        }

        # Define target dimensions and anchor text lengths
        target_longest_image_dim = [1024]
        target_anchor_text_len = [0, 6000]

        # Set a fixed seed for reproducibility
        random.seed(42)

        # Mock the functions that require actual PDF files
        with (
            patch("olmocr.prompts.anchor.get_anchor_text") as mock_get_anchor_text,
            patch("olmocr.data.renderpdf.render_pdf_to_base64png") as mock_render_pdf_to_base64png,
        ):
            # Set return values for the mocked functions
            mock_get_anchor_text.return_value = "This is the anchor text."
            # Create a red square image and encode it in base64
            img = Image.new("RGB", (100, 100), color="red")
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            mock_render_pdf_to_base64png.return_value = img_str

            # Process the example using the prepare_data_for_molmo_training function
            processed_example = prepare_data_for_molmo_training(
                example, processor, target_longest_image_dim=target_longest_image_dim, target_anchor_text_len=target_anchor_text_len
            )

            # Basic type checks
            self.assertIsInstance(processed_example["input_ids"], torch.Tensor, "input_ids should be a torch.Tensor")
            self.assertIsInstance(processed_example["attention_mask"], torch.Tensor, "attention_mask should be a torch.Tensor")
            self.assertIsInstance(processed_example["labels"], torch.Tensor, "labels should be a torch.Tensor")
            self.assertIsInstance(processed_example["images"], torch.Tensor, "images should be a torch.Tensor")
            self.assertIsInstance(processed_example["image_input_idx"], torch.Tensor, "image_input_idx should be a torch.Tensor")
            self.assertIsInstance(processed_example["image_masks"], torch.Tensor, "image_masks should be a torch.Tensor")

            # Check tensor dimensions
            self.assertEqual(len(processed_example["input_ids"].shape), 1, "input_ids should be a 1D tensor")
            self.assertEqual(
                processed_example["input_ids"].shape, processed_example["attention_mask"].shape, "input_ids and attention_mask should have the same shape"
            )
            self.assertEqual(processed_example["input_ids"].shape, processed_example["labels"].shape, "input_ids and labels should have the same shape")

            # Verify label masking
            # Find where labels start being non-masked (-100 is the mask value)
            label_indices = torch.where(processed_example["labels"] != -100)[0]

            # There should be at least one label that is not masked
            self.assertTrue(len(label_indices) > 0, "No unmasked labels found in labels array.")

            first_label_index = label_indices[0]

            # Ensure the masked portion is reasonable (at least a few tokens long)
            self.assertTrue(first_label_index >= 5, "Masked portion of labels is too short")

            # Check that all values before first_label_index are -100
            self.assertTrue(torch.all(processed_example["labels"][:first_label_index] == -100), "Labels before the first unmasked token are not all -100.")

            # Verify attention mask
            self.assertTrue(torch.all(processed_example["attention_mask"] == 1), "All attention mask values should be 1")

            # Verify image input indices
            self.assertTrue(
                torch.all(processed_example["image_input_idx"] < len(processed_example["input_ids"])),
                "Image input indices should be within the range of input_ids length",
            )

            # Decode and verify content structure
            decoded_input = processor.tokenizer.decode(processed_example["input_ids"])
            self.assertIn("This is the anchor text", decoded_input, "Anchor text should be present in the decoded input")

            # Verify that unmasked labels decode to the response text
            unmasked_labels = processed_example["labels"][processed_example["labels"] != -100]
            decoded_labels = processor.tokenizer.decode(unmasked_labels)
            self.assertIn("This is the response text", decoded_labels, "Response text should be present in the decoded labels")

    def testBatchMolmoDataPrep(self):
        """Test the batch preparation function for Molmo"""
        processor = AutoProcessor.from_pretrained("allenai/Molmo-7B-O-0924", trust_remote_code=True, torch_dtype="auto", device_map="auto")

        # Create a mock batch
        batch = {
            "local_pdf_path": [os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "edgar.pdf")],
            "page_num": [1],
            "response": ["This is the response text."],
        }

        target_longest_image_dim = [1024]
        target_anchor_text_len = [0, 6000]

        # Mock the necessary functions
        with (
            patch("olmocr.prompts.anchor.get_anchor_text") as mock_get_anchor_text,
            patch("olmocr.data.renderpdf.render_pdf_to_base64png") as mock_render_pdf_to_base64png,
        ):
            mock_get_anchor_text.return_value = "This is the anchor text."
            img = Image.new("RGB", (100, 100), color="red")
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            mock_render_pdf_to_base64png.return_value = img_str

            # Process the batch
            processed_batch = batch_prepare_data_for_molmo_training(
                batch, processor, target_longest_image_dim=target_longest_image_dim, target_anchor_text_len=target_anchor_text_len
            )

            # Verify batch structure
            self.assertEqual(len(processed_batch["input_ids"]), 1, "Batch size should be 1")
            self.assertEqual(len(processed_batch["attention_mask"]), 1, "Batch size should be 1")
            self.assertEqual(len(processed_batch["labels"]), 1, "Batch size should be 1")
            self.assertEqual(len(processed_batch["images"]), 1, "Batch size should be 1")
            self.assertEqual(len(processed_batch["image_input_idx"]), 1, "Batch size should be 1")
            self.assertEqual(len(processed_batch["image_masks"]), 1, "Batch size should be 1")

            # Verify the first item in the batch
            first_item = {k: v[0] for k, v in processed_batch.items()}
            self.assertIsInstance(first_item["input_ids"], torch.Tensor, "Batch item should contain torch.Tensor")
            self.assertTrue(torch.all(first_item["attention_mask"] == 1), "All attention mask values should be 1")
