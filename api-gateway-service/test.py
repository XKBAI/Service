from google import genai

client = genai.Client(api_key="")

my_file = client.files.upload(file="")

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[my_file, "Caption this image."],
)

print(response.text)