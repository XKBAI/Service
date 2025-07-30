import unittest

import pytest
import requests
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
)


@pytest.mark.nonci
class MolmoProcessorTest(unittest.TestCase):
    def test_molmo_demo(self):
        # load the processor
        processor = AutoProcessor.from_pretrained(
            "allenai/Molmo-7B-O-0924",
            trust_remote_code=True,
            torch_dtype="auto",
        )

        # load the model
        model = AutoModelForCausalLM.from_pretrained(
            "allenai/Molmo-7B-O-0924",
            trust_remote_code=True,
            torch_dtype="auto",
        )

        device = "cuda:0"

        model = model.to(device)

        # process the image and text
        inputs = processor.process(images=[Image.open(requests.get("https://picsum.photos/id/237/536/354", stream=True).raw)], text="Describe this image.")

        # move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

        print("Raw inputs")
        print(inputs)

        print("\nShapes")
        # {('input_ids', torch.Size([1, 589])), ('images', torch.Size([1, 5, 576, 588])), ('image_masks', torch.Size([1, 5, 576])), ('image_input_idx', torch.Size([1, 5, 144]))}
        print({(x, y.shape) for x, y in inputs.items()})

        print("\nTokens")
        print(processor.tokenizer.batch_decode(inputs["input_ids"]))

        # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
        output = model.generate_from_batch(inputs, GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"), tokenizer=processor.tokenizer)

        # only get generated tokens; decode them to text
        generated_tokens = output[0, inputs["input_ids"].size(1) :]
        generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # print the generated text
        print(generated_text)
