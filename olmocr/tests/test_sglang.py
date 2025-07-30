# The idea is that you have a Qwen2-VL-7B model located here:s3://ai2-oe-data/jakep/experiments/qwen2vl-pdf/v1/models/jakep/Qwen_Qwen2-VL-7B-Instruct-e4ecf8-01JAH8GMWHTJ376S2N7ETXRXH4/checkpoint-9500/bf16/"

# You need to load it in both hugging face transformers, and send page 1 of edgar.pdf to it from tests/gnarly_pdfs
# Compare that the temperature 0 sampled result is the same

import asyncio
import base64
import json
import math
import os
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from httpx import AsyncClient
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration

from olmocr.pipeline import (
    SGLANG_SERVER_PORT,
    build_page_query,
    get_anchor_text,
    render_pdf_to_base64png,
    sglang_server_ready,
    sglang_server_task,
)
from olmocr.prompts import PageResponse

MODEL_FINETUNED_PATH = (
    "s3://ai2-oe-data/jakep/experiments/qwen2vl-pdf/v1/models/jakep/Qwen_Qwen2-VL-7B-Instruct-e4ecf8-01JAH8GMWHTJ376S2N7ETXRXH4/checkpoint-9500/bf16/"
)


@pytest.mark.nonci
class TestSglangServer(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Mock arguments
        self.args = AsyncMock()
        self.args.workspace = "/tmp/test_workspace"
        self.args.model = [MODEL_FINETUNED_PATH]
        self.args.model_chat_template = "qwen2-vl"
        self.args.target_longest_image_dim = 1024
        self.args.target_anchor_text_len = 6000
        self.args.model_max_context = 8192

        # Create a temporary workspace directory
        os.makedirs(self.args.workspace, exist_ok=True)

        # Set up a semaphore for server tasks
        self.semaphore = asyncio.Semaphore(1)
        self.maxDiff = None

        # # Start the sglang server
        # self.my_server_task = asyncio.create_task(sglang_server_task(self.args, self.semaphore))

        # # Wait for the server to become ready
        # await sglang_server_ready()

    async def test_sglang_server_initialization_and_request(self):
        # Mock data paths
        self.test_pdf_path = Path(os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "ambiguous.pdf"))

        # Send a single request to the sglang server for page 1
        async with AsyncClient(timeout=600) as session:
            query = await build_page_query(
                str(self.test_pdf_path),
                page=1,
                target_longest_image_dim=self.args.target_longest_image_dim,
                target_anchor_text_len=self.args.target_anchor_text_len,
            )
            COMPLETION_URL = f"http://localhost:{30000}/v1/chat/completions"

            query["temperature"] = 0.0
            query["logprobs"] = True
            query["top_logprobs"] = 5
            response = await session.post(COMPLETION_URL, json=query)

        print(response.text)

        # Check the server response
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertIn("choices", response_data)
        self.assertGreater(len(response_data["choices"]), 0)

        model_response_json = json.loads(response_data["choices"][0]["message"]["content"])
        page_response = PageResponse(**model_response_json)

        print(page_response)

        self.assertEqual(page_response.natural_text, EDGAR_TEXT)

    async def asyncTearDown(self):
        pass
        # # Shut down the server
        # self.my_server_task.cancel()
        # with self.assertRaises(asyncio.CancelledError):
        #     await self.my_server_task

        # # Cleanup temporary workspace
        # if os.path.exists(self.args.workspace):
        #     for root, _, files in os.walk(self.args.workspace):
        #         for file in files:
        #             os.unlink(os.path.join(root, file))
        #     os.rmdir(self.args.workspace)


@pytest.mark.nonci
class TestHuggingFaceModel(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Set up the Hugging Face model and tokenizer
        model_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "olmocr", "model")
        download_directory([MODEL_FINETUNED_PATH], model_cache_dir)

        # Check the rope config and make sure it's got the proper key
        with open(os.path.join(model_cache_dir, "config.json"), "r") as cfin:
            config_data = json.load(cfin)

        if "rope_type" in config_data["rope_scaling"]:
            del config_data["rope_scaling"]["rope_type"]
            config_data["rope_scaling"]["type"] = "mrope"

            with open(os.path.join(model_cache_dir, "config.json"), "w") as cfout:
                json.dump(config_data, cfout)

        self.tokenizer = AutoTokenizer.from_pretrained(model_cache_dir, trust_remote_code=True)
        self.image_token_id = self.tokenizer.encode("<|image_pad|>")[0]

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_cache_dir, torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Path to the test PDF
        self.test_pdf_path = Path(os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "ambiguous.pdf"))
        self.maxDiff = None

    async def test_hugging_face_generation(self):
        query = await build_page_query(
            str(self.test_pdf_path),
            page=1,
            target_longest_image_dim=1024,
            target_anchor_text_len=6000,
        )

        messages = query["messages"]

        # Apply chat template to get the text
        text = self.processor.apply_chat_template(query["messages"], tokenize=False, add_generation_prompt=True)

        image_url = query["messages"][0]["content"][1]["image_url"]["url"]

        # Remove the "data:image/png;base64," prefix
        base64_image = image_url.split(",")[1]

        # Decode the base64 string into bytes
        image_data = base64.b64decode(base64_image)

        # Create a BytesIO object and load it into a PIL image
        main_image = Image.open(BytesIO(image_data))

        # Process inputs using processor
        inputs = self.processor(
            text=[text],
            images=[main_image],
            padding=True,
            return_tensors="pt",
        )

        image_indices = [idx for idx, token in enumerate(inputs["input_ids"][0]) if token.item() == self.image_token_id]

        print("IMAGE INDICES", image_indices)

        print(f"image_grid_thw - {inputs['image_grid_thw'].shape} {inputs['image_grid_thw']}")
        print(f"pixel_values - {inputs['pixel_values'].shape} {inputs['pixel_values'].detach().cpu().numpy()}")
        np.save("/root/pixel_values.npy", inputs["pixel_values"].detach().cpu().numpy())

        inputs = {key: value.to(self.device) for (key, value) in inputs.items()}

        generated_tokens = []
        max_steps = 50

        top_logprobs_hf = []

        for step in range(max_steps):
            # Generate the output with temperature=0
            generation_output = self.model.generate(
                **inputs,
                temperature=0.0,
                max_new_tokens=1,
                # max_length=8192,
                num_return_sequences=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

            # Extract the generated token's log probabilities
            scores = generation_output.scores  # Tuple of length 1
            logits = scores[0]  # Tensor of shape (batch_size, vocab_size)
            log_probs = F.log_softmax(logits, dim=-1)  # Apply log softmax to get log probabilities

            # Get top 5 tokens and their log probabilities
            topk_log_probs, topk_indices = torch.topk(log_probs[0], k=5)
            topk_tokens = self.tokenizer.convert_ids_to_tokens(topk_indices.tolist())

            top_logprobs_hf.append((topk_tokens, topk_log_probs.tolist()))

            # Pick the top token
            next_token_id = topk_indices[0].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1)
            next_token_str = self.tokenizer.convert_ids_to_tokens([next_token_id.item()])[0]

            generated_tokens.append(next_token_id.item())

            # Append the next token to input_ids and update attention_mask
            inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token_id], dim=-1)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones((1, 1), dtype=inputs["attention_mask"].dtype).to(self.device)], dim=-1)

        print(self.tokenizer.decode(generated_tokens))

        # Now take all the input ids and run them through sglang as a comparison
        async with AsyncClient(timeout=600) as session:
            query["temperature"] = 0.0
            query["max_tokens"] = max_steps
            query["logprobs"] = True
            query["top_logprobs"] = 5
            COMPLETION_URL = f"http://localhost:{30000}/v1/chat/completions"
            response = await session.post(COMPLETION_URL, json=query)

            response_data = response.json()

            for step, lptok in enumerate(response_data["choices"][0]["logprobs"]["content"]):
                print("\nTop 5 tokens and their log probabilities:")
                (topk_tokens, topk_log_probs) = top_logprobs_hf[step]
                for token, log_prob, lptokcur in zip(topk_tokens, topk_log_probs, lptok["top_logprobs"]):
                    print(
                        f"HF Token: {token} Log Prob: {log_prob:.2f} Prob {math.exp(log_prob)*100:.2f}%  SGLANG Token {lptokcur['token']} Logprob {lptokcur['logprob']:.2f} Prob {math.exp(lptokcur['logprob'])*100:.2f}%"
                    )

    async def asyncTearDown(self):
        # Clean up the model and tokenizer
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()


@pytest.mark.nonci
class RawSGLangTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Set up the Hugging Face model and tokenizer
        model_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "olmocr", "model")
        download_directory([MODEL_FINETUNED_PATH], model_cache_dir)

        # Check the rope config and make sure it's got the proper key
        with open(os.path.join(model_cache_dir, "config.json"), "r") as cfin:
            config_data = json.load(cfin)

        if "rope_type" in config_data["rope_scaling"]:
            del config_data["rope_scaling"]["rope_type"]
            config_data["rope_scaling"]["type"] = "mrope"

            with open(os.path.join(model_cache_dir, "config.json"), "w") as cfout:
                json.dump(config_data, cfout)

        self.model_cache_dir = model_cache_dir

        self.tokenizer = AutoTokenizer.from_pretrained(model_cache_dir, trust_remote_code=True)
        self.image_token_id = self.tokenizer.encode("<|image_pad|>")[0]

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_cache_dir, torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Path to the test PDF
        self.test_pdf_path = Path(os.path.join(os.path.dirname(__file__), "gnarly_pdfs", "ambiguous.pdf"))
        self.maxDiff = None

    async def test_vision_encoder(self):
        query = await build_page_query(
            str(self.test_pdf_path),
            page=1,
            target_longest_image_dim=1024,
            target_anchor_text_len=6000,
        )

        messages = query["messages"]

        # Apply chat template to get the text
        text = self.processor.apply_chat_template(query["messages"], tokenize=False, add_generation_prompt=True)

        image_url = query["messages"][0]["content"][1]["image_url"]["url"]

        # Remove the "data:image/png;base64," prefix
        base64_image = image_url.split(",")[1]

        # Decode the base64 string into bytes
        image_data = base64.b64decode(base64_image)

        # Create a BytesIO object and load it into a PIL image
        main_image = Image.open(BytesIO(image_data))

        # Process inputs using processor
        inputs = self.processor(
            text=[text],
            images=[main_image],
            padding=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            hf_output = self.model.visual(inputs["pixel_values"].to(self.device), grid_thw=inputs["image_grid_thw"].to(self.device))

        print("HF", hf_output, hf_output.shape)

        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.hf_transformers_utils import get_tokenizer
        from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        from sglang.srt.model_executor.model_runner import ModelRunner
        from sglang.srt.sampling.sampling_params import SamplingParams
        from sglang.srt.server_args import PortArgs, ServerArgs

        model_config = ModelConfig(self.model_cache_dir, model_override_args="{}")

        server_args = ServerArgs(model_path=self.model_cache_dir)
        # Initialize model runner
        model_runner = ModelRunner(
            model_config=model_config,
            mem_fraction_static=0.8,
            gpu_id=0,
            tp_rank=0,
            tp_size=1,
            nccl_port=12435,
            server_args=server_args,
        )

        print(model_runner)
        with torch.no_grad():
            sglang_output = model_runner.model.visual(inputs["pixel_values"].to(self.device), grid_thw=inputs["image_grid_thw"].to(self.device))

        print("SGLANG", sglang_output, sglang_output.shape)

        # Convert to float32 for numerical stability if needed
        hf = hf_output.float()
        sg = sglang_output.float()

        # Basic shape and dtype comparison
        print("\n=== Basic Properties ===")
        print(f"Shapes match: {hf.shape == sg.shape}")
        print(f"HF shape: {hf.shape}, SGLang shape: {sg.shape}")
        print(f"HF dtype: {hf.dtype}, SGLang dtype: {sg.dtype}")

        # Move tensors to CPU for numpy operations
        hf_np = hf.cpu().numpy()
        sg_np = sg.cpu().numpy()

        # Statistical metrics
        print("\n=== Statistical Metrics ===")
        print(f"Mean absolute difference: {torch.mean(torch.abs(hf - sg)).item():.6f}")
        print(f"Max absolute difference: {torch.max(torch.abs(hf - sg)).item():.6f}")
        print(f"Mean squared error: {torch.mean((hf - sg) ** 2).item():.6f}")
        print(f"Root mean squared error: {torch.sqrt(torch.mean((hf - sg) ** 2)).item():.6f}")

        # Cosine similarity (across feature dimension)
        cos_sim = F.cosine_similarity(hf, sg)
        print(f"Mean cosine similarity: {torch.mean(cos_sim).item():.6f}")
        print(f"Min cosine similarity: {torch.min(cos_sim).item():.6f}")

        # Find largest absolute differences
        print("\n=== Largest Absolute Differences ===")
        diffs = torch.abs(hf - sg)
        flat_diffs = diffs.flatten()

        # Get indices of top 10 differences
        top_k = 10
        top_values, top_flat_indices = torch.topk(flat_diffs, top_k)

        # Convert flat indices to multidimensional indices
        top_indices = np.unravel_index(top_flat_indices.cpu().numpy(), diffs.shape)

        print(f"\nTop {top_k} largest absolute differences:")
        print("Index".ljust(30) + "Difference".ljust(15) + "HF Value".ljust(15) + "SGLang Value")
        print("-" * 75)

        for i in range(top_k):
            # Get the index tuple for this difference
            idx = tuple(dim[i] for dim in top_indices)
            diff_val = top_values[i].item()
            hf_val = hf[idx].item()
            sg_val = sg[idx].item()

            # Format the index tuple and values
            idx_str = str(idx)
            print(f"{idx_str:<30}{diff_val:<15.6f}{hf_val:<15.6f}{sg_val:.6f}")
