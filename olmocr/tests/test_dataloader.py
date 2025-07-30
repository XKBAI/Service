import unittest
from functools import partial

import pytest
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

from olmocr.train.dataloader import (
    build_finetuning_dataset,
    extract_openai_batch_response,
    list_dataset_files,
    load_jsonl_into_ds,
)
from olmocr.train.dataprep import batch_prepare_data_for_qwen2_training


@pytest.mark.nonci
class TestBatchQueryResponseDataset(unittest.TestCase):
    def testLoadS3(self):
        ds = load_jsonl_into_ds("s3://ai2-oe-data/jakep/openai_batch_data_v2/*.jsonl", first_n_files=3)

        print(f"Loaded {len(ds)} entries")
        print(ds)
        print(ds["train"])

    def testFinetuningDS(self):
        ds = build_finetuning_dataset(
            response_glob_path="s3://ai2-oe-data/jakep/pdfdata/openai_batch_done_v5_1_eval/*.json",
        )

        print(ds)

        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

        ds = ds.with_transform(partial(batch_prepare_data_for_qwen2_training, processor=processor, target_longest_image_dim=1024, target_anchor_text_len=6000))

        print(ds[0])

    def testPlotSequenceLengthHistogram(self):
        import plotly.express as px

        ds = build_finetuning_dataset(
            response_glob_path="s3://ai2-oe-data/jakep/pdfdata/openai_batch_done_v5_1_eval/*.json",
        )

        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

        ds = ds.with_transform(partial(batch_prepare_data_for_qwen2_training, processor=processor, target_longest_image_dim=1024, target_anchor_text_len=6000))

        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

        initial_len = len(ds)

        train_dataloader = DataLoader(ds, batch_size=1, num_workers=30, shuffle=False)

        max_seen_len = 0
        steps = 0
        sequence_lengths = []  # List to store sequence lengths
        for entry in tqdm(train_dataloader):
            num_input_tokens = entry["input_ids"].shape[1]
            max_seen_len = max(max_seen_len, num_input_tokens)
            sequence_lengths.append(num_input_tokens)  # Collecting sequence lengths

            if steps % 100 == 0:
                print(f"Max input len {max_seen_len}")

            steps += 1

            # model.forward(**{k: v.to("cuda:0") for (k,v) in entry.items()})
        print(f"Max input len {max_seen_len}")
        print(f"Total elements before filtering: {initial_len}")
        print(f"Total elements after filtering: {steps}")

        # Plotting the histogram using Plotly
        fig = px.histogram(
            sequence_lengths, nbins=100, title="Distribution of Input Sequence Lengths", labels={"value": "Sequence Length", "count": "Frequency"}
        )

        fig.write_image("sequence_lengths_histogram.png")
