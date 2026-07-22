from typing import Optional
import math
import os
import csv
import json
import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from trm.data.common import PuzzleDatasetMetadata, dihedral_transform


# First 5 chars kept in their original order/ids for backward compatibility
# with the base sapientinc maze charset (# wall, space empty, S start, G
# goal, o solved-path). The rest are objective tiles:
#   C        checkpoint      -- must be crossed (always shown as C, visited or not)
#   R / r    reward tile     -- optional; r = reward tile that was crossed
#   N / n    penalty tile    -- optional; n = penalty tile that was crossed
CHARSET = "# SGoCRrNn"


cli = ArgParser()


class DataProcessConfig(BaseModel):
    train_csv: str = "train.csv"
    test_csv: str = "test.csv"
    output_dir: str = "data/maze-custom"

    subsample_size: Optional[int] = None
    aug: bool = False


def convert_subset(set_name: str, csv_path: str, config: DataProcessConfig):
    # Read CSV
    all_chars = set()
    grid_size = None
    inputs = []
    labels = []

    with open(csv_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header

        for source, q, a, rating in reader:
            q = q.replace("\n", "").replace("\r", "")
            a = a.replace("\n", "").replace("\r", "")
            
            all_chars.update(q)
            all_chars.update(a)

            if grid_size is None:
                n = int(len(q) ** 0.5)
                grid_size = (n, n)

            inputs.append(
                np.frombuffer(q.encode(), dtype=np.uint8).reshape(grid_size)
            )
            labels.append(
                np.frombuffer(a.encode(), dtype=np.uint8).reshape(grid_size)
            )

    # Optional subsampling
    if set_name == "train" and config.subsample_size is not None:
        total_samples = len(inputs)

        if config.subsample_size < total_samples:
            indices = np.random.choice(
                total_samples,
                size=config.subsample_size,
                replace=False
            )

            inputs = [inputs[i] for i in indices]
            labels = [labels[i] for i in indices]

    # Generate dataset
    results = {
        k: []
        for k in [
            "inputs",
            "labels",
            "puzzle_identifiers",
            "puzzle_indices",
            "group_indices",
        ]
    }

    puzzle_id = 0
    example_id = 0

    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    for inp, out in zip(tqdm(inputs), labels):

        # Dihedral transformations
        for aug_idx in range(
            8 if (set_name == "train" and config.aug) else 1
        ):
            results["inputs"].append(dihedral_transform(inp, aug_idx))
            results["labels"].append(dihedral_transform(out, aug_idx))

            example_id += 1
            puzzle_id += 1

            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)

        results["group_indices"].append(puzzle_id)

    # Character mapping
    assert len(all_chars - set(CHARSET)) == 0

    char2id = np.zeros(256, np.uint8)
    char2id[np.array(list(map(ord, CHARSET)))] = (
        np.arange(len(CHARSET)) + 1
    )

    def _seq_to_numpy(seq):
        return np.vstack(
            [char2id[s.reshape(-1)] for s in seq]
        )

    results = {
        "inputs": _seq_to_numpy(results["inputs"]),
        "labels": _seq_to_numpy(results["labels"]),
        "group_indices": np.array(
            results["group_indices"],
            dtype=np.int32,
        ),
        "puzzle_indices": np.array(
            results["puzzle_indices"],
            dtype=np.int32,
        ),
        "puzzle_identifiers": np.array(
            results["puzzle_identifiers"],
            dtype=np.int32,
        ),
    }

    # Metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=int(math.prod(grid_size)),
        vocab_size=len(CHARSET) + 1,
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        total_puzzles=len(results["group_indices"]) - 1,
        sets=["all"],
    )

    # Save
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)

    for k, v in results.items():
        np.save(
            os.path.join(save_dir, f"all__{k}.npy"),
            v
        )

    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):

    # convert_subset(
    #     "train",
    #     config.train_csv,
    #     config
    # )

    convert_subset(
        "test",
        config.test_csv,
        config
    )


if __name__ == "__main__":
    cli()