"""Download editing datasets or the Wikipedia shards used for frequency counts."""

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm
from torch.hub import download_url_to_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "cf": ("counterfact.json", "https://memit.baulab.info/data/dsets/counterfact.json"),
    "mcf": (
        "multi_counterfact.json",
        "https://memit.baulab.info/data/dsets/multi_counterfact.json",
    ),
    "zsre": (
        "zsre_mend_eval.json",
        "https://memit.baulab.info/data/dsets/zsre_mend_eval.json",
    ),
    "mquake": (
        "MQuAKE-CF-3k-v2.json",
        "https://raw.githubusercontent.com/princeton-nlp/MQuAKE/main/datasets/MQuAKE-CF-3k-v2.json",
    ),
}
WIKIPEDIA_SHARDS = 41


def download_dataset(dataset, data_dir):
    filename, url = DATASETS[dataset]
    destination = data_dir / filename
    if not destination.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        download_url_to_file(url, str(temporary))
        temporary.replace(destination)
    print(destination)


def shard_filenames(start_shard, num_shards):
    if (
        start_shard < 0
        or num_shards <= 0
        or start_shard + num_shards > WIKIPEDIA_SHARDS
    ):
        raise ValueError(
            f"Choose a valid range within the {WIKIPEDIA_SHARDS} Wikipedia shards."
        )
    return [
        f"20231101.en/train-{index:05d}-of-{WIKIPEDIA_SHARDS:05d}.parquet"
        for index in range(start_shard, start_shard + num_shards)
    ]


def download_wikipedia(output_dir, revision, start_shard, num_shards, force):
    filenames = shard_filenames(start_shard, num_shards)
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for remote_path in tqdm(filenames, desc="Downloading Wikipedia shards"):
        destination = output_dir / Path(remote_path).name
        if force or not destination.exists():
            cached_path = hf_hub_download(
                repo_id="wikimedia/wikipedia",
                filename=remote_path,
                repo_type="dataset",
                revision=revision,
            )
            shutil.copy2(cached_path, destination)
        downloaded.append(str(destination))
    manifest = {
        "repo_id": "wikimedia/wikipedia",
        "revision": revision,
        "subset": "20231101.en",
        "start_shard": start_shard,
        "num_shards": num_shards,
        "files": downloaded,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }
    with (output_dir / "manifest.json").open("w") as stream:
        json.dump(manifest, stream, indent=2)
    print(f"Saved {len(downloaded)} shards to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS)
    parser.add_argument("--data_dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--wikipedia",
        action="store_true",
        help="Download Wikipedia instead of editing datasets.",
    )
    parser.add_argument("--wikipedia_dir", type=Path)
    parser.add_argument(
        "--revision", default="main", help="Wikipedia dataset revision."
    )
    parser.add_argument("--start_shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=WIKIPEDIA_SHARDS)
    parser.add_argument(
        "--force", action="store_true", help="Re-copy existing Wikipedia shards."
    )
    args = parser.parse_args()
    datasets = (
        args.datasets
        if args.datasets is not None
        else ([] if args.wikipedia else ["mcf", "zsre", "mquake"])
    )
    for dataset in datasets:
        download_dataset(dataset, args.data_dir)
    if args.wikipedia:
        download_wikipedia(
            args.wikipedia_dir or args.data_dir / "wikipedia/20231101.en",
            args.revision,
            args.start_shard,
            args.num_shards,
            args.force,
        )


if __name__ == "__main__":
    main()
