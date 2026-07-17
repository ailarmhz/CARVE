#!/usr/bin/env python3
"""
Download CT-CLIP model weights (VocabFine, ClassFine, Text Classifier) from Hugging Face.
See: https://github.com/ibrahimethemhamamci/CT-CLIP
Weights are hosted in the CT-RATE dataset repo.
"""
import argparse
from pathlib import Path

# Default: save under same parent as existing CT-CLIP_v2.pt, or ./weights_multimodel
HF_REPO = "ibrahimhamamci/CT-RATE"
HF_REPO_TYPE = "dataset"

FILES = [
    "models/CT-CLIP-Related/CT_VocabFine_v2.pt",
    "models/CT-CLIP-Related/CT_LiPro_v2.pt",
    "models/RadBertClassifier.pth",
]


def main():
    p = argparse.ArgumentParser(description="Download CT-CLIP VocabFine, ClassFine, Text Classifier weights.")
    p.add_argument(
        "--out-dir",
        type=str,
        default="/datasets/ctrate/CT-CLIP-weights",
        help="Base dir for weights (creates models/CT-CLIP-Related/ and models/ under it)",
    )
    p.add_argument("--dry-run", action="store_true", help="Print would-be download paths only.")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Install: pip install huggingface_hub")
        raise SystemExit(1)

    for hf_path in FILES:
        dest_dir = out_dir  # files will be in out_dir / subpath of hf_path
        if args.dry_run:
            print(f"Would download {hf_path} -> {out_dir}")
            continue
        print(f"Downloading {hf_path} ...")
        try:
            path = hf_hub_download(
                repo_id=HF_REPO,
                filename=hf_path,
                repo_type=HF_REPO_TYPE,
                local_dir=out_dir,
                force_download=False,
            )
            print(f"  -> {path}")
        except Exception as e:
            print(f"  Failed: {e}")
            print(f"  Manual: https://huggingface.co/datasets/{HF_REPO}/tree/main")

    if not args.dry_run:
        print(f"\nWeights under: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
