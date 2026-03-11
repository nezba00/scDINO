"""
inference.py

Runs embedding inference on multi-channel TIFF crops using a pretrained
Vision Transformer (scDINO). Outputs embeddings + metadata to a JSONL file.

Usage:
    # Run with a config file (recommended)
    python inference.py --config inference_config.yaml

    # Run purely from the command line
    python inference.py \
        --root_crop_dir /path/to/crops \
        --checkpoint    /path/to/checkpoint.pth \
        --output_file   results.jsonl \
        --window_size   32 \
        --batch_size    512 \
        --num_workers   8 \
        --chunk_size    200

    # Mix both: config file as base, CLI flags override specific values
    python inference.py --config inference_config.yaml --batch_size 256

Priority (highest → lowest): CLI flags > config file > argparse defaults
"""

import argparse
import gc
import glob
import os
import sys
import datetime
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import tifffile as tiff
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pyscripts.vision_transformer import VisionTransformer

# ---------------------------------------------------------------------------
# ViT implementation from scDINO which itself mostly copied from TIMM
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Normalisation
#
# Training pipeline (from main_dino.py / utils.py) applies two steps:
#   1. Per-sample, per-channel min-max → [0, 1]   (normalize_numpy_0_to_1)
#   2. Dataset-level z-normalisation              (transforms.Normalize)
#
# Both steps must be reproduced here in the same order to match training.
#
# Fallback constants below are used when channel_mean / channel_std are not
# supplied via config or CLI. They are CPU tensors and must stay on CPU —
# the transform runs inside DataLoader workers before data reaches the GPU.
# ---------------------------------------------------------------------------

_FALLBACK_MEAN = [
    0.32832240462947454,
    0.32837346910405970,
    0.32838536478574950,
    0.32840581104339045,
    0.32842572009257553,
]

_FALLBACK_STD = [
    0.24012219387278322,
    0.24007172276960517,
    0.24000436388447063,
    0.23999389767263468,
    0.23998388524795460,
]


def make_normalizer(mean: List[float], std: List[float]):
    """
    Return a two-step normalisation callable that mirrors the scDINO training
    pipeline:
        Step 1 — per-sample, per-channel min-max to [0, 1]
        Step 2 — dataset-level z-normalisation with precomputed mean / std

    The closure tensors are kept on CPU and are safe to use inside DataLoader
    worker processes (never call .to(device) on them).
    """
    t_mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
    t_std  = torch.tensor(std,  dtype=torch.float32).view(-1, 1, 1)

    def normalize_channels(x: torch.Tensor) -> torch.Tensor:
        """x: (C, H, W) float32 tensor on CPU."""
        # Step 1: per-sample, per-channel min-max → [0, 1]
        # clamp(min=1e-6) guards against flat channels (max == min),
        # matching the zero-check in utils.normalize_numpy_0_to_1.
        x_min = x.flatten(1).min(dim=1).values.view(-1, 1, 1)
        x_max = x.flatten(1).max(dim=1).values.view(-1, 1, 1)
        x = (x - x_min) / (x_max - x_min).clamp(min=1e-6)

        # Step 2: dataset-level z-normalisation
        return (x - t_mean) / t_std

    return normalize_channels


# ---------------------------------------------------------------------------
# Config / argument parsing
# ---------------------------------------------------------------------------

def load_yaml_config(path: str) -> dict:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    return cfg or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="scDINO inference: extract ViT embeddings from crop TIFFs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file (optional)
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a YAML config file. CLI flags override config values.",
    )

    # Paths
    parser.add_argument("--root_crop_dir",  default=None,
                        help="Root directory containing crop TIFFs and metadata CSVs")
    parser.add_argument("--checkpoint",     default=None,
                        help="Path to the pretrained checkpoint (.pth)")
    parser.add_argument("--output_file",    default="features.jsonl",
                        help="Output JSONL file for embeddings")
    parser.add_argument("--metadata_cache", default=None,
                        help="Optional path to cache/load the metadata lookup pickle")

    # Model / data
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--batch_size",  type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--chunk_size",  type=int, default=None,
                        help="Flush to disk every N batches (controls peak RAM usage)")

    # Normalisation — variable number of floats, must match in_chans
    parser.add_argument("--channel_mean", type=float, nargs="+", default=None,
                        metavar="M",
                        help="Per-channel mean (one value per channel). "
                             "Overrides config and built-in defaults.")
    parser.add_argument("--channel_std",  type=float, nargs="+", default=None,
                        metavar="S",
                        help="Per-channel std (one value per channel). "
                             "Overrides config and built-in defaults.")

    # Metadata path patching
    parser.add_argument("--patch_old", default=None,
                        help="Token to replace in metadata file_paths (omit to skip)")
    parser.add_argument("--patch_new", default=None,
                        help="Replacement token for --patch_old")

    cli = parser.parse_args()

    # --- Merge: defaults < YAML config < CLI ---
    defaults = {
        "root_crop_dir":  None,
        "checkpoint":     None,
        "output_file":    "features.jsonl",
        "metadata_cache": None,
        "window_size":    32,
        "batch_size":     512,
        "num_workers":    8,
        "chunk_size":     200,
        "channel_mean":   None,
        "channel_std":    None,
        "patch_old":      None,
        "patch_new":      None,
    }

    cfg = {}
    if cli.config:
        cfg = load_yaml_config(cli.config)
        print(f"Loaded config from {cli.config}")

    # Start from defaults, overlay YAML, overlay explicit CLI flags
    merged = {**defaults, **cfg}
    for key, cli_val in vars(cli).items():
        if key == "config":
            continue
        if cli_val is not None:
            merged[key] = cli_val

    # Validate required fields
    missing = [k for k in ("root_crop_dir", "checkpoint") if not merged.get(k)]
    if missing:
        parser.error(
            f"The following required arguments are missing: {missing}. "
            "Provide them via --config or as CLI flags."
        )

    return argparse.Namespace(**merged)


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def get_in_chans_from_checkpoint(checkpoint_path: str) -> int:
    """
    Infer the number of input channels from the patch embedding weights stored
    in the checkpoint.  Mirrors get_pretrained_weights_in_chans() in utils.py.

    The relevant weight has shape (embed_dim, in_chans, patch_h, patch_w), so
    index 1 gives in_chans.
    """
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # scDINO checkpoints nest weights under 'teacher' → 'backbone.*'
    if "teacher" in state_dict:
        sd = state_dict["teacher"]
        key = "backbone.patch_embed.proj.weight"
    else:
        sd = state_dict
        key = "patch_embed.proj.weight"

    if key not in sd:
        # Try stripping prefixes as a fallback
        sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}
        key = "patch_embed.proj.weight"

    if key not in sd:
        print(f"Could not find '{key}' in checkpoint — defaulting to in_chans=5.")
        return 5

    in_chans = sd[key].shape[1]
    print(f"Auto-detected in_chans={in_chans} from checkpoint.")
    return in_chans


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def load_metadata_lookup(
    root_dir: Optional[str] = None,
    metadata_paths: Optional[Union[str, List[str]]] = None,
    metadata_glob: str = "metadata*.csv",
    cache_path: Optional[str] = None,
) -> pd.Series:
    """
    Load and concatenate all metadata CSVs and return a lookup Series:
        index  = file_path (absolute)
        values = tuple (track_id, t_start, y_center, x_center, filename)

    Results are cached to disk when *cache_path* is provided.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached metadata lookup from {cache_path}")
        return pd.read_pickle(cache_path)

    if metadata_paths is None and root_dir is not None:
        pattern = os.path.join(root_dir, "**", metadata_glob)
        metadata_paths = sorted(glob.glob(pattern, recursive=True))
        print(f"Auto-discovered {len(metadata_paths)} metadata files")
    elif isinstance(metadata_paths, str):
        metadata_paths = [metadata_paths]

    if not metadata_paths:
        print("Warning: No metadata files found – returning empty lookup")
        return pd.Series({}, dtype=object)

    dfs = []
    for p in tqdm(metadata_paths, desc="Loading metadata files"):
        try:
            df = pd.read_csv(p)
        except Exception as exc:
            raise RuntimeError(f"Error reading {p}: {exc}") from exc

        required = {"file_path", "track_id", "t_start", "y_center", "x_center"}
        if not required.issubset(df.columns):
            print(f"  Skipping {p}: missing columns {required - set(df.columns)}")
            continue
        dfs.append(df)

    if not dfs:
        return pd.Series({}, dtype=object)

    metadata_df = pd.concat(dfs, ignore_index=True)
    metadata_df["file_path"] = metadata_df["file_path"].apply(os.path.realpath)
    metadata_df["centroid"] = metadata_df.apply(
        lambda row: (
            int(row["track_id"]),
            int(row["t_start"]),
            float(row["y_center"]),
            float(row["x_center"]),
            str(row.get("filename", os.path.basename(row["file_path"]))),
        ),
        axis=1,
    )

    lookup = metadata_df.set_index("file_path")["centroid"]

    if cache_path:
        lookup.to_pickle(cache_path)
        print(f"Cached metadata lookup → {cache_path} ({len(lookup)} entries)")

    return lookup


def patch_metadata_paths(
    root_dir: str,
    old_token: str,
    new_token: str,
    metadata_glob: str = "metadata*.csv",
) -> None:
    """
    In-place replace *old_token* with *new_token* in the file_path column of
    every metadata CSV found under *root_dir*.  Useful when data was moved.
    """
    csv_paths = glob.glob(os.path.join(root_dir, "**", metadata_glob), recursive=True)
    patched_files = patched_rows = 0
    for p in tqdm(csv_paths, desc="Patching metadata paths"):
        real_p = os.path.realpath(p)
        df = pd.read_csv(real_p)
        mask = df["file_path"].str.contains(old_token, na=False)
        if mask.any():
            df.loc[mask, "file_path"] = df.loc[mask, "file_path"].str.replace(
                old_token, new_token, regex=False
            )
            df.to_csv(real_p, index=False)
            patched_files += 1
            patched_rows += mask.sum()
            print(f"  Patched {mask.sum():>6} rows in {os.path.basename(real_p)}")
    print(f"Done: {patched_rows} rows across {patched_files} files")


def collect_crop_paths(
    root_dir: str,
    metadata_lookup: Optional[pd.Series] = None,
    metadata_paths: Optional[Union[str, List[str]]] = None,
    metadata_glob: str = "metadata*.csv",
) -> list:
    """Return [(abs_path, centroid_tuple), …] without loading image data."""
    if metadata_lookup is None:
        metadata_lookup = load_metadata_lookup(
            root_dir=root_dir,
            metadata_paths=metadata_paths,
            metadata_glob=metadata_glob,
        )

    print(f"Metadata lookup: {len(metadata_lookup)} entries")
    all_tiffs = glob.glob(os.path.join(root_dir, "**", "*.tif*"), recursive=True)
    print(f"Found {len(all_tiffs)} TIFFs")

    samples = []
    for file_path in tqdm(all_tiffs, desc="Collecting paths"):
        abs_path = os.path.realpath(file_path)
        centroid = metadata_lookup.get(abs_path, (-1, 0, 0, 0, "Unknown"))
        samples.append((abs_path, centroid))

    return samples


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MultiChannelDataset(Dataset):
    """
    Reads multi-channel TIFF crops and returns:
        x            – (C, H, W) float32 tensor
        numeric_meta – (4,)      float32 tensor  [track_id, t, y, x]
        file_path    – str
        timelapse_id – str

    The number of channels C is determined by the TIFF on disk and must match
    the in_chans the model was trained with.

    Note: *window_size* is stored for reference but the crop size is determined
    by the TIFF on disk; no runtime cropping/padding is applied here.
    """

    def __init__(self, samples: list, transform=None, window_size: int = 32):
        self.samples = samples
        self.transform = transform
        self.window_size = window_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, centroid = self.samples[idx]
        numeric_metadata = centroid[:4]
        timelapse_id = centroid[4]

        try:
            crop = tiff.imread(file_path)  # expected shape: (H, W, C)
            imgs = [crop[:, :, i] for i in range(crop.shape[2])]
            x = torch.stack(
                [torch.from_numpy(im.astype(np.float32)) for im in imgs], dim=0
            )
        except Exception as exc:
            raise RuntimeError(f"Error reading {file_path}: {exc}") from exc

        if self.transform:
            x = self.transform(x)

        return (
            x,
            torch.tensor(numeric_metadata, dtype=torch.float32),
            file_path,
            timelapse_id,
        )


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(
    checkpoint_path: str,
    in_chans: int,
    window_size: int = 32,
) -> VisionTransformer:
    """
    Load a pretrained scDINO ViT-Small backbone.

    *in_chans* is auto-detected from the checkpoint by get_in_chans_from_checkpoint()
    and passed in here explicitly so the architecture always matches the weights.
    """
    model = VisionTransformer(
        patch_size=4,
        in_chans=in_chans,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=torch.nn.LayerNorm,
        num_classes=0,
        img_size=[window_size],
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("teacher", checkpoint)
    clean_dict = {
        k.replace("module.", "").replace("backbone.", ""): v
        for k, v in state_dict.items()
    }

    result = model.load_state_dict(clean_dict, strict=False)
    if result.missing_keys:
        print(f"  Missing keys  ({len(result.missing_keys)}): {result.missing_keys}")
    if result.unexpected_keys:
        print(f"  Unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys}")

    model.eval()
    print(f"Model loaded from {checkpoint_path}  (in_chans={in_chans})")
    return model


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

def run_inference(
    model: VisionTransformer,
    loader: DataLoader,
    output_file: str,
    chunk_size_batches: int = 200,
) -> None:
    """
    Run embedding inference over *loader* and stream results to *output_file*
    (JSONL format) in chunks to keep RAM usage bounded.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Running inference on {device}")

    chunk_embs: list = []
    chunk_centroids: list = []
    chunk_paths: list = []
    chunk_timelapse_ids: list = []

    def _flush_chunk() -> None:
        data = {
            "track_id":  [int(c[0]) for c in chunk_centroids],
            "t":         [int(c[1]) for c in chunk_centroids],
            "y":         [int(c[2]) for c in chunk_centroids],
            "x":         [int(c[3]) for c in chunk_centroids],
            "embedding": [e.tolist() for e in chunk_embs],
            "path":      chunk_paths,
            "filename":  chunk_timelapse_ids,
        }
        df = pd.DataFrame(data)[
            ["track_id", "t", "y", "x", "embedding", "path", "filename"]
        ]
        with open(output_file, "a") as fh:
            fh.write(df.to_json(orient="records", lines=True))
        del df, data

    flush_threshold = chunk_size_batches * loader.batch_size

    model.eval()
    with torch.no_grad():
        for x_batch, cent_batch, path_batch, timelapse_batch in tqdm(
            loader, desc="Inference"
        ):
            embs = model(x_batch.to(device)).cpu().numpy()
            chunk_embs.extend(embs)
            chunk_centroids.extend(cent_batch.numpy())
            chunk_paths.extend(path_batch)
            chunk_timelapse_ids.extend(timelapse_batch)

            if len(chunk_embs) >= flush_threshold:
                _flush_chunk()
                chunk_embs.clear()
                chunk_centroids.clear()
                chunk_paths.clear()
                chunk_timelapse_ids.clear()
                gc.collect()

    # Flush any remaining samples
    if chunk_embs:
        print(f"Saving final chunk ({len(chunk_embs)} samples)…")
        _flush_chunk()

    print(f"Inference complete. Results written to {output_file}")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Handle output file collision
    if os.path.exists(args.output_file):
        base, ext = os.path.splitext(args.output_file)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        args.output_file = f"{base}_{timestamp}{ext}"
        print(f"Output file exists; using unique name: {args.output_file}")

    # Auto-detect in_chans from the checkpoint
    in_chans = get_in_chans_from_checkpoint(args.checkpoint)

    # Resolve and validate normalisation constants
    mean = args.channel_mean if args.channel_mean is not None else _FALLBACK_MEAN
    std  = args.channel_std  if args.channel_std  is not None else _FALLBACK_STD

    if args.channel_mean is None and args.channel_std is None:
        print(
            f"No channel_mean/std provided — using built-in fallback constants "
            f"({len(_FALLBACK_MEAN)} channels)."
        )

    if len(mean) != in_chans or len(std) != in_chans:
        print(
            f"ERROR: channel_mean has {len(mean)} values and channel_std has "
            f"{len(std)} values, but the checkpoint expects in_chans={in_chans}. "
            "Please provide the correct per-channel statistics."
        )
        sys.exit(1)

    print(
        f"Normalisation — step 1: per-sample min-max → [0,1]  "
        f"step 2: z-norm with mean={mean}  std={std}"
    )
    normalize = make_normalizer(mean, std)

    # Optionally patch stale paths in metadata CSVs
    if args.patch_old and args.patch_new:
        patch_metadata_paths(
            root_dir=args.root_crop_dir,
            old_token=args.patch_old,
            new_token=args.patch_new,
        )

    # Build metadata lookup
    metadata_lookup = load_metadata_lookup(
        root_dir=args.root_crop_dir,
        cache_path=args.metadata_cache,
    )
    print(f"Total metadata entries: {len(metadata_lookup)}")

    # Collect paths and build dataset / loader
    samples = collect_crop_paths(
        root_dir=args.root_crop_dir,
        metadata_lookup=metadata_lookup,
    )
    dataset = MultiChannelDataset(
        samples,
        transform=normalize,
        window_size=args.window_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    # Build model and run
    model = build_model(args.checkpoint, in_chans=in_chans, window_size=args.window_size)
    run_inference(
        model=model,
        loader=loader,
        output_file=args.output_file,
        chunk_size_batches=args.chunk_size,
    )


if __name__ == "__main__":
    main()