"""
X-vector speaker embedding loader for cmu-arctic-xvectors dataset.

Script-free, datasets5-compatible loader that directly downloads and parses
pre-extracted .npy files from HuggingFace Hub, avoiding custom dataset scripts.

Source: Matthijs/cmu-arctic-xvectors@5c1297a9eb6c91714ea77c0d4ac5aca9b6a952e5
File: spkrec-xvect.zip (7931 utterances, 512-dim x-vectors)
License: MIT
"""

import os
import zipfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from huggingface_hub import hf_hub_download


# Verified immutable commit SHA for cmu-arctic-xvectors dataset
_DATASET_REVISION = "5c1297a9eb6c91714ea77c0d4ac5aca9b6a952e5"
_XVECTOR_DIM = 512


def load_speaker_embeddings(
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    cache_dir: Optional[str] = None,
) -> List[torch.Tensor]:
    """
    Load speaker embeddings from cmu-arctic-xvectors dataset.

    Downloads spkrec-xvect.zip from HuggingFace Hub and extracts .npy files.
    Let HF manage cache via HF_HUB_OFFLINE=1 and local_files_only config.

    Args:
        device: Target device for tensors ("cpu", "cuda", etc.)
        dtype: Target dtype for tensors (default: torch.float32)
        cache_dir: Optional cache directory (let HF manage by default)

    Returns:
        List of torch.Tensor, each shape [1, 512], sorted by filename.

    Raises:
        ValueError: If data is malformed (wrong shape, non-finite, duplicates)

    Example:
        >>> from HelloSippyTTSRT.xvector_loader import load_speaker_embeddings
        >>> embeddings = load_speaker_embeddings(device='cpu')
        >>> len(embeddings)  # 7931
        >>> embeddings[0].shape  # torch.Size([1, 512])
    """
    # Download zip from verified commit
    zip_path = hf_hub_download(
        repo_id="Matthijs/cmu-arctic-xvectors",
        filename="spkrec-xvect.zip",
        repo_type="dataset",
        revision=_DATASET_REVISION,
        cache_dir=cache_dir,
    )

    # Extract and validate .npy files
    xvectors = []
    seen_stems = set()

    with zipfile.ZipFile(zip_path, "r") as zf:
        npy_files = sorted([n for n in zf.namelist() if n.endswith(".npy")])

        for npy_path in npy_files:
            filename = Path(npy_path).stem

            # Reject duplicate stems
            if filename in seen_stems:
                raise ValueError(f"Duplicate filename stem: {filename}")
            seen_stems.add(filename)

            # Load xvector with allow_pickle=False (data is safe)
            with zf.open(npy_path) as f:
                xvector = np.load(f, allow_pickle=False)

            # Validate: non-empty, numeric, correct shape
            if xvector.size == 0:
                raise ValueError(f"Empty xvector: {filename}")
            if not np.issubdtype(xvector.dtype, np.number):
                raise ValueError(f"Non-numeric xvector: {filename}, dtype={xvector.dtype}")
            if xvector.shape != (_XVECTOR_DIM,):
                raise ValueError(
                    f"Invalid shape for {filename}: {xvector.shape}, expected ({_XVECTOR_DIM},)"
                )
            # Reject complex dtype (has imaginary component)
            if np.issubdtype(xvector.dtype, np.complexfloating):
                raise ValueError(f"Complex dtype not allowed: {filename}, dtype={xvector.dtype}")
            if not np.isfinite(xvector).all():
                raise ValueError(f"Non-finite values in {filename}")

            xvectors.append((filename, xvector))

    if not xvectors:
        raise ValueError("Archive contains no x-vector .npy files")

    # Sort by filename stem after reading (preserve legacy ordering)
    xvectors.sort(key=lambda x: x[0])

    # Convert to torch tensors and verify float32 finite
    embeddings = []
    for _, xv in xvectors:
        emb = torch.tensor(xv, dtype=dtype, device=device).unsqueeze(0)
        # Verify converted to float32 and finite
        if emb.dtype != torch.float32:
            raise RuntimeError(f"Embedding dtype mismatch: {emb.dtype}, expected float32")
        if not torch.isfinite(emb).all():
            raise RuntimeError(f"Non-finite after conversion")
        embeddings.append(emb)

    return embeddings
