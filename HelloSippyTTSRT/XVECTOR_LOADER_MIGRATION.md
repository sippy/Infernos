# X-Vector Loader Migration

## Summary

Replaced `load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")` with script-free loader using verified commit SHA.

## Source Pin

- **Repo:** `Matthijs/cmu-arctic-xvectors`
- **Revision:** `5c1297a9eb6c91714ea77c0d4ac5aca9b6a952e5`
- **File:** `spkrec-xvect.zip`
- **License:** MIT
- **Contents:** 7931 `.npy` files, 512-dim x-vectors

## Changes

### HelloSippyRTPipe.py
```python
# Before:
from datasets import load_dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
self.speaker_embeddings = [torch.tensor(ed["xvector"], device='cpu').unsqueeze(0)
                           for ed in sorted(embeddings_dataset, key=lambda x: x['filename'])]

# After:
from HelloSippyTTSRT.xvector_loader import load_speaker_embeddings
self.speaker_embeddings = load_speaker_embeddings(device='cpu')
```

## API

```python
from HelloSippyTTSRT.xvector_loader import load_speaker_embeddings

embeddings = load_speaker_embeddings(
    device='cpu',      # target device
    dtype=torch.float32,  # target dtype
    cache_dir=None,    # let HF manage cache
)
# Returns: List[torch.Tensor], each shape [1, 512], sorted by filename
```

## Validation

- Rejects wrong shape (must be 512)
- Rejects non-numeric dtypes
- Rejects non-finite values (inf/nan)
- Rejects empty arrays
- Rejects duplicate filename stems
- Preserves filename ordering (sorted)

## Offline Mode

```bash
export HF_HUB_OFFLINE=1
python3 -c "from HelloSippyTTSRT.xvector_loader import load_speaker_embeddings; load_speaker_embeddings()"
```

## Tests

```bash
cd /home/sobomax/projects/Infernos
python3 -m pytest tests/test_xvector_loader.py -v
```

## Dependencies

No new dependencies. Uses existing:
- `numpy`
- `torch`
- `huggingface_hub` (via transformers/datasets)
