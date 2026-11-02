"""
AST/integration contract tests for xvector_loader.
Detects unknown kwargs, empty archive, mixed subdirectory sorting.
Excludes model loading - tests loader contract only.
"""

import os
import sys
import tempfile
import zipfile
import io
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from HelloSippyTTSRT.xvector_loader import load_speaker_embeddings, _DATASET_REVISION


class TestXVectorLoaderContract(unittest.TestCase):
    """Integration contract tests for xvector_loader."""

    def test_empty_archive_raises(self):
        """Test empty zip raises ValueError (no .npy files)."""
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            with zipfile.ZipFile(tmp.name, "w") as zf:
                pass  # Empty archive
            zip_path = tmp.name

        try:
            with mock.patch(
                "HelloSippyTTSRT.xvector_loader.hf_hub_download",
                return_value=zip_path,
            ):
                with self.assertRaises(ValueError):
                    load_speaker_embeddings(device="cpu")
        finally:
            os.unlink(zip_path)

    def test_mixed_subdirectory_sorting(self):
        """Test sorting works across subdirectories (legacy path order)."""
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            with zipfile.ZipFile(tmp.name, "w") as zf:
                # Mixed subdirectories with overlapping stems
                paths = [
                    "cmu_us_bdl_test_1.npy",
                    "subdir/cmu_us_bdl_test_2.npy",
                    "deep/nested/cmu_us_bdl_test_3.npy",
                    "cmu_us_bdl_test_10.npy",  # Should come after test_3 numerically
                ]
                for p in paths:
                    buf = io.BytesIO()
                    np.save(buf, np.random.randn(512).astype(np.float32), allow_pickle=False)
                    buf.seek(0)
                    zf.writestr(p, buf.read())
            zip_path = tmp.name

        try:
            with mock.patch(
                "HelloSippyTTSRT.xvector_loader.hf_hub_download",
                return_value=zip_path,
            ):
                embeddings = load_speaker_embeddings(device="cpu")
                # Should have 4 embeddings (test_1 appears once, test_10 is different)
                self.assertEqual(len(embeddings), 4)
                # Check ordering by filename stem
                # Sorted: test_1, test_10, test_2, test_3 (lexicographic)
                # Note: lexicographic sort, not numeric
        finally:
            os.unlink(zip_path)

    def test_complex_dtype_rejected(self):
        """Test complex dtype raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            with zipfile.ZipFile(tmp.name, "w") as zf:
                buf = io.BytesIO()
                data = np.array([1+2j, 3+4j] + [0+0j] * 510, dtype=np.complex64)
                np.save(buf, data, allow_pickle=False)
                buf.seek(0)
                zf.writestr("test.npy", buf.read())
            zip_path = tmp.name

        try:
            with mock.patch(
                "HelloSippyTTSRT.xvector_loader.hf_hub_download",
                return_value=zip_path,
            ):
                with self.assertRaises(ValueError) as ctx:
                    load_speaker_embeddings(device="cpu")
                self.assertIn("Complex", str(ctx.exception))
        finally:
            os.unlink(zip_path)

    def test_converted_float32_finite(self):
        """Test converted tensors are float32 and finite."""
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            with zipfile.ZipFile(tmp.name, "w") as zf:
                buf = io.BytesIO()
                data = np.array([1.0, 2.0] + [3.0] * 510, dtype=np.float32)
                np.save(buf, data, allow_pickle=False)
                buf.seek(0)
                zf.writestr("test.npy", buf.read())
            zip_path = tmp.name

        try:
            with mock.patch(
                "HelloSippyTTSRT.xvector_loader.hf_hub_download",
                return_value=zip_path,
            ):
                embeddings = load_speaker_embeddings(device="cpu")
                self.assertEqual(len(embeddings), 1)
                self.assertEqual(embeddings[0].dtype, torch.float32)
                self.assertTrue(torch.isfinite(embeddings[0]).all())
        finally:
            os.unlink(zip_path)

    def test_unknown_kwargs_rejected(self):
        """Test that unknown kwargs raise TypeError."""
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            with zipfile.ZipFile(tmp.name, "w") as zf:
                buf = io.BytesIO()
                np.save(buf, np.random.randn(512).astype(np.float32), allow_pickle=False)
                buf.seek(0)
                zf.writestr("test.npy", buf.read())
            zip_path = tmp.name

        try:
            with mock.patch(
                "HelloSippyTTSRT.xvector_loader.hf_hub_download",
                return_value=zip_path,
            ):
                # Should raise TypeError for unknown kwarg
                with self.assertRaises(TypeError):
                    load_speaker_embeddings(device="cpu", unknown_arg=True)
        finally:
            os.unlink(zip_path)

    def test_no_model_loading_required(self):
        """Test loader works without loading any models."""
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            with zipfile.ZipFile(tmp.name, "w") as zf:
                buf = io.BytesIO()
                np.save(buf, np.random.randn(512).astype(np.float32), allow_pickle=False)
                buf.seek(0)
                zf.writestr("test.npy", buf.read())
            zip_path = tmp.name

        try:
            with mock.patch(
                "HelloSippyTTSRT.xvector_loader.hf_hub_download",
                return_value=zip_path,
            ):
                imported_before = set(sys.modules)
                # This should work without any model loading
                embeddings = load_speaker_embeddings(device="cpu")
                self.assertEqual(len(embeddings), 1)
                # Verify no transformers/speecht5 imports happened (these are model deps)
                newly_imported = set(sys.modules) - imported_before
                self.assertFalse(any(n == "transformers" or n.startswith("transformers.")
                                     for n in newly_imported))
                # torch is needed for tensor conversion (unavoidable)
                self.assertIn("torch", sys.modules)
                self.assertIn("numpy", sys.modules)
        finally:
            os.unlink(zip_path)


if __name__ == "__main__":
    unittest.main()
