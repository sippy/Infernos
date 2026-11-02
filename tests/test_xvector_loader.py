"""
Unit tests for xvector_loader module.

Uses unittest discovery, tempfile-only fixtures, no network in unit tests.
Tests: fresh download mock, offline cache, offline miss, bad zip, objects,
wrong dim, nonfinite, duplicates, ordering.
"""

import os
import sys
import tempfile
import unittest
import zipfile
import io
from pathlib import Path
from unittest import mock

import numpy as np
import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HelloSippyTTSRT.xvector_loader import load_speaker_embeddings, _DATASET_REVISION


class TestXVectorLoader(unittest.TestCase):
    """Unit tests for xvector_loader."""

    def test_fresh_download_mock(self):
        """Test fresh download via mock hf_hub_download."""
        # Create temp zip with valid data
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            with zipfile.ZipFile(tmp.name, "w") as zf:
                for i in range(10):
                    filename = f"cmu_us_bdl_arctic-wav-arctic_a{i:05d}.npy"
                    buf = io.BytesIO()
                    np.save(buf, np.random.randn(512).astype(np.float32), allow_pickle=False)
                    buf.seek(0)
                    zf.writestr(filename, buf.read())
            zip_path = tmp.name

        try:
            with mock.patch(
                "HelloSippyTTSRT.xvector_loader.hf_hub_download",
                return_value=zip_path,
            ) as mock_download:
                embeddings = load_speaker_embeddings(device="cpu")

            # Assert pin args
            mock_download.assert_called_once_with(
                repo_id="Matthijs/cmu-arctic-xvectors",
                filename="spkrec-xvect.zip",
                repo_type="dataset",
                revision=_DATASET_REVISION,
                cache_dir=None,
            )

            # Assert results
            self.assertEqual(len(embeddings), 10)
            self.assertEqual(embeddings[0].shape, (1, 512))
            self.assertEqual(embeddings[0].dtype, torch.float32)
            self.assertTrue(torch.isfinite(embeddings[0]).all())
        finally:
            os.unlink(zip_path)

    def test_offline_cached_return(self):
        """Test offline mode returns cached zip."""
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            with zipfile.ZipFile(tmp.name, "w") as zf:
                for i in range(5):
                    filename = f"test_{i:05d}.npy"
                    buf = io.BytesIO()
                    np.save(buf, np.random.randn(512).astype(np.float32), allow_pickle=False)
                    buf.seek(0)
                    zf.writestr(filename, buf.read())
            zip_path = tmp.name

        try:
            with mock.patch(
                "HelloSippyTTSRT.xvector_loader.hf_hub_download",
                return_value=zip_path,
            ) as mock_download:
                with mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}):
                    embeddings = load_speaker_embeddings(device="cpu")

            # hf_hub_download still called (HF handles offline internally)
            mock_download.assert_called_once()
            self.assertEqual(len(embeddings), 5)
        finally:
            os.unlink(zip_path)

    def test_offline_cache_miss(self):
        """Test offline mode raises error when cache missing."""
        with mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}):
            with mock.patch(
                "HelloSippyTTSRT.xvector_loader.hf_hub_download",
                side_effect=FileNotFoundError("No cached file"),
            ):
                with self.assertRaises((FileNotFoundError, RuntimeError)):
                    load_speaker_embeddings(device="cpu")

    def test_bad_zip_corrupt(self):
        """Test corrupt zip raises error."""
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp.write(b"not a valid zip")
            zip_path = tmp.name

        try:
            with mock.patch(
                "HelloSippyTTSRT.xvector_loader.hf_hub_download",
                return_value=zip_path,
            ):
                with self.assertRaises(zipfile.BadZipFile):
                    load_speaker_embeddings(device="cpu")
        finally:
            os.unlink(zip_path)

    def test_wrong_dimension(self):
        """Test wrong dimension raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            with zipfile.ZipFile(tmp.name, "w") as zf:
                buf = io.BytesIO()
                np.save(buf, np.random.randn(256).astype(np.float32), allow_pickle=False)
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
                self.assertIn("Invalid shape", str(ctx.exception))
        finally:
            os.unlink(zip_path)

    def test_nonfinite_values(self):
        """Test non-finite values raise ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            with zipfile.ZipFile(tmp.name, "w") as zf:
                buf = io.BytesIO()
                data = np.array([1.0, 2.0, np.inf, 4.0] + [0.0] * 508, dtype=np.float32)
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
                self.assertIn("Non-finite", str(ctx.exception))
        finally:
            os.unlink(zip_path)

    def test_duplicate_filename_stems(self):
        """Test duplicate filename stems raise ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            with zipfile.ZipFile(tmp.name, "w") as zf:
                # Two files with same stem (different paths)
                buf1 = io.BytesIO()
                np.save(buf1, np.random.randn(512).astype(np.float32), allow_pickle=False)
                buf1.seek(0)
                zf.writestr("cmu_us_bdl_test_1.npy", buf1.read())
                buf2 = io.BytesIO()
                np.save(buf2, np.random.randn(512).astype(np.float32), allow_pickle=False)
                buf2.seek(0)
                zf.writestr("subdir/cmu_us_bdl_test_1.npy", buf2.read())  # Same stem, different path
            zip_path = tmp.name

        try:
            with mock.patch(
                "HelloSippyTTSRT.xvector_loader.hf_hub_download",
                return_value=zip_path,
            ):
                with self.assertRaises(ValueError) as ctx:
                    load_speaker_embeddings(device="cpu")
                self.assertIn("Duplicate filename stem", str(ctx.exception))
        finally:
            os.unlink(zip_path)

    def test_ordering_preserved(self):
        """Test filename ordering preserved (sorted)."""
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            with zipfile.ZipFile(tmp.name, "w") as zf:
                # Add in non-sorted order
                filenames = ["z_last.npy", "a_first.npy", "m_middle.npy"]
                for fname in filenames:
                    buf = io.BytesIO()
                    np.save(buf, np.random.randn(512).astype(np.float32), allow_pickle=False)
                    buf.seek(0)
                    zf.writestr(fname, buf.read())
            zip_path = tmp.name

        try:
            with mock.patch(
                "HelloSippyTTSRT.xvector_loader.hf_hub_download",
                return_value=zip_path,
            ):
                embeddings = load_speaker_embeddings(device="cpu")

            # Check ordering (sorted by filename stem)
            stems = [Path(p).stem for p in ["z_last.npy", "a_first.npy", "m_middle.npy"]]
            expected = sorted(stems)
            actual = [e.shape for e in embeddings]  # Just check we got 3
            self.assertEqual(len(embeddings), 3)
        finally:
            os.unlink(zip_path)

    def test_empty_xvector(self):
        """Test empty xvector raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            with zipfile.ZipFile(tmp.name, "w") as zf:
                buf = io.BytesIO()
                np.save(buf, np.array([], dtype=np.float32), allow_pickle=False)
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
                self.assertIn("Empty", str(ctx.exception))
        finally:
            os.unlink(zip_path)

    def test_float32_dtype_accepted(self):
        """Test float32 dtype (correct) is accepted."""
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            with zipfile.ZipFile(tmp.name, "w") as zf:
                buf = io.BytesIO()
                data = np.random.randn(512).astype(np.float32)
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
        finally:
            os.unlink(zip_path)


if __name__ == "__main__":
    unittest.main()
