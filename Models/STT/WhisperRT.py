"""CUDA-first Whisper worker based on faster-whisper for streaming STT."""
from __future__ import annotations

import os
from functools import partial
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from faster_whisper import WhisperModel

from Cluster.InfernBatchedWorker import InfernBatchedWorker
from Cluster.STTSession import STTRequest, STTResult


class WhisperRT(InfernBatchedWorker):
    """Inference worker that runs Whisper via faster-whisper on CUDA."""

    provides: str = "whisper_rt"
    schema: dict = {}
    max_batch_size: int = 2
    max_chunk_duration: float = 24.0
    sample_rate: int = 16000

    def __init__(
        self,
        device: str = "cuda",
        model_size: str | None = None,
        compute_type: str | None = None,
        beam_size: int | None = None,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        self._device = device
        self._beam_size = beam_size or int(os.getenv("INFERNOS_STT_BEAM_SIZE", "1"))
        self._compute_type = compute_type or os.getenv("INFERNOS_STT_COMPUTE", "float16")
        self._model_size = model_size or os.getenv("INFERNOS_STT_MODEL", "medium")
        download_root = Path(cache_dir or os.getenv("INFERNOS_STT_CACHE", "~/.cache/Infernos")).expanduser()
        download_root.mkdir(parents=True, exist_ok=True)
        self._model = WhisperModel(
            self._model_size,
            device=device,
            compute_type=self._compute_type,
            download_root=str(download_root),
        )

    def process_batch(self, wis: List[Tuple[STTRequest, callable, Sequence[int] | None]]) -> None:
        for req, text_cb, context in wis:
            audio = req.chunk.audio
            if isinstance(audio, np.ndarray):
                audio_np = audio.astype(np.float32, copy=False)
            else:
                audio_np = np.asarray(audio, dtype=np.float32)
            segments, info = self._model.transcribe(
                audio_np,
                language=req.lang,
                beam_size=self._beam_size,
                vad_filter=True,
                without_timestamps=not req.timestamps,
                temperature=0.0,
                initial_prompt=None,
            )
            pieces: List[str] = []
            token_trace: List[int] = []
            for seg in segments:
                if seg.text:
                    pieces.append(seg.text.strip())
                if seg.tokens:
                    token_trace.extend(seg.tokens)
            text = " ".join(pieces).strip()
            no_speech_prob = getattr(info, "no_speech_probability", 0.0)
            result = STTResult(text=text, no_speech_prob=no_speech_prob, req=req)
            if context is not None and token_trace:
                # Preserve trailing tokens for prompt continuation (match legacy behaviour).
                new_context = (list(context) + token_trace)[-224:]
                context[:] = new_context
            text_cb(result=result)
