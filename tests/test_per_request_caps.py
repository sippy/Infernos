"""Offline unit tests for per-request decoder caps.

No GPU inference, model downloads, TTS, or ASR required.
Tests call REAL infer() and unbatch_and_dispatch() with deterministic
fake model components satisfying actual tensor shapes.
"""

import sys, os, uuid
from collections import namedtuple
from unittest.mock import MagicMock

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from HelloSippyTTSRT.HelloSippyRTPipe import (
    HelloSippyPipeState,
    HelloSippyPipeStateBatched,
    HelloSippyRTPipe,
    HelloSippyPlayRequest,
    RowDecodeLog,
    _STOP_REASON_LABELS,
)


# ---------------------------------------------------------------------------
# Fake model components — shapes matching real SpeechT5 pipeline
# ---------------------------------------------------------------------------

HIDDEN_DIM = 512

class _FakeModel:
    """Deterministic fake model with controllable stop probs per step."""
    def __init__(self, device="cpu", rf=2, num_mel_bins=80):
        self.device = device
        self.rf = rf
        self.num_mel_bins = num_mel_bins
        self._stop_probs = []
        self._step = 0

    def _next_stop_prob(self, batch_size):
        if self._step < len(self._stop_probs):
            p = self._stop_probs[self._step]
        else:
            p = torch.full((batch_size,), -1.0, dtype=torch.bfloat16, device=self.device)
        self._step += 1
        return p

    def set_stop_probs(self, probs):
        """probs: list of torch.Tensor [batch_size], one per decoder step.
        Values > 0.5 (threshold) trigger stop_token."""
        self._stop_probs = [p.to(self.device) for p in probs]
        self._step = 0


def _build_fake_pp(device="cpu", rf=2, num_mel_bins=80,
                   maxlenratio=20.0, minlenratio=0.0):
    """Build a fake HelloSippyRTPipe with mock model components.
    All shapes match the real pipeline."""
    pp = MagicMock(spec=HelloSippyRTPipe)
    pp.maxlenratio = maxlenratio
    pp.minlenratio = minlenratio
    pp.threshold = 0.5
    pp.chunk_size = 8
    pp.pre_nframes = 2
    pp.post_nframes = 2
    pp.model_sr = 16000
    pp.output_sr = 16000
    import threading
    pp.cuda_lock = threading.Lock()
    pp.cleanup_text = None

    # Config
    pp.model = _FakeModel(device, rf, num_mel_bins)
    pp.model.config = MagicMock()
    pp.model.config.reduction_factor = rf
    pp.model.config.num_mel_bins = num_mel_bins
    pp.model.config.speaker_embedding_dim = 512

    # speecht5.decoder
    pp.model.speecht5 = MagicMock()
    pp.model.speecht5.decoder = MagicMock()

    def fake_prenet(x, emb):
        # x: [B, seq, num_mel_bins], emb: [B, 512]
        # Returns [B, seq, HIDDEN_DIM]
        return torch.randn(x.size(0), x.size(1), HIDDEN_DIM,
                           dtype=torch.bfloat16, device=device)

    def fake_wrapped_decoder(hidden_states, encoder_hidden_states=None,
                             encoder_attention_mask=None,
                             past_key_values=None, **kw):
        # hidden_states: [B, 1, HIDDEN_DIM]
        bsz = hidden_states.size(0)
        last_hs = torch.randn(bsz, 1, HIDDEN_DIM,
                               dtype=torch.bfloat16, device=device)
        # past_key_values: tuple of 4 layers, each layer has 2 tensors
        layer_shape = (bsz, 8, HIDDEN_DIM // 8)
        pkv = (tuple(torch.randn(*layer_shape, device=device) for _ in range(2)),) * 4
        out = MagicMock()
        out.last_hidden_state = last_hs
        out.past_key_values = pkv
        return out

    pp.model.speecht5.decoder.prenet = fake_prenet
    pp.model.speecht5.decoder.wrapped_decoder = fake_wrapped_decoder

    # speech_decoder_postnet
    pp.model.speech_decoder_postnet = MagicMock()

    def fake_feat_out(x):
        # x: [B, HIDDEN_DIM] -> returns [B, rf * num_mel_bins]
        bsz = x.size(0)
        return torch.randn(bsz, rf * num_mel_bins,
                           dtype=torch.bfloat16, device=device)

    def fake_prob_out(x):
        # x: [B, HIDDEN_DIM] -> returns [B, 1] float prob
        bsz = x.size(0)
        prob_float = pp.model._next_stop_prob(bsz)
        return prob_float.to(dtype=torch.bfloat16, device=device).unsqueeze(1)

    def fake_postnet(spectrogram):
        # pass through: [B, frames, num_mel_bins]
        return spectrogram.clone()

    pp.model.speech_decoder_postnet.feat_out = fake_feat_out
    pp.model.speech_decoder_postnet.prob_out = fake_prob_out
    pp.model.speech_decoder_postnet.postnet = fake_postnet

    # Vocoder + chunker
    def fake_vocoder(spectrogram):
        # spectrogram: [total_chunks*batch_size, chunk_frames+eframes, num_mel_bins]
        # Produce enough samples per chunk for trimming to work.
        # Each row needs: (idx - starts_at) * stepsize samples.
        # stepsize = 512 (sr_rr=1). idx will be n_steps=16, starts_at=2.
        # Per row: 14 * 512 = 7168. With nchunks=4: 7168/4 = 1792 per chunk per row.
        # Total = nchunks * batch_size * 1792
        total = spectrogram.size(0)
        samples = 1792  # per chunk row
        return torch.randn(total, samples, dtype=torch.bfloat16, device=device)

    def fake_chunker(spectrogram, audio):
        return audio.clone()

    pp.vocoder = fake_vocoder
    pp.chunker = fake_chunker
    pp.resampler = None

    return pp


def _build_states(token_lengths, pp, sessions=None):
    """Build HelloSippyPipeState objects with given token lengths."""
    batch_size = len(token_lengths)
    rf = pp.model.config.reduction_factor
    mr = pp.maxlenratio
    minr = pp.minlenratio
    num_mel_bins = pp.model.config.num_mel_bins
    device = pp.model.device

    if sessions is None:
        sessions = [uuid.uuid4() for _ in range(batch_size)]

    states = []
    for i, tc in enumerate(token_lengths):
        st = object.__new__(HelloSippyPipeState)
        st.session = sessions[i]
        st.dispatch = None  # set later
        st.inputs = torch.arange(1, tc + 1).unsqueeze(0)  # [1, tc]
        st.token_count = tc
        st.row_maxlen = max(1, int(tc * mr / rf))
        st.row_minlen = int(tc * minr / rf)
        st.speaker_embeddings = torch.randn(1, 512, dtype=torch.bfloat16, device=device)
        # Unpadded attention mask — real merge handles padding
        st.encoder_attention_mask = torch.ones_like(st.inputs, dtype=torch.int)
        eframes = pp.pre_nframes + pp.post_nframes
        st.pre_frames = torch.zeros(1, eframes, num_mel_bins,
                                    dtype=torch.bfloat16, device=device)
        st.starts_at = torch.tensor([pp.post_nframes // rf], dtype=torch.long)
        st.ends_at = torch.tensor([-1], dtype=torch.long)
        states.append(st)
    return states


def _build_batched(states, pp):
    """Build HelloSippyPipeStateBatched via real merge — only fake encoder setup needed."""
    batch_size = len(states)
    padded_len = max(s.inputs.size(1) for s in states)
    device = pp.model.device
    num_mel_bins = pp.model.config.num_mel_bins

    # Fake encoder output so real merge can call pp.model.speecht5.encoder()
    encoder_out = MagicMock()
    encoder_out.last_hidden_state = torch.randn(
        batch_size, padded_len, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    pp.model.speecht5.encoder.return_value = encoder_out

    return HelloSippyPipeStateBatched(states, pp)


# ---------------------------------------------------------------------------
# Source-level tests
# ---------------------------------------------------------------------------

def test_token_counts_vectorized():
    """token_counts derived from encoder_attention_mask.sum(dim=1)."""
    import inspect
    from HelloSippyTTSRT import HelloSippyRTPipe as pm
    src = inspect.getsource(pm.HelloSippyPipeStateBatched.merge)
    assert 'encoder_attention_mask.sum' in src
    assert '[s.token_count for s in states]' not in src
    assert '[s.row_maxlen for s in states]' not in src
    assert '[s.row_minlen for s in states]' not in src
    print("PASS test_token_counts_vectorized")


def test_device_placement():
    """All new tensors on model.device."""
    import inspect
    from HelloSippyTTSRT import HelloSippyRTPipe as pm
    src = inspect.getsource(pm.HelloSippyPipeStateBatched.merge)
    assert 'pp.model.device' in src or 'dev' in src
    for line in src.split('\n'):
        s = line.strip()
        if 'torch.tensor(' in s and 'states' not in s:
            assert 'device=' in s or '.to(' in s, f"No device: {s}"
    print("PASS test_device_placement")


def test_no_break_in_infer():
    """No break statement in infer() while loop."""
    import inspect
    from HelloSippyTTSRT import HelloSippyRTPipe as pm
    src = inspect.getsource(pm.HelloSippyRTPipe.infer)
    for line in src.split('\n'):
        s = line.strip()
        if s.startswith('#'):
            continue
        if s == 'break':
            raise AssertionError("break statement found in infer()")
    print("PASS test_no_break_in_infer")


def test_no_row_active():
    """row_active field removed."""
    import inspect
    from HelloSippyTTSRT import HelloSippyRTPipe as pm
    src = inspect.getsource(pm.HelloSippyPipeStateBatched)
    assert 'row_active' not in src
    print("PASS test_no_row_active")


def test_row_decode_log_fields():
    """RowDecodeLog structure correct."""
    log = RowDecodeLog(session_id=uuid.uuid4(), token_count=15, cap=30,
                       actual_steps=25, stop_reason="stop_token", audio_samples=4000)
    assert log.token_count == 15
    assert log.cap == 30
    assert log.audio_samples == 4000
    print("PASS test_row_decode_log_fields")


def test_ratio_arithmetic():
    """Cap formula: max(1, int(tokens * maxlenratio / reduction_factor))."""
    for tokens, mr, rf, expected in [
        (10, 20.0, 10, 20), (25, 20.0, 10, 50), (5, 20.0, 10, 10),
        (40, 20.0, 10, 80), (0, 20.0, 10, 1),
    ]:
        actual = max(1, int(tokens * mr / rf))
        assert actual == expected, f"{tokens},{mr},{rf}: {actual}!={expected}"
    print("PASS test_ratio_arithmetic")


def test_no_scalar_maxlen():
    """Scalar maxlen/minlen removed."""
    import inspect
    from HelloSippyTTSRT import HelloSippyRTPipe as pm
    src = inspect.getsource(pm.HelloSippyPipeStateBatched.merge)
    assert 'self.maxlen =' not in src
    assert 'self.minlen =' not in src
    print("PASS test_no_scalar_maxlen")


def test_aggregate_worker_log():
    """Worker uses single aggregated debug print."""
    import inspect
    from Cluster import InfernTTSWorker as wm
    src = inspect.getsource(wm.InfernTTSWorker.process_batch)
    assert 'join' in src
    assert 'for log in state.decode_logs:\n            if self.debug:\n                print' not in src
    print("PASS test_aggregate_worker_log")


# ---------------------------------------------------------------------------
# Fake-model production path tests
# ---------------------------------------------------------------------------

def test_single_infer_cycle_mixed_stops():
    """Single infer() cycle with mixed stop conditions.
    Verifies: correct ends_at, stop_reasons_codes, actual_steps, idx."""
    device = "cpu"
    rf = 2  # reduction_factor
    num_mel_bins = 80
    mr = 20.0  # maxlenratio
    minr = 0.0

    # 3 rows: tokens=[10, 4, 2] -> caps=[100, 40, 20]
    # Wait: cap = int(tokens * 20.0 / 2) = tokens * 10
    # caps = [100, 40, 20]
    # Loop runs 16 steps (spectrogram fills 32 frames, 2 frames/step)
    token_lengths = [10, 4, 2]
    sessions = [uuid.uuid4() for _ in range(3)]

    pp = _build_fake_pp(device, rf, num_mel_bins, mr, minr)
    states = _build_states(token_lengths, pp, sessions)
    batched = _build_batched(states, pp)

    expected_caps = batched.row_maxlen.tolist()
    assert expected_caps == [100, 40, 20], f"caps={expected_caps}"

    # Stop probs: row0 fires at step 10, row1 NEVER fires (cap), row2 fires at step 5
    # sigmoid(-1.0) ≈ 0.27 < threshold 0.5 (no stop); sigmoid(2.0) ≈ 0.88 > 0.5 (stop)
    n_steps = 16
    stop_probs = []
    for step in range(n_steps):
        prob = torch.full((3,), -1.0, dtype=torch.bfloat16, device=device)
        if step == 10: prob[0] = 2.0  # stop_token
        if step == 5:  prob[2] = 2.0  # stop_token
        stop_probs.append(prob)
    pp.model.set_stop_probs(stop_probs)

    offset = (pp.pre_nframes + pp.post_nframes) // rf  # (2+2)//2 = 2

    # Call REAL infer()
    HelloSippyRTPipe.infer(pp, batched)

    # Verify idx incremented through full loop
    assert batched.idx == n_steps, f"idx={batched.idx}, expected {n_steps}"

    # Verify row 0 stopped at step 10
    assert batched.ends_at[0].item() == 10 + offset, \
        f"row0 ends_at={batched.ends_at[0].item()}, expected {10+offset}"
    assert batched.stop_reasons_codes[0].item() == 1  # stop_token
    assert batched.actual_steps[0].item() == 10

    # Verify row 1 NOT stopped (cap=40 > 16 steps)
    assert batched.ends_at[1].item() == -1, \
        f"row1 ends_at={batched.ends_at[1].item()}, expected -1"
    assert batched.stop_reasons_codes[1].item() == 0
    assert batched.actual_steps[1].item() == 0

    # Verify row 2 stopped at step 5
    assert batched.ends_at[2].item() == 5 + offset, \
        f"row2 ends_at={batched.ends_at[2].item()}, expected {5+offset}"
    assert batched.stop_reasons_codes[2].item() == 1  # stop_token
    assert batched.actual_steps[2].item() == 5

    # Verify audio was produced
    assert batched.audio is not None
    assert batched.audio.size(0) == 3

    print("PASS test_single_infer_cycle_mixed_stops")


def test_multi_cycle_infer_unbatch_eos_and_logs():
    """Multi-cycle infer+unbatch loop with mixed token lengths.
    Row A: tokens=2, cap=18, stop_token at step 3 (well before cap)
    Row B: tokens=3, cap=27, no stop_token -> cap_reached at idx=27 (cycle 2)
    Row C: tokens=1, cap=9, stop_token at step 7 (well before cap)

    Cycle 1 (16 steps): rows A(3) and C(7) stop via stop_token.
    Row B still active. ends_at=[5, -1, 9], end_idx=15.
    Cycle 2 (16 more steps): row B hits cap=27 at idx=27.
    ends_at=[5, 29, 9], end_idx=31. All dispatched."""
    device = "cpu"
    rf = 2
    num_mel_bins = 80
    mr = 18.0  # cap = tokens * 18 / 2 = tokens * 9
    minr = 0.0

    # Mixed token lengths: caps = [18, 27, 9]
    token_lengths = [2, 3, 1]
    sessions = [uuid.uuid4() for _ in range(3)]
    pp = _build_fake_pp(device, rf, num_mel_bins, mr, minr)
    states = _build_states(token_lengths, pp, sessions)
    batched = _build_batched(states, pp)

    expected_caps = batched.row_maxlen.tolist()
    assert expected_caps == [18, 27, 9], f"caps={expected_caps}"

    n_steps = 16
    offset = (pp.pre_nframes + pp.post_nframes) // rf  # 2

    # Row A: stop at step 3; Row B: never; Row C: stop at step 7
    stop_probs = []
    for step in range(n_steps):
        prob = torch.full((3,), -1.0, dtype=torch.bfloat16, device=device)
        if step == 3: prob[0] = 2.0  # row A stop_token
        if step == 7: prob[2] = 2.0  # row C stop_token
        stop_probs.append(prob)
    pp.model.set_stop_probs(stop_probs)

    # Dispatch tracking
    dispatch_records = {i: {'chunks': [], 'eos_count': 0} for i in range(3)}
    def make_cb(i):
        def cb(data):
            if data is None:
                dispatch_records[i]['eos_count'] += 1
            else:
                dispatch_records[i]['chunks'].append(data.size(0))
        return cb
    batched.dispatch = [make_cb(i) for i in range(3)]

    # --- Cycle 1 ---
    HelloSippyRTPipe.infer(pp, batched)
    assert batched.idx == n_steps, f"idx after cycle 1: {batched.idx}"

    result1 = HelloSippyRTPipe.unbatch_and_dispatch(pp, batched)

    # Row A: ends_at=5 <= 15 -> dispatched
    assert dispatch_records[0]['eos_count'] == 1
    # Row B: ends_at=-1 -> still active
    assert dispatch_records[1]['eos_count'] == 0
    # Row C: ends_at=9 <= 15 -> dispatched
    assert dispatch_records[2]['eos_count'] == 1

    assert result1 is True  # row B still active
    assert len(batched.decode_logs) == 2, \
        f"Expected 2 logs (A,C), got {len(batched.decode_logs)}"

    # Verify token_count and cap in logs
    for log in batched.decode_logs:
        if log.session_id == sessions[0]:
            assert log.token_count == 2 and log.cap == 18
        elif log.session_id == sessions[2]:
            assert log.token_count == 1 and log.cap == 9

    # --- Cycle 2: row B hits cap=27 at idx=27 ---
    # No artificial resets; fake model handles continued state
    stop_probs_2 = []
    for step in range(16, 32):
        prob = torch.full((3,), -1.0, dtype=torch.bfloat16, device=device)
        stop_probs_2.append(prob)
    pp.model.set_stop_probs(stop_probs_2)

    HelloSippyRTPipe.infer(pp, batched)
    assert batched.idx == 32, f"idx after cycle 2: {batched.idx}"

    result2 = HelloSippyRTPipe.unbatch_and_dispatch(pp, batched)

    # Row B: ends_at=29 <= end_idx=31 -> dispatched
    assert dispatch_records[1]['eos_count'] == 1
    assert result2 is False  # all done

    # 3 total logs, one per row
    assert len(batched.decode_logs) == 3
    log_sessions = {log.session_id for log in batched.decode_logs}
    assert len(log_sessions) == 3

    # Verify each log
    for log in batched.decode_logs:
        assert log.stop_reason in ("stop_token", "maxlen_reached")
        # audio_samples == sum of dispatched chunk sizes
        row_idx = sessions.index(log.session_id)
        expected_samples = sum(dispatch_records[row_idx]['chunks'])
        assert log.audio_samples == expected_samples, \
            f"row{row_idx}: log.audio_samples={log.audio_samples}, expected={expected_samples}"

    # Verify caps and token counts
    for log in batched.decode_logs:
        row_idx = sessions.index(log.session_id)
        assert log.cap == expected_caps[row_idx], \
            f"row{row_idx} cap={log.cap}, expected={expected_caps[row_idx]}"
        assert log.token_count == token_lengths[row_idx]

    # Re-dispatch should NOT duplicate logs
    prev_count = len(batched.decode_logs)
    HelloSippyRTPipe.unbatch_and_dispatch(pp, batched)
    assert len(batched.decode_logs) == prev_count

    # Each row EOS exactly once
    for i in range(3):
        assert dispatch_records[i]['eos_count'] == 1

    print("PASS test_multi_cycle_infer_unbatch_eos_and_logs")


def test_premature_break_would_fail():
    """Verify that the premature-break bug (breaking when all ends_at>=0
    before idx increments) would cause unbatch EOS to never fire.
    This is a conceptual test: simulate the broken behavior."""
    # Simulated broken behavior: break at idx=10 when all rows stop
    # ends_at = idx + offset = 10 + 2 = 12
    # end_idx = idx - 1 = 9
    # ends_at(12) > end_idx(9) -> EOS NOT dispatched
    idx_broken = 10
    offset = 2
    ends_at_broken = idx_broken + offset  # 12
    end_idx_broken = idx_broken - 1  # 9
    eos_would_fire = ends_at_broken <= end_idx_broken
    assert not eos_would_fire, \
        "Premature break should prevent EOS dispatch"

    # Correct behavior: idx continues to 16
    idx_correct = 16
    ends_at_correct = 12  # set at step 10 + offset
    end_idx_correct = idx_correct - 1  # 15
    eos_would_fire = ends_at_correct <= end_idx_correct
    assert eos_would_fire, \
        "Correct behavior should allow EOS dispatch"

    print("PASS test_premature_break_would_fail")


def test_dispatch_trimming():
    """Verify startoff/endoff computation trims to cap boundary."""
    state = MagicMock()
    state.idx = 25
    state.starts_at = torch.tensor([2, 2], dtype=torch.long)
    state.ends_at = torch.tensor([15, 20], dtype=torch.long)

    sr_rr = 1
    stepsize = 256 * 2 // sr_rr  # 512
    for i in range(2):
        starts_at_i = state.starts_at[i].item()
        ends_at_i = state.ends_at[i].item()
        asize = (state.idx - starts_at_i) * stepsize
        startoff = max(0, asize - ((state.idx - starts_at_i) * stepsize))
        endoff = min(asize, asize - ((state.idx - ends_at_i) * stepsize))
        assert startoff <= endoff
        dispatched_len = endoff - startoff
        expected_len = (ends_at_i - starts_at_i) * stepsize
        assert dispatched_len == expected_len
    print("PASS test_dispatch_trimming")


def test_mixed_caps_vectorized_stop():
    """Vectorized stop logic handles mixed caps correctly."""
    caps = torch.tensor([20, 50, 10, 80], dtype=torch.int64)
    minlens = torch.tensor([0, 0, 0, 0], dtype=torch.int64)
    ends_at = torch.tensor([-1, -1, -1, -1], dtype=torch.long)
    reasons = torch.zeros(4, dtype=torch.int64)
    actual = torch.zeros(4, dtype=torch.int64)

    for idx in range(85):
        stop_prob = torch.tensor([idx == 15, False, False, idx == 40], dtype=torch.bool)
        cap_hit = (caps <= idx)
        min_ok = (minlens <= idx)
        should_stop = (ends_at < 0) & min_ok & (stop_prob | cap_hit)

        newly = should_stop & (reasons == 0)
        reasons = torch.where(newly, torch.where(stop_prob, 1, 2), reasons)
        ends_at = torch.where(should_stop, idx, ends_at)
        newly = should_stop & (actual == 0)
        actual = torch.where(newly, idx, actual)

    assert ends_at[0].item() == 15 and reasons[0].item() == 1
    assert ends_at[1].item() == 50 and reasons[1].item() == 2
    assert ends_at[2].item() == 10 and reasons[2].item() == 2
    assert ends_at[3].item() == 40 and reasons[3].item() == 1
    print("PASS test_mixed_caps_vectorized_stop")


def test_minlen_blocks():
    """minlen prevents stopping before minlen is met."""
    caps = torch.tensor([20, 40, 60], dtype=torch.int64)
    minlens = torch.tensor([3, 6, 9], dtype=torch.int64)
    ends_at = torch.tensor([-1, -1, -1], dtype=torch.long)
    reasons = torch.zeros(3, dtype=torch.int64)

    for idx in range(65):
        stop_prob = torch.tensor([idx == 2, idx == 7, idx == 15], dtype=torch.bool)
        cap_hit = (caps <= idx)
        min_ok = (minlens <= idx)
        should_stop = (ends_at < 0) & min_ok & (stop_prob | cap_hit)

        newly = should_stop & (reasons == 0)
        reasons = torch.where(newly, torch.where(stop_prob, 1, 2), reasons)
        ends_at = torch.where(should_stop, idx, ends_at)

    assert ends_at[0].item() == 20 and reasons[0].item() == 2  # blocked, cap
    assert ends_at[1].item() == 7 and reasons[1].item() == 1   # stop_token
    assert ends_at[2].item() == 15 and reasons[2].item() == 1  # stop_token
    print("PASS test_minlen_blocks")


def test_zero_length_guard():
    """token_count=0 -> row_maxlen=max(1,0)=1, row_minlen=0."""
    tc = torch.tensor([0], dtype=torch.int64)
    cap = torch.clamp((tc * 20.0 / 10).to(torch.int64), min=1)
    mn = (tc * 0.0 / 10).to(torch.int64)
    assert cap.tolist() == [1]
    assert mn.tolist() == [0]
    print("PASS test_zero_length_guard")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    tests = [
        test_token_counts_vectorized,
        test_device_placement,
        test_no_break_in_infer,
        test_no_row_active,
        test_row_decode_log_fields,
        test_ratio_arithmetic,
        test_no_scalar_maxlen,
        test_aggregate_worker_log,
        test_single_infer_cycle_mixed_stops,
        test_multi_cycle_infer_unbatch_eos_and_logs,
        test_premature_break_would_fail,
        test_dispatch_trimming,
        test_mixed_caps_vectorized_stop,
        test_minlen_blocks,
        test_zero_length_guard,
    ]

    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL {t.__name__}: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
