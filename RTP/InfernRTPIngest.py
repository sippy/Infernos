"""sippy jitter/codec handling followed by queued Infernos VAD processing."""

from array import array
from queue import Queue
from threading import Lock
from uuid import UUID
import torch
from rtpsynth.RtpJBuf import RtpJBuf
from sippy.Rtp.Ingest import RTPInStream as SippyRTPInStream
from sippy.Rtp.Codecs.G722 import G722Codec
from Core.AudioChunk import AudioChunk
from Core.InfernWrkThread import InfernWrkThread, RTPWrkTRun
from Core.VAD.SileroVAD import SileroVADWorker, VADChannel
from RTP.AudioBridge import to_infernos
from RTP.AudioInput import AudioInput


class WIStreamAudio:
    def __init__(self, stream, chunk):
        self.stream, self.chunk = stream, chunk


class WIStreamConnect:
    def __init__(self, stream, ain):
        self.stream, self.ain = stream, ain


class WIStreamUpdate:
    def __init__(self, stream):
        self.stream = stream


class _G722IngressCodec(G722Codec):
    """Bridge the installed sippy silence() argument mismatch locally.

    Encoding/decoding remain inherited from sippy. Concealment is sized in the
    codec's native sample rate, not the resampled VAD output rate.
    """

    def silence(self, nframes):
        pcm = array('h', [0]) * self.e2d_frames(nframes, out_srate=self.srate)
        return self.encode(self.make_chunk(pcm, self.srate))


class RTPInStream(SippyRTPInStream):
    def __init__(self, ring, rtp_params, get_direct_soundout):
        self.ring = ring
        self.rtp_params = rtp_params
        self.get_direct_soundout = get_direct_soundout
        self.ain = AudioInput()
        self.ain_lock = Lock()
        self._closed = False
        self._pending_audio = torch.empty(0)
        self.vchan = VADChannel(self.audio_chunk_out, self.vad_chunk_out, None, ring.device)
        super().__init__(rtp_params, self._queue_audio)
        self._configure_codec()
        ring.add_stream(self)

    def _configure_codec(self):
        if type(self.codec) is G722Codec:
            self.codec = _G722IngressCodec()
        self.codec.out_srate = self.rtp_params.out_sr

    def _queue_audio(self, chunk):
        if not self._closed:
            self.ring.pkt_queue.put(WIStreamAudio(self, chunk))

    def stream_connect(self, ain):
        def resolve(callback):
            return self.get_direct_soundout(callback) if isinstance(callback, UUID) else callback
        connected = AudioInput(audio_in=resolve(ain.audio_in), vad_chunk_in=resolve(ain.vad_chunk_in))
        self.ring.pkt_queue.put(WIStreamConnect(self, connected))

    def stream_update(self):
        with self.ring_lock:
            self.jbuf = RtpJBuf(self.jb_size)
            self.last_output_lseq = None
            self.out_buffer = b''
            self.ob_rtime = 0.0
            self.codec = self.rtp_params.codec()
            self._configure_codec()
            self.out_chunk_sz_samples = self.rtp_params.out_sr // 10
            # Preserve reset/PCM ordering against concurrent packet callbacks.
            self.ring.pkt_queue.put(WIStreamUpdate(self))

    def _proc_in_tread(self, wi, svad):
        if self._closed:
            return
        if isinstance(wi, WIStreamConnect):
            with self.ain_lock:
                self.ain = wi.ain
            return
        if isinstance(wi, WIStreamUpdate):
            self._pending_audio = torch.empty(0)
            return
        chunk = to_infernos(wi.chunk)
        if chunk.samplerate != svad.input_sr:
            chunk.resample(svad.input_sr)
        self._pending_audio = torch.cat((self._pending_audio, chunk.audio.cpu()))
        size = svad.window_size_samples
        while self._pending_audio.numel() >= size:
            audio, self._pending_audio = self._pending_audio[:size], self._pending_audio[size:]
            svad.infer((self.vchan, AudioChunk(audio, svad.input_sr)))

    def audio_chunk_out(self, chunk, active):
        chunk.active = active
        with self.ain_lock:
            callback = None if self._closed else self.ain.audio_in
        if callback is not None:
            callback(chunk=chunk)

    def vad_chunk_out(self, chunk):
        with self.ain_lock:
            callback = None if self._closed else self.ain.vad_chunk_in
        if callback is not None:
            callback(chunk=chunk)

    def close(self):
        if not self._closed:
            self._closed = True
            self.ring.remove_stream(self)
            with self.ain_lock:
                self.ain = AudioInput()


class InfernRTPIngest(InfernWrkThread):
    debug = False

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.pkt_queue = Queue()
        self.streams = []

    def add_stream(self, stream):
        with self.state_lock:
            self.streams.append(stream)

    def remove_stream(self, stream):
        with self.state_lock:
            if stream in self.streams:
                self.streams.remove(stream)

    def start(self):
        self._start_queue = Queue()
        super().start()
        result = self._start_queue.get()
        del self._start_queue
        if isinstance(result, Exception):
            super().join()
            raise result

    def stop(self):
        self.pkt_queue.put(None)
        super().stop()

    def run(self):
        self.thread_started()
        try:
            svad = SileroVADWorker(self.device)
            svad.start()
        except Exception as error:
            self._start_queue.put(error)
            return
        self._start_queue.put(None)
        try:
            while self.get_state() == RTPWrkTRun:
                wi = self.pkt_queue.get()
                if wi is None:
                    break
                wi.stream._proc_in_tread(wi, svad)
        finally:
            svad.stop()

    def dprint(self, *args):
        if self.debug:
            print(*args)
