"""Infernos tracks/markers feeding sippy's codec, packetizer and scheduler."""

from queue import Empty, Queue
from sippy.Rtp.OutputWorker import RTPOutputWorker as SippyRTPOutputWorker
from Core.AudioChunk import AudioChunk
from Core.AStreamMarkers import ASMarkerGeneric
from Core.OutputMuxer import OutputMTMuxer
from RTP.AudioBridge import copy_metadata, to_infernos, to_sippy


class RTPOutputWorker(SippyRTPOutputWorker):
    def __init__(self, device, rtp_params):
        self.device = device
        self._input_queue = Queue()
        super().__init__(rtp_params)
        self._mix = OutputMTMuxer(self.samplerate_out, self._out_fsize, device)

    def soundout(self, chunk):
        if not isinstance(chunk, ASMarkerGeneric):
            original = to_infernos(chunk)
            if original.audio.ndim != 1 or original.audio.numel() == 0:
                raise ValueError('RTP audio must be a nonempty mono waveform')
            chunk = copy_metadata(original, AudioChunk(original.audio.detach().to(self.device).clone(),
                                                       original.samplerate))
        self._input_queue.put(chunk)
        return (self._input_queue.qsize(), False)

    def _proc_in(self, now_ns, deadline_ns):
        if self.ended():
            return None
        while True:
            try:
                chunk = self._input_queue.get_nowait()
            except Empty:
                break
            self._mix.chunk_in(chunk)
        # Markers run only after previously encoded audio has drained.
        if len(self._enc_buffer) < self._out_psize and self.data_queue.empty():
            audio = self._mix.idle(self)
            if audio is not None:
                super().soundout(to_sippy(AudioChunk(audio, self.samplerate_out)))
        return super()._proc_in(now_ns, deadline_ns)
