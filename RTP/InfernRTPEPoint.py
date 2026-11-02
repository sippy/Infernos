"""sippy transport with the existing Infernos actor/session interface."""

from functools import partial
from sippy.Rtp.EPoint import RTPEPoint
from sippy.Rtp.Handlers import RTPHandlers
from Core.AStreamMarkers import ASMarkerGeneric, ASMarkerNewSent
from RTP.InfernRTPIngest import RTPInStream
from RTP.RTPOutputWorker import RTPOutputWorker


class InfernRTPEPoint(RTPEPoint):
    def __init__(self, rc, rtp_params, ring, get_direct_soundout):
        handlers = RTPHandlers()
        handlers.writer_cls = partial(RTPOutputWorker, 'cpu')
        handlers.rtp_instream_cls = lambda params, audio_in: RTPInStream(ring, params, get_direct_soundout)
        # Install factories before sippy constructs the initial stream/writer.
        super().__init__(rc, rtp_params, None, handlers=handlers)

    def soundout(self, chunk):
        if isinstance(chunk, ASMarkerNewSent):
            self.firstframe = True
        elif not isinstance(chunk, ASMarkerGeneric):
            self.firstframe = False
        with self.state_lock:
            writer = self.writer
        if writer is not None:
            return writer.soundout(chunk)

    def update(self, rtp_params):
        old_writer = None
        with self.state_lock:
            if self.writer is not None and self.rtp_params.codec is not rtp_params.codec:
                old_writer, self.writer = self.writer, None
        if old_writer is not None:
            old_writer.end()
            old_writer.join()
        super().update(rtp_params)

    def shutdown(self):
        stream = getattr(self, 'rsess', None)
        super().shutdown()
        if stream is not None:
            stream.close()
