from zlib import compress

class VADResult():
    chunk: bytes
    active: bool = True

class ZlibVAD():
    vad_duration: float = 0.1
    vad_threshold: float = 0.6
    vad_frames: int
    max_vad_frames: int
    vad_buffer: bytes = b''
    chunk_buffer: bytes = b''
    ninactive: int = 0
    activation_threshold: int = 5
    def __init__(self, input_sr: int = 8000):
        self.vad_frames = int(input_sr * self.vad_duration)
        self.max_vad_frames = input_sr * 30 # 30 seconds for Whisper

    def ingest(self, data: bytes, vad_chunk_in: callable):
        self.vad_buffer += data
        if len(self.vad_buffer) < self.vad_frames:
            return None
        chunk = self.vad_buffer[:self.vad_frames]
        self.vad_buffer = self.vad_buffer[self.vad_frames:]
        r = len(compress(chunk))/len(chunk)
        v = VADResult()
        active = False if r < self.vad_threshold else True
        vad_chunk_in(chunk, active)
        max_len_reached = len(self.chunk_buffer) >= (self.max_vad_frames - (self.vad_frames * self.activation_threshold))
        if active:
            self.ninactive = 0
            if not max_len_reached:
                self.chunk_buffer += chunk
                return None
            v.chunk = self.chunk_buffer[:self.max_vad_frames]
            self.chunk_buffer = self.chunk_buffer[self.max_vad_frames:]
            return v
        else:
            if self.ninactive > self.activation_threshold:
                assert len(self.chunk_buffer) > self.vad_frames * self.activation_threshold
                chunk = self.chunk_buffer[:-self.vad_frames*self.activation_threshold]
                if len(chunk) < self.vad_frames * self.activation_threshold:
                    v = None
                else:
                    v.chunk = chunk
                self.chunk_buffer = b''
                self.ninactive = 0
                return v
            self.chunk_buffer += chunk
            self.ninactive += 1
        return None