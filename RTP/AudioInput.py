from typing import Optional

class AudioInput():
    vad_chunk_in:Optional[callable]
    audio_in:Optional[callable]
    def __init__(self, audio_in:Optional[callable]=None, vad_chunk_in:Optional[callable]=None):
        self.vad_chunk_in = vad_chunk_in
        self.audio_in = audio_in
