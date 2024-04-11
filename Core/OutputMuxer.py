from typing import Union, Dict, List
from time import monotonic

import torch
import torch.nn.functional as F

from .AudioChunk import AudioChunk
from .AStreamMarkers import ASMarkerGeneric, ASMarkerNewSent

class OutputMuxer():
    output_sr:int
    qsize:int
    device:str
    chunks_in: List[Union[AudioChunk, ASMarkerGeneric]]
    def __init__(self, output_sr:int, qsize:int, device:str):
        self.output_sr = output_sr
        self.qsize = qsize
        self.device = device
        self.chunks_in = []

    def chunk_in(self, chunk:Union[AudioChunk, ASMarkerGeneric]):
        if isinstance(chunk, AudioChunk):
            if chunk.samplerate != self.output_sr:
                chunk = chunk.resample(self.output_sr)
            if len(self.chunks_in) > 0 and isinstance(self.chunks_in[-1], AudioChunk):
                chunk.audio = torch.cat((self.chunks_in.pop().audio, chunk.audio), dim=0)
        self.chunks_in.append(chunk)

    def idle(self, rtp_worker):
        chunk_o = torch.empty(0).to(self.device)
        if len(self.chunks_in) == 1 and isinstance(self.chunks_in[0], AudioChunk) and \
          self.chunks_in[0].audio.size(0) < self.qsize:
            return None
        while len(self.chunks_in) > 0 and (rsize:=self.qsize-chunk_o.size(0)) > 0:
            chunk = self.chunks_in[0]
            if isinstance(chunk, ASMarkerNewSent):
                #self.update_frm_ctrs(prcsd_inc=pos.get_buf_nframes())
                if chunk_o.size(0) > 0:
                    return chunk_o
                print(f'{monotonic():4.3f}: ASMarkerNewSent {chunk.on_proc=}')
                self.chunks_in.pop(0)
                chunk.on_proc(rtp_worker)
                continue
            chunk_o = torch.cat((chunk_o, chunk.audio[:rsize]), dim=0)
            if chunk.audio.size(0) > rsize:
                chunk.audio = chunk.audio[rsize:]
            else:
                self.chunks_in.pop(0)
        if chunk_o.size(0) > 0 and chunk_o.size(0) < self.qsize:
            print(f'{monotonic():4.3f}: Reinserting {chunk_o.size()=}')
            self.chunks_in.insert(0, AudioChunk(chunk_o, self.output_sr))
            return None

        return chunk_o if chunk_o.size(0) > 0 else None

class OutputMTMuxer():
    tracks:Dict[int, OutputMuxer]
    output_sr:int
    qsize:int
    device:str
    def __init__(self, output_sr:int, qsize:int, device:str):
        self.tracks = {}
        self.output_sr = output_sr
        self.qsize = qsize
        self.device = device

    def chunk_in(self, chunk:Union[AudioChunk, ASMarkerGeneric]):
        if chunk.track_id not in self.tracks:
            self.tracks[chunk.track_id] = OutputMuxer(self.output_sr, self.qsize, self.device)
        self.tracks[chunk.track_id].chunk_in(chunk)

    def idle(self, rtp_worker):
        chunks = [chunk for chunk in [track.idle(rtp_worker) for track in self.tracks.values()] if chunk is not None]
        if len(chunks) == 0: return None
        if len(chunks) == 1: return chunks[0]
        max_len = max([chunk.size(0) for chunk in chunks])
        chunks = [F.pad(chunk, (0, max_len-chunk.size(0)), "constant", 0) if chunk.size(0) < max_len else chunk
                  for chunk in chunks]
        merged = torch.sum(torch.stack(chunks), dim=0) / len(self.tracks)
        #max_val = torch.max(torch.abs(merged))
        #if max_val > 1:
        #    merged /= max_val
        return merged
