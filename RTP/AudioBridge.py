"""PCM conversion at the Infernos/sippy boundary."""

from array import array
import torch
from sippy.Rtp.Core.AudioChunk import AudioChunk as SippyAudioChunk
from Core.AudioChunk import AudioChunk


def copy_metadata(source, target):
    for name in ('track_id', 'active', 'debug', 'rtime'):
        if hasattr(source, name):
            setattr(target, name, getattr(source, name))
    return target


def to_sippy(chunk):
    if isinstance(chunk, SippyAudioChunk):
        return chunk
    if not isinstance(chunk, AudioChunk):
        raise TypeError('Expected an Infernos or sippy AudioChunk')
    pcm = torch.clamp(chunk.audio.detach().float() * 32768.0, -32768, 32767).to(torch.int16)
    return copy_metadata(chunk, SippyAudioChunk(array('h', pcm.cpu().reshape(-1).tolist()), chunk.samplerate))


def to_infernos(chunk):
    if isinstance(chunk, AudioChunk):
        return chunk
    if not isinstance(chunk, SippyAudioChunk):
        raise TypeError('Expected an Infernos or sippy AudioChunk')
    pcm = torch.tensor(chunk.audio, dtype=torch.float32) / 32768.0
    return copy_metadata(chunk, AudioChunk(pcm, chunk.samplerate))
