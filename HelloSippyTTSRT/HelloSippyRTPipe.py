#try: import intel_extension_for_pytorch as ipex
#except ModuleNotFoundError: ipex = None

import weakref, uuid
from typing import List, Optional

import torch
from torch.nn import functional as F
from datasets import load_dataset
from methodtools import lru_cache

try: from config.InfernGlobals import InfernGlobals
except ModuleNotFoundError:
    from sys import path as sys_path
    from os import getcwd
    sys_path.append(getcwd())
    from config.InfernGlobals import InfernGlobals

from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGanConfig, SpeechT5HifiGan, SpeechT5Processor, \
        SpeechT5Config
import torchaudio.transforms as T

from HelloSippyTTSRT.HelloSippyRT import AmendmentNetwork1, AmendmentNetwork1Config

def pad_tensor_to_target(tensor, target_size, pre=False):
    if tuple(tensor.shape) == tuple(target_size): return tensor
    # Get the current size of the tensor
    current_size = tensor.size()
    # Calculate the padding needed for each dimension
    padding = []
    for c, t in zip(current_size[::-1], target_size[::-1]):  # Reverse to start from the last dimension
        pad = max(t - c, 0)
        padding.extend([pad, 0] if pre else [0, pad])
    # Apply padding
    return F.pad(tensor, tuple(padding), "constant", 0.0)

class SessCmd: pass

class SessSyncCmd(SessCmd):
    live: List[uuid.UUID]
    def __init__(self, sessions:weakref.WeakValueDictionary[InfernGlobals]): self.live = tuple(sorted(sessions.keys()))

class SessDispatchCmd(SessCmd):
    session: uuid.UUID
    def __init__(self, session_id:uuid.UUID): self.session = session_id

class HelloSippyPlayRequest(SessDispatchCmd):
    text:str
    speaker:torch.Tensor
    dispatch:callable
    def __init__(self, session_id:uuid.UUID, text:str, speaker:torch.Tensor, dispatch:callable):
        self.text, self.speaker, self.dispatch = text, speaker, dispatch
        super().__init__(session_id)

def make_tensor(x): return torch.tensor([x], dtype=torch.long)

def maybe_half(x): return x.to(memory_format=torch.channels_last, dtype=torch.bfloat16) if not isinstance(x, torch.Tensor) or len(x.shape) > 3 else x.to(dtype=torch.bfloat16)

class HelloSippyPipeState:
    session:uuid.UUID
    dispatch:callable
    inputs:torch.Tensor
    speaker_embeddings:torch.Tensor
    encoder_last_hidden_state:torch.Tensor
    output_sequence:torch.Tensor
    encoder_attention_mask:torch.Tensor
    pre_frames:torch.Tensor
    starts_at:torch.Tensor
    ends_at:torch.Tensor

    def __init__(self, pp:'HelloSippyRTPipe', req:HelloSippyPlayRequest):
        self.session, self.dispatch = req.session, req.dispatch
        text = req.text if pp.cleanup_text is None else pp.cleanup_text(req.text)
        self.inputs = pp.processor(text=text, return_tensors="pt")["input_ids"]
        self.speaker_embeddings = maybe_half(req.speaker)
        self.encoder_attention_mask = torch.ones_like(self.inputs, dtype=torch.int)
        self.pre_frames = maybe_half(torch.zeros(1, pp.pre_nframes + pp.post_nframes, pp.model.config.num_mel_bins))
        self.starts_at = make_tensor(pp.post_nframes // pp.model.config.reduction_factor)
        self.ends_at = make_tensor(-1)

class HelloSippyPipeStateBatched:
    speaker_embeddings:torch.Tensor
    encoder_last_hidden_state:torch.Tensor
    output_sequence:torch.Tensor
    past_key_values:Optional[List[torch.Tensor]] = None
    encoder_attention_mask:torch.Tensor
    pre_frames:torch.Tensor
    starts_at:torch.Tensor
    ends_at:torch.Tensor
    minlen:int
    maxlen:int
    idx: int = 0
    dispatch:List[callable]
    sessions:List[uuid.UUID]

    def __init__(self, states: List[HelloSippyPipeState], pp:'HelloSippyRTPipe'):
        self.merge(states, pp)

    def merge(self, states:List[HelloSippyPipeState], pp:'HelloSippyRTPipe'):
        self.dispatch = [s.dispatch for s in states]
        max_statelen = max([x.encoder_attention_mask.size(1) for x in states])
        for aname in ('inputs', 'speaker_embeddings', 'encoder_attention_mask', 'pre_frames', 'starts_at', 'ends_at'):
            aval = [getattr(s, aname) for s in states]
            if aname in ('inputs', 'encoder_attention_mask'):
                for ia in [ia for ia, a in enumerate(aval) if a.size(1) < max_statelen]:
                    new_size = list(aval[ia].size())
                    new_size[1] = max_statelen
                    aval[ia] = pad_tensor_to_target(aval[ia], new_size)
            #print(f'{aname=} {[x.shape for x in aval]=}')
            setattr(self, aname, torch.cat(aval).to(pp.model.device).contiguous())
        encoder_out = pp.model.speecht5.encoder(
            input_values=self.inputs,
            attention_mask=self.encoder_attention_mask,
            return_dict=True,
        )
        self.encoder_last_hidden_state = encoder_out.last_hidden_state
        self.maxlen = int(self.encoder_last_hidden_state.size(1) * pp.maxlenratio / pp.model.config.reduction_factor)
        self.minlen = int(self.encoder_last_hidden_state.size(1) * pp.minlenratio / pp.model.config.reduction_factor)

        # Start the output sequence with a mel spectrum that is all zeros.
        self.output_sequence = self.encoder_last_hidden_state.new_zeros(self.inputs.size(0), 1, pp.model.config.num_mel_bins)
        #if self.past_key_values is not None:
        #        batch_size = self.speaker_embeddings.size(0)
        #        past_key_values = [list(x) for x in self.past_key_values]
        #        for past_key_value, idx, t in [(x, i, _x) for x in past_key_values for i, _x in enumerate(x)]:
        #            new_size = list(t.size())
        #            new_size[0] = batch_size
        #            if idx >= 2: new_size[2] = max_statelen
        #            #new_size[-1] = max_statelen
        #            assert id(past_key_value[idx]) == id(t)
        #            past_key_value[idx] = pad_tensor_to_target(t, new_size)
        #            #raise Exception(f"FIXME: NOT IMPLEMENTED: {past_key_value[idx].shape=}")
        #        self.past_key_values = tuple([tuple(x) for x in past_key_values])
        #if self.past_key_values is not None: print(f"FIXME: NOT IMPLEMENTED: {self.past_key_values[0][1].shape=} {self.past_key_values[1][0].shape=}")
        #self.past_key_values = None
        #print(f'{self.dispatch=}')


class HelloSippyRTPipe:
    processor: SpeechT5Processor
    model: SpeechT5ForTextToSpeech
    chunker: AmendmentNetwork1
    resampler: Optional[T.Resample]
    minlenratio: float = 0.0
    maxlenratio: float = 20.0
    threshold: float = 0.5
    chunk_size: int = 8
    pre_nframes: int = 2
    post_nframes: int = 2
    model_sr: int = 16000
    output_sr: int = 16000
    default_model = "microsoft/speecht5_tts"
    cleanup_text: Optional[callable] = None

    def __init__(self, device, model=default_model, get_processor:Optional[callable]=None, output_sr:int=output_sr, **kwa):
        self.cuda_lock = InfernGlobals().torcher
        self.cleanup_text = kwa.get('cleanup_text', self.cleanup_text)
        with self.cuda_lock:
            mc = SpeechT5Config.from_pretrained(model, **kwa)
            if get_processor is None:
               self.processor = SpeechT5Processor.from_pretrained(model, config=mc)
            else:
                self.processor = get_processor(device, model, config=mc)
            model = maybe_half(SpeechT5ForTextToSpeech.from_pretrained(model,
                                                            config=mc)).to(device)
            model.speecht5.decoder = maybe_half(model.speecht5.decoder)
            model.speecht5.encoder = maybe_half(model.speecht5.encoder)
            model.eval()
            self.model = model
            _vc_conf = SpeechT5HifiGanConfig()
            vocoder = maybe_half(SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan",
                                                    config = _vc_conf)).to(device)
            vocoder.eval()
            self.vocoder = vocoder
            self.c_conf = AmendmentNetwork1Config()
            chunker = AmendmentNetwork1.from_pretrained("sobomax/speecht5-rt.post_vocoder.v2",
                                                        config=self.c_conf)
            chunker = maybe_half(chunker).to(device)
            chunker.eval()
            self.chunker = chunker
            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            self.speaker_embeddings = [torch.tensor(ed["xvector"], device='cpu').unsqueeze(0)
                                        for ed in sorted(embeddings_dataset, key=lambda x: x['filename'])]
            for x in [_x for x in (self.model.parameters, self.vocoder.parameters, self.chunker.parameters) for _x in x()] + self.speaker_embeddings: x.requires_grad = False
            if self.model_sr != output_sr:
                self.resampler = maybe_half(T.Resample(orig_freq=self.model_sr, new_freq=output_sr)).to(device)
            else:
                self.resampler = None
            self.output_sr = output_sr

    def infer(self, state:HelloSippyPipeStateBatched) -> None:
        with self.cuda_lock:
            batch_size = state.output_sequence.size(0)
            spectrogram = maybe_half(torch.zeros(batch_size, 0, self.model.config.num_mel_bins)).to(self.model.device)
            while spectrogram.size(1) < (self.chunk_size * 4):
                decoder_hidden_states = self.model.speecht5.decoder.prenet(state.output_sequence, state.speaker_embeddings)[:, -1:]
                decoder_out = self.model.speecht5.decoder.wrapped_decoder(
                    hidden_states=decoder_hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=state.encoder_last_hidden_state,
                    encoder_attention_mask=state.encoder_attention_mask,
                    past_key_values=state.past_key_values,
                    use_cache=True,
                    output_attentions=False,
                    return_dict=True,
                )
                last_decoder_output = decoder_out.last_hidden_state[:, -1, :]
                state.past_key_values = decoder_out.past_key_values

                # Predict the new mel spectrum for this step in the sequence.
                spectrum = self.model.speech_decoder_postnet.feat_out(last_decoder_output)

                spectrum = spectrum.view(batch_size, self.model.config.reduction_factor, self.model.config.num_mel_bins)
                spectrogram = torch.cat((spectrogram, spectrum), dim=1)

                # Extend the output sequence with the new mel spectrum.
                spv = spectrum[:, -1:, :] #.view(spectrum.size(0), 1, self.model.config.num_mel_bins)
                state.output_sequence = torch.cat((state.output_sequence, spv), dim=1)

                # Predict the probability that this is the stop token.
                prob = self.model.speech_decoder_postnet.prob_out(last_decoder_output).sigmoid()

                # Finished when stop token or maximum length is reached.
                # if state.idx >= state.minlen and (int(sum(prob >= self.threshold)) > 0 or state.idx >= state.maxlen):
                #print(f"{(state.minlen <= state.idx)=} {(torch.sum(prob >= self.threshold, (1,)) > 0)=}")
                #raise Exception(f"{(state.maxlen <= state.idx)=}")
                state.ends_at = torch.where((state.ends_at < 0) & (state.minlen <= state.idx) & ((torch.sum(prob >= self.threshold, (1,)) > 0) | (state.maxlen <= state.idx)),
                                             state.idx+(self.pre_nframes+self.post_nframes)//self.model.config.reduction_factor, state.ends_at)
                state.idx += 1
            spectrogram = self.model.speech_decoder_postnet.postnet(spectrogram)
            spectrogram = torch.cat((state.pre_frames, spectrogram), dim=1)
            eframes = self.pre_nframes + self.post_nframes
            state.pre_frames = spectrogram[:, -eframes:, :]
            nchunks = spectrogram.size(1) // self.chunk_size
            spectrogram = torch.cat([spectrogram[:, i*self.chunk_size:(i+1)*self.chunk_size+eframes, :] for i in range(nchunks)], dim=0)
            audio = self.vocoder(spectrogram)
            audio = self.chunker(spectrogram, audio)
            slices = audio.split(batch_size, dim=0)
            audio = torch.cat(slices, dim=1)
            state.audio = self.resampler(audio) if self.resampler else audio

    def unbatch_and_dispatch(self, state:HelloSippyPipeStateBatched):
        audio, sr_rr = state.audio, self.model_sr // self.output_sr
        end_idx = state.idx - 1
        stepsize = 256 * 2 // sr_rr
        with self.cuda_lock:
            for i, dispatch in [(i, _cbq) for i, _cbq in enumerate(state.dispatch) if _cbq is not None]:
                startoff = max(0, (asize:=audio[i].size(0)) - ((state.idx - state.starts_at[i].item()) * stepsize))
                endoff = min(asize, asize - (((state.idx - ends_at) * stepsize) if (ends_at:=state.ends_at[i].item()) >=0 else 0))
                assert startoff <= endoff
                if startoff != endoff:
                    dispatch(audio[i][startoff:endoff].cpu())
                if ends_at >= 0 and ends_at <= end_idx:
                    dispatch(None)
                    state.dispatch[i] = None
            mask = ((state.ends_at < 0) | (state.ends_at > end_idx))
            if torch.all(~mask):
                return False
        return True

    def get_rand_voice_id(self):
        return torch.randint(0, len(self.speaker_embeddings), (1,)).item()

    def get_rand_voice(self):
        s_index = self.get_rand_voice_id()
        rv = self.speaker_embeddings[s_index]
        return (rv, s_index)

    #@lru_cache(maxsize=16)
    def get_voice(self, s_index:int):
        rv = self.speaker_embeddings[s_index]
        return rv
