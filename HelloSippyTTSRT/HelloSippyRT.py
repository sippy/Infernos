from typing import Callable, Optional
from time import monotonic
import torch
from transformers import SpeechT5ForTextToSpeech, SpeechT5PreTrainedModel, \
        SpeechT5HifiGanConfig, SpeechT5HifiGan, SpeechT5Processor, \
        SpeechT5Config
from transformers.models.speecht5.modeling_speecht5 import \
        SpeechT5EncoderWithSpeechPrenet
from transformers import PretrainedConfig, PreTrainedModel
from datasets import load_dataset
from threading import Lock
import torch.nn as nn

from config.InfernGlobals import InfernGlobals

GenerateSpeech_cb = Callable[[torch.FloatTensor], None]

class HelloSippyRT():
    pass

def _generate_speech_rt(
    hsrt: HelloSippyRT,
    input_values: torch.FloatTensor,
    speech_cb: GenerateSpeech_cb,
    speaker_embeddings: Optional[torch.FloatTensor] = None,
    threshold: float = 0.5,
    minlenratio: float = 0.0,
    maxlenratio: float = 20.0,
) -> int:
    with hsrt.cuda_lock:
        encoder_attention_mask = torch.ones_like(input_values)

        model = hsrt.model
        encoder_out = model.speecht5.encoder(
            input_values=input_values,
            attention_mask=encoder_attention_mask,
            return_dict=True,
        )

        encoder_last_hidden_state = encoder_out.last_hidden_state

        # downsample encoder attention mask
        if isinstance(model.speecht5.encoder, SpeechT5EncoderWithSpeechPrenet):
            encoder_attention_mask = model.speecht5.encoder.prenet._get_feature_vector_attention_mask(
                encoder_out[0].shape[1], encoder_attention_mask
            )

        maxlen = int(encoder_last_hidden_state.size(1) * maxlenratio / model.config.reduction_factor)
        minlen = int(encoder_last_hidden_state.size(1) * minlenratio / model.config.reduction_factor)

        # Start the output sequence with a mel spectrum that is all zeros.
        output_sequence = encoder_last_hidden_state.new_zeros(1, 1, model.config.num_mel_bins)

        spectrogram = torch.zeros(0, model.config.num_mel_bins).to(model.device)
        past_key_values = None
        idx = 0

        ###stime_pre = None
        btime = monotonic()
        p_ch = hsrt.chunker
        _c =  hsrt.c_conf
        prfs = torch.zeros(_c.pre_frames, model.config.num_mel_bins,
                        device=model.device)
        pofs = torch.zeros(_c.post_frames, model.config.num_mel_bins,
                        device=model.device)
        oschedule = [_c.chunk_size, _c.chunk_size, _c.chunk_size*2]
        output_len = oschedule[0]
        chunk_size = _c.chunk_size
        vocoder = hsrt.vocoder
        while True:
            idx += 1

            # Run the decoder prenet on the entire output sequence.
            decoder_hidden_states = model.speecht5.decoder.prenet(output_sequence, speaker_embeddings)

            # Run the decoder layers on the last element of the prenet output.
            decoder_out = model.speecht5.decoder.wrapped_decoder(
                hidden_states=decoder_hidden_states[:, -1:],
                attention_mask=None,
                encoder_hidden_states=encoder_last_hidden_state,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )

            last_decoder_output = decoder_out.last_hidden_state[0, -1]
            past_key_values = decoder_out.past_key_values

            # Predict the new mel spectrum for this step in the sequence.
            spectrum = model.speech_decoder_postnet.feat_out(last_decoder_output)
            spectrum = spectrum.view(model.config.reduction_factor, model.config.num_mel_bins)
            spectrogram = torch.cat((spectrogram, spectrum), dim=0)

            # Extend the output sequence with the new mel spectrum.
            spv = spectrum[-1].view(1, 1, model.config.num_mel_bins)
            output_sequence = torch.cat((output_sequence, spv), dim=1)

            # Predict the probability that this is the stop token.
            prob = model.speech_decoder_postnet.prob_out(last_decoder_output).sigmoid()

            # Finished when stop token or maximum length is reached.
            theend = theend_cb = False
            if idx >= minlen and (int(sum(prob >= threshold)) > 0 or idx >= maxlen):
                theend = True

            if (len(spectrogram) >= output_len and len(spectrogram) + prfs.size(0) >= chunk_size + _c.eframes) \
            or (theend and len(spectrogram) > 0):
                _s = spectrogram.unsqueeze(0)
                _s = model.speech_decoder_postnet.postnet(_s)
                _s = _s.squeeze(0)
                #print(_s.size(0), prfs.size(0), _s.device)
                in_size = _s.size()
                _s = [prfs, _s]
                if theend:
                    _s.append(pofs)
                _s = torch.cat(_s, dim=0)
                extra_pad = (_s.size(0) - _c.eframes) % chunk_size
                assert extra_pad < chunk_size
                if extra_pad > 0:
                    extra_pad = chunk_size - extra_pad
                    #print(_s.size())
                    _pofs = torch.zeros(extra_pad,
                                        _s.size(1), device=_s.device)
                    _s = torch.cat((_s, _pofs), dim=0)
                outputs = []
                while _s.size(0) >= _c.eframes + chunk_size:
                    #print(_s.size(), _s.device)
                    _i = _s[:_c.eframes + chunk_size, :]
                    _o = vocoder(_i).unsqueeze(0)
                    _o = p_ch(_i, _o)
                    outputs.append(_o.squeeze(0))
                    #print('out', _o.size(), outputs[-1].size())
                    _s = _s[chunk_size:, :]
                if extra_pad > 0:
                    ep_trim = extra_pad * _c.frame_size
                    assert outputs[-1].size(0) > ep_trim
                    outputs[-1] = outputs[-1][:-ep_trim]
                outputs = torch.cat(outputs, dim=0)
                #print('_s after:', _s.size(0))
                assert _s.size(0) >= _c.eframes and _s.size(0) < _c.eframes + chunk_size
                #print('prfs', prfs.size(), 'inputs', in_size, 'outputs', outputs.size(), '_s', _s.size())
                #print(_s.shape, outputs.shape)
                prfs = _s
                #print(monotonic() - btime)
                hsrt.cuda_lock.release()
                qlen, theend_cb = speech_cb(outputs)
                hsrt.cuda_lock.acquire()
                if output_len in oschedule:
                    oschedule.pop(0)
                    if len(oschedule) > 0:
                        output_len = oschedule[0]
                elif qlen > 1 and output_len < 64:
                    output_len *= 2
                spectrogram = torch.zeros(0, model.config.num_mel_bins).to(model.device)
            if theend or theend_cb:
                break

    return idx

class AmendmentNetwork1Config(PretrainedConfig):
    chunk_size = 8
    pre_frames = 2
    post_frames = 2
    frame_size = 256
    num_mels = 80
    chunk_size: int
    trim_pr: int
    trim_po: int
    output_size: int
    eframes: int

    def __init__(self, *a, **ka):
        super().__init__(*a, **ka)
        self.eframes = self.pre_frames + self.post_frames
        self.trim_pr = self.pre_frames * self.frame_size
        self.trim_po = self.post_frames * self.frame_size
        self.output_size = self.chunk_size * self.frame_size

class SimpleResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, stride=1,
                               padding=1, dilation=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, stride=1,
                               padding=3, dilation=3)

    def forward(self, x, lrelu):
        assert lrelu is not None
        residual = x
        x = lrelu(x)
        x = self.conv1(x)
        x = lrelu(x)
        x = self.conv2(x)
        x += residual
        return x

class AmendmentNetwork1(PreTrainedModel):
    config_class = AmendmentNetwork1Config
    def __init__(self, config=None):
        if config is None:
            config = self.config_class()
        super().__init__(config)
        _c = self._c = config

        self.conv_pre_m = nn.Conv1d(_c.num_mels, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_a = nn.Conv1d(_c.frame_size, 160, kernel_size=3, stride=1, padding=1)
        self.upsampler = nn.ModuleList([
          nn.ConvTranspose1d(192, 128, kernel_size=8, stride=4, padding=2),
          nn.ConvTranspose1d(128, 64, kernel_size=8, stride=4, padding=2),
        ])
        self.lrelu = nn.LeakyReLU(0.01)
        self.resblock = SimpleResidualBlock(64)
        self.post_conv = nn.Conv1d(in_channels=64, out_channels=_c.frame_size,
                                   kernel_size=8, stride=24, padding=0)

    def forward(self, mel, audio):
        batch_size, total_length = audio.size()
        T = mel.size(-1)
        #print(Exception(f"BP: ms:{mel.size()} as:{audio.size()}"))
        audio_reshaped = audio.view(batch_size, self._c.frame_size, -1)
        mel = mel.view(batch_size, T, -1)
        #print(Exception(f"BP: ms:{mel.size()} as:{audio.size()} ars:{audio_reshaped.size()}"))
        x_mel = self.conv_pre_m(mel)
        x_audio = self.conv_pre_a(audio_reshaped)
        am_comb = torch.cat((x_mel, x_audio), dim=1)
        for i, layer in enumerate(self.upsampler):
            am_comb = self.lrelu(am_comb)
            am_comb = layer(am_comb)
        am_comb = self.resblock(am_comb, self.lrelu)
        am_comb = self.lrelu(am_comb)
        am_comb = self.post_conv(am_comb).squeeze(-1)
        am_comb = self.lrelu(am_comb).view(batch_size, -1)
        audio = audio[:, self._c.trim_pr:-self._c.trim_po] * am_comb
        return audio.tanh()

class HelloSippyRT():
    processor: SpeechT5Processor
    chunker: AmendmentNetwork1
    c_conf: AmendmentNetwork1Config
    vocoder: SpeechT5HifiGan
    model: SpeechT5ForTextToSpeech
    cuda_lock = InfernGlobals().torcher
    def __init__(self, device, model="microsoft/speecht5_tts"):
        with self.cuda_lock:
            self.processor = SpeechT5Processor.from_pretrained(model)
            mc = SpeechT5Config(max_speech_positions=4000)
            model = SpeechT5ForTextToSpeech.from_pretrained(model,
                                                            config=mc).to(device)
            model.eval()
            self.model = model
            _vc_conf = SpeechT5HifiGanConfig()
            vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan",
                                                    config = _vc_conf).to(device)
            vocoder.eval()
            self.vocoder = vocoder
            self.c_conf = AmendmentNetwork1Config()
            chunker = AmendmentNetwork1.from_pretrained("sobomax/speecht5-rt.post_vocoder.v2",
                                                        config=self.c_conf)
            chunker = chunker.to(device)
            chunker.eval()
            self.chunker = chunker
            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            self.speaker_embeddings = [torch.tensor(ed["xvector"]).unsqueeze(0)
                                        for ed in embeddings_dataset]

    def get_rand_voice(self):
        with self.cuda_lock:
            s_index = torch.randint(0, len(self.speaker_embeddings), (1,)).item()
            rv = self.speaker_embeddings[s_index].to(self.model.device)
            return rv

    @torch.no_grad()
    def generate_speech_rt(
        self,
        input_ids: torch.LongTensor,
        speech_cb: GenerateSpeech_cb,
        speaker_embeddings: Optional[torch.FloatTensor] = None,
        threshold: float = 0.5,
        minlenratio: float = 0.0,
        maxlenratio: float = 20.0,
    ) -> int:
        return _generate_speech_rt(
            self,
            input_ids,
            speech_cb,
            speaker_embeddings,
            threshold,
            minlenratio,
            maxlenratio,
        )

    @torch.no_grad()
    def tts_rt(self, text, speech_cb, speaker=None):
        with self.cuda_lock:
            inputs = self.processor(text=text,
                                    return_tensors="pt").to(self.model.device)
        if speaker is None:
            speaker = self.get_rand_voice()
        self.generate_speech_rt(inputs["input_ids"], speech_cb,
                                  speaker)
