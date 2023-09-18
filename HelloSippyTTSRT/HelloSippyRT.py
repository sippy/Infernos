from typing import Callable, Optional
from time import monotonic
import torch
from transformers import SpeechT5ForTextToSpeech, SpeechT5PreTrainedModel, \
        SpeechT5HifiGanConfig, SpeechT5HifiGan, SpeechT5Processor, \
        SpeechT5Config
from transformers.models.speecht5.modeling_speecht5 import \
        SpeechT5EncoderWithSpeechPrenet
from datasets import load_dataset
import torch.nn as nn

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
    prfs = torch.zeros(p_ch.pre_frames, model.config.num_mel_bins,
                      device=model.device)
    pofs = torch.zeros(p_ch.post_frames, model.config.num_mel_bins,
                       device=model.device)
    trim_pr = p_ch.pre_frames * p_ch.frame_size
    trim_po = p_ch.post_frames * p_ch.frame_size
    eframes = p_ch.pre_frames + p_ch.post_frames
    oschedule = [p_ch.chunk_size, p_ch.chunk_size, p_ch.chunk_size*2]
    output_len = oschedule[0]
    chunk_size = p_ch.chunk_size
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

        if (len(spectrogram) >= output_len and len(spectrogram) + prfs.size(0) >= chunk_size + eframes) \
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
            extra_pad = (_s.size(0) - eframes) % chunk_size
            assert extra_pad < chunk_size
            if extra_pad > 0:
                extra_pad = chunk_size - extra_pad
                #print(_s.size())
                _pofs = torch.zeros(extra_pad,
                                    _s.size(1), device=_s.device)
                _s = torch.cat((_s, _pofs), dim=0)
            outputs = []
            while _s.size(0) >= eframes + chunk_size:
                #print(_s.size(), _s.device)
                _i = _s[:eframes + chunk_size, :]
                _o = vocoder(_i).unsqueeze(0)
                _o = p_ch(_i, _o)
                outputs.append(_o.squeeze(0))
                #outputs.append(_o[trim_pr:-trim_po])
                #print('out', _o.size(), outputs[-1].size())
                _s = _s[chunk_size:, :]
            if extra_pad > 0:
                ep_trim = extra_pad * p_ch.frame_size
                assert outputs[-1].size(0) > ep_trim
                outputs[-1] = outputs[-1][:-ep_trim]
            outputs = torch.cat(outputs, dim=0)
            #print('_s after:', _s.size(0))
            assert _s.size(0) >= eframes and _s.size(0) < eframes + chunk_size
            #print('prfs', prfs.size(), 'inputs', in_size, 'outputs', outputs.size(), '_s', _s.size())
            #outputs = outputs[trim_pr:]
            #print(_s.shape, outputs.shape)
            prfs = _s
            #print(monotonic() - btime)
            qlen, theend_cb = speech_cb(outputs)
            if output_len in oschedule:
                oschedule.pop(0)
                if len(oschedule) > 0:
                    output_len = oschedule[0]
            elif qlen > 1 and output_len < 128:
                output_len *= 2
            spectrogram = torch.zeros(0, model.config.num_mel_bins).to(model.device)
        if theend or theend_cb:
            break

    return idx

class AmendmentNetwork(nn.Module):
    hidden_dim = 31
    kernel_size = 5
    pre_frames: int
    post_frames: int
    frame_size: int
    chunk_size: int
    trim_pr: int
    trim_po: int
    output_size: int
    def __init__(self, chunk_size=8, pre_frames=2, post_frames=2,
                 frame_size=256, num_mels=80):
        super(AmendmentNetwork, self).__init__()

        eframes = pre_frames + post_frames
        self.pre_frames = pre_frames
        self.post_frames = post_frames
        self.frame_size = frame_size
        self.chunk_size = chunk_size
        self.trim_pr = pre_frames * frame_size
        self.trim_po = post_frames * frame_size
        input_size_m = num_mels * (chunk_size + eframes)
        input_size_a = frame_size * (chunk_size + eframes)
        self.output_size = chunk_size * frame_size

        self.fc1_mel = nn.Linear(input_size_m, self.hidden_dim)
        self.fc1_audio = nn.Linear(input_size_a, self.hidden_dim)
        self.fc2 = nn.Linear(2 * self.hidden_dim, self.hidden_dim * 2)
        self.conv_out = nn.Conv1d(in_channels=self.hidden_dim * 2,
                                  out_channels=self.output_size,
                                  kernel_size=self.kernel_size,
                                  padding=self.kernel_size // 2)
        self.lrelu = nn.LeakyReLU(0.01)  # default negative slope is 0.01

    def forward(self, mel, audio):
        batch_size = audio.size(0)

        mel = mel.view(batch_size, -1)  # Flatten the mel input

        mel_out = self.lrelu(self.fc1_mel(mel))
        audio_out = self.lrelu(self.fc1_audio(audio))

        combined = torch.cat([mel_out, audio_out], dim=1)

        x = self.lrelu(self.fc2(combined))
        x = x.unsqueeze(2)
        x = self.conv_out(x).squeeze(2)
        amended_audio = audio[:, self.trim_pr:-self.trim_po] * self.lrelu(x)

        return amended_audio.clamp(min=-1.0, max=1.0)

class HelloSippyRT():
    processor: SpeechT5Processor
    chunker: AmendmentNetwork
    vocoder: SpeechT5HifiGan
    model: SpeechT5ForTextToSpeech
    def __init__(self, device):
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        mc = SpeechT5Config(max_speech_positions=4000)
        model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts",
                                                        config=mc).to(device)
        model.eval()
        self.model = model
        _vc_conf = SpeechT5HifiGanConfig()
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan",
                                                  config = _vc_conf).to(device)
        vocoder.eval()
        self.vocoder = vocoder
        self.chunker = AmendmentNetwork().to(device)
        self.chunker.eval()
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.speaker_embeddings = [torch.tensor(ed["xvector"]).unsqueeze(0)
                                       for ed in embeddings_dataset]

    def get_rand_voice(self):
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
        inputs = self.processor(text=text,
                                return_tensors="pt").to(self.model.device)
        if speaker is None:
            speaker = self.get_rand_voice()
        self.generate_speech_rt(inputs["input_ids"], speech_cb,
                                  speaker)
