from typing import Callable, Optional
from time import monotonic
import torch
from transformers import SpeechT5ForTextToSpeech, SpeechT5PreTrainedModel
from transformers.models.speecht5.modeling_speecht5 import \
        SpeechT5EncoderWithSpeechPrenet

GenerateSpeech_cb = Callable[[torch.FloatTensor], None]

def _generate_speech_rt(
    model: SpeechT5PreTrainedModel,
    input_values: torch.FloatTensor,
    speech_cb: GenerateSpeech_cb,
    speaker_embeddings: Optional[torch.FloatTensor] = None,
    threshold: float = 0.5,
    minlenratio: float = 0.0,
    maxlenratio: float = 20.0,
    vocoder: Optional[torch.nn.Module] = None,
) -> int:
    encoder_attention_mask = torch.ones_like(input_values)

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
    output_len = 2
    idx = 0

    ###stime_pre = None
    btime = monotonic()
    pfs = torch.zeros(model.pre_frames, model.config.num_mel_bins,
                      device=model.device)
    pf_trim = model.pre_frames * model._frame_size
    oschedule = [2, 2, 4, 8, 16]
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

        if len(spectrogram) >= output_len or (theend and len(spectrogram) > 0):
            _s = spectrogram.unsqueeze(0)
            _s = model.speech_decoder_postnet.postnet(_s)
            _s = _s.squeeze(0)
            if vocoder is not None:
                #_s = _s.permute(1, 0).unsqueeze(0)
                #print(_s.shape, _s.device)
                _s = torch.cat((pfs, _s), dim=0)
                outputs = vocoder(_s)
                #print(outputs.size())
                outputs = outputs[pf_trim:]
                #print(_s.shape, outputs.shape)
                pfs = _s[-model.pre_frames:, :]
            else:
                outputs = _s
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
            # Sentinel
            speech_cb(None)
            break

    return idx

class HelloSippyRT(SpeechT5ForTextToSpeech):
    pre_frames: int = 2
    _frame_size: int = 256

    @torch.no_grad()
    def generate_speech_rt(
        self,
        input_ids: torch.LongTensor,
        speech_cb: GenerateSpeech_cb,
        speaker_embeddings: Optional[torch.FloatTensor] = None,
        threshold: float = 0.5,
        minlenratio: float = 0.0,
        maxlenratio: float = 20.0,
        vocoder: Optional[torch.nn.Module] = None,
    ) -> int:
        return _generate_speech_rt(
            self,
            input_ids,
            speech_cb,
            speaker_embeddings,
            threshold,
            minlenratio,
            maxlenratio,
            vocoder,
        )
