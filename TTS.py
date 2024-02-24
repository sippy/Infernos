try:
    import intel_extension_for_pytorch as ipex
except ModuleNotFoundError:
    ipex = None

from HelloSippyTTSRT.HelloSippyRT import HelloSippyRT

from TTSRTPOutput import TTSRTPOutput, TTSSMarkerEnd

class TTS(HelloSippyRT):
    device = 'cuda' if ipex is None else 'xpu'
    debug = False

    def __init__(self):
        super().__init__(self.device)
        if ipex is not None:
            self.model = ipex.optimize(self.model)
            self.vocoder = ipex.optimize(self.vocoder)
            self.chunker = ipex.optimize(self.chunker)
            #raise Exception(f"{type(hsrt.chunker)}")

    def dotts(self, text, ofname):
        if False:
            tts_voc, so_voc = None, self.vocoder
        else:
            tts_voc, so_voc = self.vocoder, None
        writer = TTSRTPOutput(self.model.config.num_mel_bins,
                                self.device,
                                vocoder=so_voc)
        if self.debug:
            writer.enable_datalog(ofname)
        writer.start()

        speaker_embeddings = self.hsrt.get_rand_voice()

        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        speech = self.generate_speech_rt(inputs["input_ids"], writer.soundout,
                                               speaker_embeddings,
                                               vocoder=tts_voc)
        writer.soundout(TTSSMarkerEnd())

if __name__ == '__main__':
    tts = TTS()
    prompts = (
        "Hello and welcome to Sippy Software, your VoIP solution provider.",
        "Today is Wednesday twenty third of August two thousand twenty three, five thirty in the afternoon.",
        "For many applications, such as sentiment analysis and text summarization, pretrained models work well without any additional model training.",
        "This message has been generated by combination of the Speech tee five pretrained text to speech models by Microsoft and fine tuned Hello Sippy realtime vocoder by Sippy Software Inc.",
        )
    for i, prompt in enumerate(prompts):
        print(i)
        fname = f"tts_example{i}.wav"
#    prompt = "Hello and welcome to Sippy Software, your VoIP solution provider. Today is Wednesday twenty-third of August two thousand twenty three, five thirty in the afternoon."
#    prompt = 'For many applications, such as sentiment analysis and text summarization, pretrained models work well without any additional model training.'
    #prompt = "I've also played with the text-to-speech transformers those are great actually. I have managed making API more realtime and it works nicely!"
        tts.dotts(prompt, fname)
