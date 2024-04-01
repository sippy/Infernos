try:
    import intel_extension_for_pytorch as ipex
except ModuleNotFoundError:
    ipex = None

from typing import List

from Cluster.InfernBatchedWorker import InfernBatchedWorker
from HelloSippyTTSRT.HelloSippyRTPipe import HelloSippyRTPipe, HelloSippyPlayRequest, \
    HelloSippyPipeState, HelloSippyPipeStateBatched

def get_ja_T5Processor(device, model_name):
    from utils.speecht5_openjtalk_tokenizer import SpeechT5OpenjtalkTokenizer
    from transformers import SpeechT5Processor, SpeechT5FeatureExtractor

    print(f'get_ja_T5Processor: device = {device}, model_name = {model_name}')
    tokenizer = SpeechT5OpenjtalkTokenizer.from_pretrained(model_name)
    tokenizer._in_target_context_manager = False
    tokenizer.split_special_tokens = True
    tokenizer._added_tokens_encoder = {}
    tokenizer._unk_token = None
    feature_extractor = SpeechT5FeatureExtractor.from_pretrained(model_name)
    return SpeechT5Processor(feature_extractor, tokenizer)

lang2model = {'en': {},
              'it': {'model':'Sandiago21/speecht5_finetuned_voxpopuli_it'},
              'de': {'model':'JFuellem/speecht5_finetuned_voxpopuli_de'},
              'ru': {'model':'zaebee/speecht5_tts_common_ru'},
              'ja': {'model': 'esnya/japanese_speecht5_tts', 'get_processor': get_ja_T5Processor},
             }

class InfernTTSWorker(InfernBatchedWorker):
    max_batch_size: int = 8
    device = 'cuda' if ipex is None else 'xpu'
    debug = False
    tts_engine: HelloSippyRTPipe

    def __init__(self, lang, output_sr):
        super().__init__()
        tts_engine = HelloSippyRTPipe(self.device, output_sr=output_sr, **lang2model[lang])
        if ipex is not None:
            self.model = ipex.optimize(tts_engine.model)
            self.vocoder = ipex.optimize(tts_engine.vocoder)
            self.chunker = ipex.optimize(tts_engine.chunker)
        self.tts_engine = tts_engine

    def process_batch(self, wis:List[HelloSippyPlayRequest]):
        new_states = [HelloSippyPipeState(self.tts_engine, r) for r in wis]
        state = HelloSippyPipeStateBatched(new_states, self.tts_engine)
        while True:
            self.tts_engine.infer(state)
            if not self.tts_engine.unbatch_and_dispatch(state): break

    def get_voice(self, *args):
        return self.tts_engine.get_voice(*args)

    def get_rand_voice(self):
        return self.tts_engine.get_rand_voice()


