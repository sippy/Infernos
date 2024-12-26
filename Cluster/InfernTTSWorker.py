from typing import List

import torch

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

class cleanup_text_eu():
    replacements = [("Ä", "E"), ("Æ", "E"), ("Ç", "C"), ("É", "E"), ("Í", "I"), ("Ó", "O"), ("Ö", "E"), ("Ü", "Y"), ("ß", "S"),
        ("à", "a"), ("á", "a"), ("ã", "a"), ("ä", "e"), ("å", "a"), ("ë", "e"), ("í", "i"), ("ï", "i"), ("ð", "o"), ("ñ", "n"),
        ("ò", "o"), ("ó", "o"), ("ô", "o"), ("ö", "u"), ("ú", "u"), ("ü", "y"), ("ý", "y"), ("Ā", "A"), ("ā", "a"), ("ă", "a"),
        ("ą", "a"), ("ć", "c"), ("Č", "C"), ("č", "c"), ("ď", "d"), ("Đ", "D"), ("ę", "e"), ("ě", "e"), ("ğ", "g"), ("İ", "I"),
        ("О", "O"), ("Ł", "L"), ("ń", "n"), ("ň", "n"), ("Ō", "O"), ("ō", "o"), ("ő", "o"), ("ř", "r"), ("Ś", "S"), ("ś", "s"),
        ("Ş", "S"), ("ş", "s"), ("Š", "S"), ("š", "s"), ("ū", "u"), ("ź", "z"), ("Ż", "Z"), ("Ž", "Z"), ("ǐ", "i"), ("ǐ", "i"),
        ("ș", "s"), ("ț", "t"), ("ù", "u"),
    ]
    r_from, r_to = [''.join(x) for x in zip(*replacements)]
    replacements = str.maketrans(r_from, r_to)

    def __call__(self, text):
        return text.translate(self.replacements)

lang2model = {'en': {'cleanup_text':cleanup_text_eu()},
              'it': {'model':'Sandiago21/speecht5_finetuned_voxpopuli_it', 'cleanup_text':cleanup_text_eu()},
              'es': {'model':'Sandiago21/speecht5_finetuned_facebook_voxpopuli_spanish', 'cleanup_text':cleanup_text_eu()},
              'fr': {'model':'Sandiago21/speecht5_finetuned_facebook_voxpopuli_french', 'cleanup_text':cleanup_text_eu()},
              'de': {'model':'JFuellem/speecht5_finetuned_voxpopuli_de', 'cleanup_text':cleanup_text_eu()},
              'pt': {'model':'evertonaleixo/speecht5_finetuned_fleurs_ptbr', 'cleanup_text':cleanup_text_eu()},
              'ru': {'model':'zaebee/speecht5_tts_common_ru'},
              'ja': {'model': 'esnya/japanese_speecht5_tts', 'get_processor': get_ja_T5Processor},
             }

def get_torch_hw():
    if torch.cuda.is_available():
        return 'cuda' 
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return 'xpu'
    if hasattr(torch, 'mps'):
        return 'mps'
    raise AttributeError('Could not find CUDA deivces')

class InfernTTSWorker(InfernBatchedWorker):
    max_batch_size: int = 8
    debug = False
    tts_engine: HelloSippyRTPipe
    output_sr: int

    def __init__(self, lang, output_sr, device=None):
        from warnings import filterwarnings
        filterwarnings("ignore", category=FutureWarning)
        filterwarnings("ignore", category=UserWarning)
        try:
            import intel_extension_for_pytorch as ipex
        except ModuleNotFoundError:
            ipex = None
        super().__init__()
        if device is None:
            device = get_torch_hw()
        tts_engine = HelloSippyRTPipe(device, output_sr=output_sr, **lang2model[lang])
        if ipex is not None:
            for a in ('model', 'vocoder', 'chunker'):
                x = getattr(tts_engine, a)
                try: x = ipex.optimize(x)
                except AttributeError: continue
                setattr(tts_engine, a, x)
        self.tts_engine = tts_engine
        self.output_sr = output_sr

    def process_batch(self, wis:List[HelloSippyPlayRequest]):
        new_states = [HelloSippyPipeState(self.tts_engine, r) for r in wis]
        state = HelloSippyPipeStateBatched(new_states, self.tts_engine)
        while True:
            try:
                self.tts_engine.infer(state)
            except RuntimeError as e:
                self.handle_runtime_error(e, wis, state)
                raise
            if not self.tts_engine.unbatch_and_dispatch(state): break

    def handle_runtime_error(self, e, state, wis:List[HelloSippyPlayRequest]):
        print(f'InfernTTSWorker.handle_runtime_error: {e}')
        affected = [(d, w) for d, w in zip(state, wis) if d.dispatch is not None]

    def get_voice(self, *args):
        return self.tts_engine.get_voice(*args)

    def get_rand_voice(self):
        return self.tts_engine.get_rand_voice()


