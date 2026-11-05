"""Japanese SpeechT5 tokenizer using pyopenjtalk for G2P.

Compatible with transformers >= 5.x (PreTrainedTokenizer base class).
pyopenjtalk is imported lazily so the module can be loaded without it.
"""

import json
import logging
import os
from pathlib import Path
import re
from itertools import chain
from typing import List, Optional, Tuple

from transformers import AddedToken, PreTrainedTokenizer, SpeechT5FeatureExtractor, SpeechT5Processor

logger = logging.getLogger(__name__)

NP_CHARCTERS = " !\"#$%&'()=~|`{+*}<>?_-^\\@[;:],./　！”＃＄％＆’（）＝～｜｀｛＋＊｝＜＞？＿ー＾￥＠「；：」、。・`"


def _g2p_with_np(text: str, np_lsit: str) -> List[str]:
    """Run pyopenjtalk.g2p on *text*, preserving non-phoneme characters."""
    from pyopenjtalk import g2p

    np_pattern = re.compile(f"([{re.escape(np_lsit)}])")

    return list(
        chain.from_iterable(
            [
                (text,) if text in np_lsit else g2p(text, kana=False, join=False)
                for text in np_pattern.split(text)
                if len(text) > 0
            ]
        )
    )


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "esnya/japanese_speecht5_tts": "https://huggingface.co/esnya/japanese_speecht5_tts/resolve/main/vocab.json",
    },
}


class SpeechT5OpenjtalkTokenizer(PreTrainedTokenizer):
    """SpeechT5 tokenizer that uses pyopenjtalk for Japanese G2P.

    Inherits from PreTrainedTokenizer (not SpeechT5Tokenizer / SentencePieceBackend)
    because the vocabulary is a plain JSON mapping from phoneme strings to IDs,
    not a SentencePiece model.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    model_input_names = ["input_ids", "attention_mask"]
    is_fast = False

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        non_phenome_characters: str = NP_CHARCTERS,
        **kwargs,
    ):
        self.non_phenome_characters = non_phenome_characters
        self.vocab_file = vocab_file
        self._label2id: dict[str, int] = {}
        self._id2label: dict[int, str] = {}

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            **kwargs,
        )

        if vocab_file is not None:
            self._load_vocab()
            self._align_special_tokens_to_vocab()

    # -- vocabulary helpers ---------------------------------------------------

    def _load_vocab(self):
        if isinstance(self.vocab_file, str) and self.vocab_file.endswith(".json"):
            with open(self.vocab_file, encoding="utf-8") as f:
                self._label2id = json.load(f)
            self._id2label = {v: k for k, v in self._label2id.items()}
        else:
            raise ValueError(f"vocab_file must be a path to a .json file, got {self.vocab_file!r}")

    @property
    def label2id(self) -> dict[str, int]:
        return self._label2id

    @property
    def id2label(self) -> dict[int, str]:
        return self._id2label

    @property
    def vocab_size(self) -> int:
        return len(self._label2id)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._label2id)

    # -- low-level conversion -------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        return _g2p_with_np(text, self.non_phenome_characters)

    def _convert_token_to_id(self, token: str) -> int:
        return self._label2id.get(token, self._label2id.get(self.unk_token, 3))

    def _convert_id_to_token(self, index: int) -> str:
        return self._id2label.get(index, self.unk_token)

    # -- special token alignment ------------------------------------------

    def _align_special_tokens_to_vocab(self):
        """Align PreTrainedTokenizer's added_tokens_decoder to vocab special-token IDs.

        PreTrainedTokenizer auto-assigns sequential IDs 0–3 to <s>, </s>, <unk>, <pad>,
        but the SpeechT5 Japanese vocab uses different IDs (<s>=0, </s>=2, <unk>=3, <pad>=1).
        This method rebuilds added_tokens_decoder / added_tokens_encoder so that
        self.eos_token_id, self.pad_token_id, etc. return the vocab IDs.
        """
        for tok_str in (self.bos_token, self.eos_token, self.unk_token, self.pad_token):
            if tok_str is None:
                continue
            vocab_id = self._label2id.get(tok_str)
            if vocab_id is None:
                continue
            entry = AddedToken(
                tok_str,
                rstrip=False,
                lstrip=False,
                single_word=False,
                normalized=False,
                special=True,
            )
            self._added_tokens_decoder[vocab_id] = entry
            self._added_tokens_encoder[tok_str] = vocab_id

    # -- special token handling (mimics SpeechT5Tokenizer) ----------------

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> list:
        """Build model inputs from a sequence by appending eos_token_id."""
        if token_ids_1 is None:
            return token_ids_0 + [self.eos_token_id]
        return token_ids_0 + token_ids_1 + [self.eos_token_id]

    def get_special_tokens_mask(
        self, token_ids_0: list, token_ids_1: list = None, already_has_special_tokens: bool = False
    ) -> list:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )
        suffix_ones = [1]
        if token_ids_1 is None:
            return ([0] * len(token_ids_0)) + suffix_ones
        return ([0] * len(token_ids_0)) + ([0] * len(token_ids_1)) + suffix_ones

    def create_token_type_ids_from_sequences(
        self, token_ids_0: list, token_ids_1: list = None
    ) -> list:
        """SpeechT5 does not use token type ids, so return list of zeros."""
        eos = [self.eos_token_id]
        if token_ids_1 is None:
            return len(token_ids_0 + eos) * [0]
        return len(token_ids_0 + token_ids_1 + eos) * [0]

    # -- serialization --------------------------------------------------------

    def __getstate__(self):
        state = super().__getstate__()
        # Remove any keys that don't survive pickle
        for key in ("sp_model",):
            state.pop(key, None)
        return state

    def __setstate__(self, d):
        self.__dict__.update(d)
        if self.vocab_file is not None and not self._label2id:
            self._load_vocab()

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str, ...]:
        if filename_prefix is None:
            filename_prefix = ".json"

        save_path = Path(save_directory)
        if not save_path.is_dir():
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return ()

        vocab_path = Path(save_directory) / Path(f"vocab{filename_prefix}")
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self._label2id, f, ensure_ascii=False, indent=2)

        return (str(vocab_path),)


# ---------------------------------------------------------------------------
# Module-level convenience used by the checker
# ---------------------------------------------------------------------------

def get_ja_processor(device, model_name, config=None):
    """Build a SpeechT5Processor with the Japanese OpenJTalk tokenizer.

    Parameters
    ----------
    device : str
        Torch device string (unused here, kept for API compat).
    model_name : str
        HuggingFace model id, e.g. ``"esnya/japanese_speecht5_tts"``.
    config : SpeechT5Config, optional
        Pre-loaded config (unused, kept for API compat).

    Returns
    -------
    SpeechT5Processor
    """
    tokenizer = SpeechT5OpenjtalkTokenizer.from_pretrained(model_name)
    feature_extractor = SpeechT5FeatureExtractor.from_pretrained(model_name)
    return SpeechT5Processor(feature_extractor, tokenizer)
