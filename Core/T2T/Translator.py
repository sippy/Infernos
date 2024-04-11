from typing import Tuple, Optional
from functools import partial

import argostranslate.package
from argostranslate.translate import get_installed_languages

def load_pair(from_code, to_code):
    print(f'load_pair({from_code=}, {to_code=})')
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    package_to_install = next(
        filter(
            lambda x: x.from_code == from_code and x.to_code == to_code, available_packages
        )
    )
    print(f'{package_to_install=}')
    argostranslate.package.install_from_path(package_to_install.download())

class Translator():
    supported_langs = ["en", "it", "de", "ru", "ja"]
    translators: Tuple[callable]
    def __init__(self, from_code: str, to_code: str, filter:Optional[callable]=None):
        to_code_p = [to_code,]
        inter_codes = [x for x in self.supported_langs if x not in (from_code, to_code)]
        success = False
        while not success:
            try: load_pair(from_code, to_code)
            except StopIteration: pass
            else:
                success = True
                break
            while len(inter_codes) > 0:
                inter_code = inter_codes.pop()
                try:
                    load_pair(from_code, inter_code)
                    load_pair(inter_code, to_code)
                except StopIteration:
                    if len(inter_codes) == 0: raise
                    continue
                to_code_p.insert(0, to_code)
                success = True
                break
        ilangs = dict((x.code, x) for x in get_installed_languages())
        from_lang = ilangs[from_code]
        translators = []
        for tc in to_code_p:
            to_lang = ilangs[tc]
            tr = from_lang.get_translation(to_lang).translate
            if filter is not None: tr = partial(filter, from_code=from_code, to_code=tc, tr=tr)
            translators.append(tr)
            from_lang, from_code = to_lang, tc
        self.translators = tuple(translators)

    def translate(self, sourceText):
        for translator in self.translators:
            sourceText = translatedText = translator(sourceText)
        return translatedText

if __name__ == '__main__':
    tr = Translator('en', 'ja')
    t0 = tr.translate('Hello world!')
    tr = Translator('ru', 'it')
    tr1 = Translator('it', 'de')
    #print(tr.to_code_p, tr1.to_code_p)
    sourceText = "Привет, как твои дела?"
    t1 = tr.translate(sourceText)
    t2 = tr1.translate(t1)
    print(t0, t1, t2)

