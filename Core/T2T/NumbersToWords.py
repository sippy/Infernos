from typing import Optional
import re
import inflect

from config.InfernGlobals import InfernGlobals as IG

class NumbersToWords:
    tr:Optional[callable]
    cache:dict
    def __init__(self, lang='en'):
        self.p = inflect.engine()
        self.tr, self.cache = (None, None) if lang == 'en' else (IG.get_translator('en', lang).translate, {})

    def __call__(self, text):
        # Find all instances of numbers in the text
        numbers = re.findall(r'\b\d[\d.,]*%?(?=[\s.,!]|$)', text)

        # For each number found, replace it with its word equivalent
        for number in numbers:
            if number.endswith('%'):
                tr_number = number[:-1]
                suffix = ' percent'
            elif number[-1] in ('.', ',', '!'):
                tr_number = number[:-1]
                suffix = number[-1]
            else:
                suffix = ''
                tr_number = number
            word = self.p.number_to_words(tr_number) + suffix
            if self.tr is not None:
                if (word_tr:=self.cache.get(number, None)) is None:
                    self.cache[number] = word_tr = self.tr(word)
                word = word_tr
            text = text.replace(number, word, 1)
        return text

if __name__ == '__main__':
    n2w = NumbersToWords()
    print(n2w('I have 3 cats and 2 dogs.'))
    print(n2w('I have 3% cats and 2% dogs.'))
    print(n2w('I have 30000 cats and 2999 dogs.'))
    print(n2w('I have 50% cats and 29.0% dogs.'))
    print(n2w('I have 3,090.6 cats and 21,188,128 dogs.%,'))
    print(n2w('I have 3% cats and dogs 2%.'))
    print(n2w('I have 3% cats and dogs 20%, and mice 3.0%.'))
    print(n2w('I have 3% cats and dogs since 2024, or 2023.'))
