import re


def _extraction_test(sentence, index):
    return {
        'Word':sentence[index],
        'Index':index
        }


def extraction_simple(sentence, index):
    return {
        'word': sentence[index],
        'length': len(sentence[index]),
        'has_hyphen': '-' in sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index].capitalize() == sentence[index],
        'is_upper': sentence[index].isupper(),
        'is_lower': sentence[index].islower(),
        'is_digit': sentence[index].isdigit(),
        'suffix_1': sentence[index][-1:],
        'suffix_2': sentence[index][-2:],
        'suffix_3': sentence[index][-3:],
        'prev_word': sentence[index - 1] if index > 0 else '',
        'next_word': sentence[index + 1] if index < len(sentence) - 1 else '',
        }


def extraction_syllables(sentence, index):
    'Extract word features with syllables'
    syllable_expr = r'[{con}]?[{vow}]{{1,2}}(?:(?=[{vow}])|[{con}]{{0,2}}$|[{con}]{{0,3}}(?=[-{con}]))'.format(
        con='bcdfghjklmnpqrstvxzw', vow='aeiouyäö')
    
    word_normalized = sentence[index].lower()
    splitted = re.findall(syllable_expr, word_normalized)
    
    splitted_prev = '' if index == 0 else re.findall(
        syllable_expr, sentence[index-1])

    splitted_next = '' if index == len(sentence) - 1 else re.findall(
        syllable_expr, sentence[index+1])
    
    return {
        'word': sentence[index],
        'length': len(sentence[index]),
        'has_hyphen': '-' in sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index].capitalize() == sentence[index],
        'is_upper': sentence[index].isupper(),
        'is_lower': sentence[index].islower(),
        'is_digit': sentence[index].isdigit(),
        'suffix_1': splitted[-1] if len(splitted) > 1 else '',
        'suffix_2': splitted[-2] if len(splitted) > 2 else '',
        'suffix_3': splitted[-3] if len(splitted) > 3 else '',
        'prev_word_suffix': splitted_prev[-1] if len(splitted_prev) > 0 else '',
        'next_word_suffix': splitted_next[-1] if len(splitted_next) > 0 else ''
        }
