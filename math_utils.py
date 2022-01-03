SYMBOL_DICT = {'plus': '+',
               'minus': '-',
               'div': '/',
               'mul': '*',
               'colon_open': '(',
               'colon_close': ')',
               }


def symbol_words_to_symbol_characters(expression):
    for key, value in SYMBOL_DICT.items():
        expression = expression.replace(key, value, -1)
    return expression
