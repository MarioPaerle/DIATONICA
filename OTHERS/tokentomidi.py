import joblib
from GridTokenizer2 import remove_bars, detokenize


tokenizer = joblib.load("tokenizer_test3.pkl")
# print(tokenizer.tokenize_vector(["timesig.4.4"]))
to_detok = [2691,
 2632,
 2632,
 2632,
 2632,
 2632,
 2632,
 2632,
 1650,
 2632,
 2632,
 2632,
 1661,
 2632,
 1414,
 2632,
 2632,
 2632,
 2632,
 1514,
 2632,
 2632,
 2632,
 2632,
 2632,
 2632,
 2632,
 2632,
 2632,
 2632,
 2632,
 2632,
 2632,
 2632,
 2275,
 2632,
 2632,
 2632,
 2632,
 1606,
 2632,
 2632,
 2632,
 2632,
 2632,
 2632,
 2632,
 1632,
 2632,
 2632,
 2632,
 2632,
 2632,
 2632,
 2632,
 2632,
 2632,
 2632,
 2632,
 2632,
 2632]

detok = tokenizer.detokenize(to_detok)
print(detok)
detok = remove_bars(detok)
print(detok[-60:])
detokenized = detokenize(detok, 'output.mid')
print(detokenized)