import joblib
from GridTokenizer2 import remove_bars, detokenize

tokenizer = joblib.load("tokenizer_test3.pkl")
print(tokenizer.tokenize_vector(["timesig.4.4"]))
to_detok = [2691,
  2662,
  2178,
  1762,
  1987,
  2049,
  2213,
  2249,
  2186,
  2369,
  1091,
  1890,
  2051,
  1925,
  2214,
  2153,
  615,
  1736,
  1650,
  1554,
  2081,
  1957,
  1512,
  1480,
  1544,
  1514,
  1708,
  2671,
  1468,
  527,
  932,
  1892,
  1796,
  2019,
  1923,
  1573,
  1034,
  1579,
  1547,
  1675,
  1484]

detok = tokenizer.detokenize(to_detok)
print(detok)
detok = remove_bars(detok)
print(detok[-60:])
detokenized = detokenize(detok, 'output.mid')
print(detokenized)