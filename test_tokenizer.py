
from tokenizer import BasicTokenizer,RegexTokenizer

#t = BasicTokenizer()
t = RegexTokenizer()
t.load("tokenizer_models/regexTest7k.model")

text = 'COc1ccc2nc(S(=O)Cc3ncc(C)c(OC)c3C)[nH]c2c'

if text == t.decode(t.encode(text)):
    print('true')     
else:
    print('false') 

print(type(t.encode(text)))

