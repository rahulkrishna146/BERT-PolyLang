import os
import time 
from tokenizer import BasicTokenizer, RegexTokenizer


# load txt dataset 

with open("dataset/7k_psmiles.txt", 'r') as f:
    text1 = f.read().replace("\n", " ")

with open("dataset/6k_psmiles.txt", 'r') as f:
    text2 = f.read().replace("\n", " ")

# concatenating strings 
text = text1 + text2

# create a directory for models, so we don't polute the current directory
os.makedirs("tokenizer_models", exist_ok = True)

# time stamping
t0 = time.time()

#vocab_size -- hyperparameter
vocab_size = 600#

#tokenizer = BasicTokenizer()
tokenizer = RegexTokenizer()
tokenizer.train(text, vocab_size, verbose = True)

# write two files in the models directory: name.model, and name.vocab
prefix = os.path.join("tokenizer_models","regex_speed_13k")
tokenizer.save(prefix)

t1 = time.time()

print(f"Training took { t1 - t0:.2f} seconds")
