import torch
import os
from tokenizer import BasicTokenizer
from LanguageModel import BERTLanguageModel

#device
device = 'cuda' if torch.cuda.is_available() else cpu

# Load tokenizer 

#t = BasicTokenizer()
#t.load("tokenizer_models/regexTest7k.model")

from tokenizers import Tokenizer # import hugging face library
t_hf = Tokenizer.from_file("tokenizer_models/tokenizer-100m-HF.json")

# create a model class and load states
# bertmodel hyperparameters
vocab_size = t_hf.get_vocab_size()
block_size = 128
n_embd = 768

# model arch hyperparameters
n_head = 12 
n_layer = 12
dropout = 0.2

# trainer hyper parameters
lr = 1e-4
weight_decay = 0.01
betas = (0.9, 0.999)

# initialize BERT
bert_lm =  BERTLanguageModel(
    vocab_size = vocab_size,
    n_embd = n_embd,
    block_size = block_size,
    n_head = n_head,
    n_layer = n_layer,
    dropout = dropout,
    lr = lr,
    weight_decay= weight_decay,
    betas= betas,
).to(device)

# load from state dict
bert_lm.load_state_dict(torch.load(os.path.join("bert_models","file_name.pth")))

#set model to evaluation
bert_lm.eval()

# example psmile and embedding generation
psmile = '*CC(*)c1ccc(C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F)cc1'
embd = bert_lm.get_psmile_embedding(psmile, tokenizer = t_hf)
print(f'Embedding dimention : {embd.shape}')
print(f'The embedding looks like :....  {embd[0][:5]}')