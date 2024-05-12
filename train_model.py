import torch 
from LanguageModel import BERTDataset, BERTLanguageModel
from torch.utils.data import DataLoader
from tokenizers import Tokenizer # import hugging face library
import random
import os
import time
import datetime
import warnings
warnings.filterwarnings("ignore")
import torch.utils.data as data

# pytorch lightning 
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.strategies import DDPStrategy



print(f'Loading vocab in tokenizer')
t_hf = Tokenizer.from_file("tokenizer_models/tokenizer-100m-HF-vocab1000.json")
print(f"Tokenizer loaded successfully, vocab_size = {t_hf.get_vocab_size()}")

# bertmodel hyperparameters
vocab_size = t_hf.get_vocab_size()
block_size = 512
batch_size = 64

# load txt files
print(f'Loading textfiles')
text = open("dataset/20m_block512_traindata.txt", 'r')
train_set = text.read().splitlines()

print(f'Total number of lines in corpus in train + val: {len(train_set)}')

# use 20% of training data for validation
train_set_size = int(len(train_set) * 0.8)
valid_set_size = len(train_set) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(42)
train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)

# initialize dataset 
print(f"Loading training dataset")
train_dataset = BERTDataset(data = train_set, 
    tokenizer = t_hf, 
    seq_len = block_size)

# initialize dataset 
print(f"Loading validation dataset")
valid_dataset = BERTDataset(data = valid_set, 
    tokenizer = t_hf, 
    seq_len = block_size)

# initialize dataloader
train_loader = DataLoader(dataset = train_dataset , 
    batch_size = batch_size,
    shuffle=True)

# initialize dataloader
valid_loader = DataLoader(dataset = valid_dataset, 
    batch_size = batch_size,
    shuffle=True)

print(f'Dataloader loaded succesfully...')

# model arch hyperparameters
n_embd = 768
n_head = 12 
n_layer = 12
dropout = 0.2

# trainer hyper parameters
lr = 1e-4
weight_decay = 0.01
betas = (0.9, 0.999)
warmup_steps = 10000

print("Building BERT model")
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
    warmup_steps =warmup_steps
)
# early stopping criterion
early_stop_callback = EarlyStopping(monitor="val_loss", 
    patience=5,
    verbose=False, 
    mode="min")

#configure the ddp time out issue
ddp =DDPStrategy(process_group_backend="nccl", timeout=datetime.timedelta(seconds=9000),find_unused_parameters=True )

# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
trainer = L.Trainer( 
    max_epochs = 3,
    devices = 4,
    accelerator= 'gpu',
    strategy = ddp,
    accumulate_grad_batches = 4,
    callbacks=[early_stop_callback]
    )
# time stamping
t0 = time.time()

trainer.fit(bert_lm,train_loader,valid_loader)

t1 = time.time()
print(f"Training took { t1 - t0:.2f} seconds")
print(f'BERT training Ends...')

print(f'Saving the model...') 
import os
# create a directory for models, so we don't polute the current directory
os.makedirs("bert_models", exist_ok = True)

# save the model 
torch.save(bert_lm.state_dict(), os.path.join("bert_models","20M_model.pth"))
