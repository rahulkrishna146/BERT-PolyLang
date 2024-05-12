import torch
import itertools
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import tqdm
import lightning as L
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Head(nn.Module):
    """
    One head on self - attention 
    """
    def __init__(self, head_size, n_embd, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) #B, T, head_size
        q = self.query(x) # B, T, head_size

        # compute the self attention scores
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) --> (B, T, T)
        wei = F.softmax(wei, dim =-1) # softmax
        wei = self.dropout(wei)
        # perform weigthed aggregation 
        v = self.value(x) # B, T, C
        out = wei @ v # (B,T,T) @ (B,T, C) ---> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self attention in parallel"""
    def __init__(self, n_head, head_size, n_embd, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size=head_size, n_embd=n_embd, dropout=dropout ) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout =nn.Dropout(dropout)
    
    def forward(self, x):
        out =  torch.cat([h(x) for h in self.heads], dim =-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """a simple linear layer followe dby a non-linearilty """

    def __init__(self,n_embd,dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Trnaformer bloack : communication followed b computation"""
    def __init__(self, n_embd, n_head,dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head = n_head, head_size=head_size,n_embd= n_embd, dropout =dropout )
        self.ffwd = FeedForward(n_embd=n_embd,dropout=dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


#embedding and language model 
class BERTLanguageModel(L.LightningModule):
    
    def __init__(self,vocab_size,n_embd,block_size,n_head, n_layer,dropout,lr,weight_decay, betas, warmup_steps):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas,
        self.warmup_steps = warmup_steps
        
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size  = block_size

        self.save_hyperparameters()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd, padding_idx =0)
        self.position_embedding_table = nn.Embedding(block_size, n_embd, padding_idx = 0)
        self.blocks = nn.Sequential(*[Block(n_embd=n_embd, n_head= n_head,dropout=dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        print("Total Parameters:", sum([p.nelement() for p in self.parameters()]))

    def forward(self, idx, targets):
        B,T = idx.shape

        tok_embedding = self.token_embedding_table(idx) # BTC C- embed C
        pos_embedding = self.position_embedding_table(torch.arange(T).to(self.device)) # toch.arange(T)  #(T,C)

        x = tok_embedding + pos_embedding
        x = self.blocks(x)
        logits = self.lm_head(x) # BTC
        return logits
        
    def training_step(self, batch):
        idx , targets = batch['bert_input'], batch['bert_labels']
        logits = self(idx, targets)
        B,T,C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets, ignore_index = 0)
        self.log("train_loss", loss,prog_bar=True)
        return loss
    
    def validation_step(self, batch):
        idx , targets = batch['bert_input'], batch['bert_labels']
        logits = self(idx, targets)
        B,T,C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets, ignore_index = 0)
        self.log("val_loss", loss, prog_bar=True)


    def configure_optimizers(self):
        optimizer = Adam(self.parameters(),lr =1e-4, weight_decay= 0.01 ,betas=(0.9,0.999))
        scheduler = ReduceLROnPlateau(optimizer, 'min')
        return {"optimizer": optimizer , "scheduler": scheduler}


    @torch.no_grad()
    def generate_embedding(self, encoding):
        encoding = encoding.view(1, self.block_size)
        token_embedding = self.token_embedding_table(encoding)
        position_embedding = self.position_embedding_table(torch.arange(self.block_size).to(self.device))

        x = token_embedding + position_embedding
        x = self.blocks(x)
        emd = x.view(self.block_size, self.n_embd)
        return emd
    
    def mean_pooling(self,model_out,attention_mask):
        input_mask_expanded = (attention_mask.unsqueeze(-1).expand(model_out.size()).float())
        return torch.sum(model_out, 0) / torch.clamp(input_mask_expanded.sum(0), min = 1e-9)

    def get_psmile_embedding(self, text, tokenizer):
        # add special tokens begging and end 
        text = "<|SOS|>" + text + "<|EOS|>"
        out  = tokenizer.encode(text)
        encoding = out.ids
        # if the smile length is grater than bloack sie truncate
        if len(encoding) > self.block_size:
            encoding = encoding[:self.block_size]
            attention_mask = [1 for _ in range(len(encoding))]
        else:
            padding = [0 for _ in range(self.block_size - len(encoding))]
            attention_mask = [1 for _ in range(len(encoding))]
            encoding = encoding + padding
            attention_mask = attention_mask + padding 
        encod = torch.tensor(encoding).to(self.device)
        attention_mask = torch.tensor(attention_mask).to(self.device)
        emb = self.generate_embedding(encod)
        return F.normalize(self.mean_pooling(emb, attention_mask).view(1, self.n_embd))