import pandas as pd
import numpy as np
import torch
import os
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #  2 = INFO and WARNING messages are not printed
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout ,Flatten
from keras.callbacks import ReduceLROnPlateau

from tokenizer import BasicTokenizer
from LanguageModel import BERTLanguageModel


#device
device = 'cuda' if torch.cuda.is_available() else cpu

# Load tokenizer 

#t = BasicTokenizer()
#t.load("tokenizer_models/basic.model")
from tokenizers import Tokenizer # import hugging face library
t_hf = Tokenizer.from_file("tokenizer_models/tokenizer-100m-HF.json")
print(f"Tokenizer loaded succesfully...")

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

#initialize the model
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
# load the model
bert_lm.load_state_dict(torch.load(os.path.join("bert_models","1L_new.pth")))

# set it to eval mode
bert_lm.eval()

print(f'Bert model loaded sucessfully..')
# load the dataset
data = pd.read_csv("dataset/7k_psmiles.csv")

smiles =data['SMILES']
Y = data['tg'].to_numpy()

def canonize(mol):
    return Chem.MolToSmiles(Chem.MolFromSmiles(mol), isomericSmiles=True, canonical=True)

canon_smile = []
for molecule in smiles:
    canon_smile.append(canonize(molecule))

tensor_encoding = []   
for psmile in canon_smile:
    tensor_encoding.append(bert_lm.get_psmile_embedding(psmile, tokenizer = t_hf)[0].cpu().numpy())

print(f'Encodings generated succesfully...')

t_encod = np.array(tensor_encoding)

X_train, X_test, Y_train, Y_test = train_test_split(t_encod, Y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# model 
model = Sequential()

model.add(Dense(256, input_dim = X_train.shape[1], activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(32,activation = 'relu'))
model.add(Dropout(0.1))

model.add(Dense(1,activation = 'linear'))

opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error'])

rlrop = ReduceLROnPlateau(monitor='mean_absolute_error', factor=0.01, patience=10)
model.fit(X_train, Y_train, epochs = 200,batch_size= 128, callbacks=[rlrop], verbose = 0,)

#Making predictions on the test set
predictions = model.predict(X_test)

mse = mean_squared_error(Y_test, predictions)
r2 = r2_score(Y_test, predictions)

print("Root-Mean Squared Error:", np.sqrt(mse))
print("R-squared Score:", r2)
