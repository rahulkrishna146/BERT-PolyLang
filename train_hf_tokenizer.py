
from tokenizers import Tokenizer
from tokenizers.models import BPE
import time
from tokenizers.trainers import BpeTrainer
import regex as re
from tokenizers.pre_tokenizers import Whitespace

# helper function Encode
def remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


# tokens that pad, mask and mark attachment sites, size = 7
special_tokens = ["<pad>","<|SOS|>","<|EOS|>","<mask>","[UNK]","*", "(*)", "(/*)", "[*]", "[/*]"] 
# 94 functional groups 
functional_groups = ['CC(F)(F)F', 'C1(C2=CC=CC=C2)=CC=CC=C1', 'C1(CC=C2)=C2C=CC=C1', '[NH]1CCCC1', 'CC#CC', 'CCC(CC)CO', 'CC=C=C(C)C', 'C/N=N/C', '(C)','CC(N(C)C)=O', 'C/C(C)=N/C', 'C/C(N(C)C)=N/C', 'CC(=O)OC(=O)C', 'C(=O)Br', 'C(=O)Cl', 'C(=O)F', 'C(=O)I', 'CC=O', 'C(=O)N', '*N', 'C12=CC=CC=C1C=C3C(C=CC=C3)=C2', 'C([N-][N+]#N)', 'C1=CC=CC=C1', 'C1=CC=C(C=C1)S', 'C1CCCCC1C1CCCCC1', 'Br', 'CCC=C', 'CCC#C', 'O=C=O', 'C(=O)O', 'Cl', 'COCCl', 'C1=CC=C1', 'C1CCC1', 'C1CCCCCC1', 'C1CCCCC1', 'C1=CCCC=C1', 'C1=CCC=CC1', 'C=1CCCCC=1', 'C1CCCC1', 'C1=CCC=C1', 'C1CC1', 'C1=CC1', '[2H][CH2]C', 'COC', 'CCOCC', 'CC(C)OC(C)C', 'C&1&1&1&1', 'C=[N+]=[N-]', '[NH4+].[NH4+].[O-]S(=O)(=O)[S-]', 'CCS', 'CCO', 'C=C', 'COC', 'C(=O)OC', 'F', 'C=O','C(=O)', 'C1OC=CC=1', 'C&1&1&1', 'C#N', '[OH-]', 'NO', 'C1=CC=CC(CCC2)=C12', 'CC(=O)C', 'CS', 'CC(OC)=O', 'CN1CCCC1', 'CC(C)(C)OC', 'C12=CC=CC=C1C=CC=C2', '[N+](=O)[O-]', 'C[N+]([O-])=O', 'C12=CC=CC1=CC=C2', 'N1CC2CCCC2CC1', 'OC1CCCCC1', 'C=1(C=CC=CC1)', 'c1ccccc1C&1&1', 'CC(C)=O', 'CCC=O', 'CC=C', 'CC#C', 'N1CCCCC1', 'O=N1CCCCC1', 'NC', 'C12(CCCCC1)CCCCC2', 'S(=O)(=O)', 'C[N+](C)(C)C', 'S', 'OS(=O)(=S)O', 'CN(C)C', 'C1(C=CC=C2)=C2C(C=CC=C3)=C3C4=C1C=CC=C4']
# 118 elements ---> but remove all common ones like C N O --> this will destroy the sequences
elements = ['H', 'He', 'Li', 'Be', 'B', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
# 118 lower case elements, size = 118
misc = ['[H]','[H+]','[2H]','[nH]', '[NH4+]', '[N+]']
small_elements = [i.lower() for i in elements]
# Build special_tokens--> spcial_tokens + functional_groups + elements + small_elements
special_tokens_ = remove_duplicates(special_tokens + functional_groups + elements + small_elements +misc)

vocab_size = 1000
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(vocab_size= vocab_size,special_tokens=special_tokens_ )

# load txt dataset 
with open("dataset/100m_tokenizer.txt", 'r') as f:
    text = f.read().replace("\n", " ")

# using rex pattern to split bonds and rings 
bond_pattern = r""" ?[^-=#:\s]*(?:[-=#:].{1}[)]?)*"""
text_chunks = re.findall(bond_pattern, text)
print(f"Data loading successful....")
print(f'Tokenizer training begins...')

# time stamping
t0 = time.time()
files = []
tokenizer.train_from_iterator(text_chunks, trainer)
t1 = time.time()
print(f"Training took { t1 - t0:.2f} seconds")
tokenizer.save("tokenizer_models/tokenizer-100m-HF-voacb1000.json")