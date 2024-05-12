from .base import Tokenizer, get_stats, merge
import regex as re


# Declare the split patterns 

bond_pattern = r"""[^-=#:]+..[)]?|.+"""# prevents double and triple bonds from disolving

# helper function Encode
def remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

# Builld the vocabulary, size = 50
characters = ['#', '%', '(', ')', '*', '+', '-','.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',':', '=','@','A', 'B', 'C', 'F', 'G', 'H', 'I', 'K', 'L', 'N', 'O', 'P', 'S', 'T', 'Z', '[', '\\', ']', 'a', 'b', 'c', 'd', 'e', 'i', 'l', 'n', 'o', 'r', 's'] 
# tokens that pad, mask and mark attachment sites, size = 7
special_tokens = ["<pad>", "<mask>", "<unk>","*", "(*)", "(/*)", "[*]", "[/*]"] 
# 94 functional groups , size = 90
functional_groups = ['CC(F)(F)F', 'C1(C2=CC=CC=C2)=CC=CC=C1', 'C1(CC=C2)=C2C=CC=C1', '[NH]1CCCC1', 'CC#CC', 'CCC(CC)CO', 'CC=C=C(C)C', 'C/N=N/C', 'CC(N(C)C)=O', 'C/C(C)=N/C', 'C/C(N(C)C)=N/C', 'CC(=O)OC(=O)C', 'C(=O)Br', 'C(=O)Cl', 'C(=O)F', 'C(=O)I', 'CC=O', 'C(=O)N', '*N', 'C12=CC=CC=C1C=C3C(C=CC=C3)=C2', 'C([N-][N+]#N)', 'C1=CC=CC=C1', 'C1=CC=C(C=C1)S', 'C1CCCCC1C1CCCCC1', 'Br', 'CCC=C', 'CCC#C', 'O=C=O', 'C(=O)O', 'Cl', 'COCCl', 'C1=CC=C1', 'C1CCC1', 'C1CCCCCC1', 'C1CCCCC1', 'C1=CCCC=C1', 'C1=CCC=CC1', 'C=1CCCCC=1', 'C1CCCC1', 'C1=CCC=C1', 'C1CC1', 'C1=CC1', '[2H][CH2]C', 'COC', 'CCOCC', 'CC(C)OC(C)C', 'C&1&1&1&1', 'C=[N+]=[N-]', '[NH4+].[NH4+].[O-]S(=O)(=O)[S-]', 'CCS', 'CCO', 'C=C', 'COC', 'C(=O)OC', 'F', 'C=O','C(=O)', 'C1OC=CC=1', 'C&1&1&1', 'C#N', '[OH-]', 'NO', 'C1=CC=CC(CCC2)=C12', 'CC(=O)C', 'CS', 'CC(OC)=O', 'CN1CCCC1', 'CC(C)(C)OC', 'C12=CC=CC=C1C=CC=C2', '[N+](=O)[O-]', 'C[N+]([O-])=O', 'C12=CC=CC1=CC=C2', 'N1CC2CCCC2CC1', 'OC1CCCCC1', 'C=1(C=CC=CC1)', 'c1ccccc1C&1&1', 'CC(C)=O', 'CCC=O', 'CC=C', 'CC#C', 'N1CCCCC1', 'O=N1CCCCC1', 'NC', 'C12(CCCCC1)CCCCC2', 'S(=O)(=O)', 'C[N+](C)(C)C', 'S', 'OS(=O)(=S)O', 'CN(C)C', 'C1(C=CC=C2)=C2C(C=CC=C3)=C3C4=C1C=CC=C4']
# 118 elements ---> but remove all common ones like C N O --> this will destroy the sequences
elements = ['H', 'He', 'Li', 'Be', 'B', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
# 118 lower case elements, size = 118
misc = ['[H]','[H+]','[2H]','[nH]', '[NH4+]', '[N+]']
small_elements = [i.lower() for i in elements]

# combine every list and remove duplicates using set()
start_vocab = remove_duplicates(special_tokens + characters + functional_groups + elements + small_elements + misc )
# Build special_tokens--> spcial_tokens + functional_groups + elements + small_elements
special_tokens_ = remove_duplicates(special_tokens + functional_groups + elements + small_elements +misc)

# build vocab and inverse vocab
vocab = {int(i):token for i,token in enumerate(start_vocab)}
vocab_inverse = {token:int(i) for i,token in enumerate(start_vocab)}

# save start _vocab
vocab_start = vocab.copy()

#build special_tokens
# we created this dict but we will use vocab_inverse for pattern-->id 
special_tokens_dict = {token:int(i) for i,token in enumerate(special_tokens_)}


def encode_character(text):
    ids = []
    for char in text:
        id = vocab_inverse.get(char, 2)
        ids.append(id)
    return ids

# class regex tokenizer

class RegexTokenizer(Tokenizer):
    def __init__(self, pattern = None):
        super().__init__()
        self.pattern = bond_pattern if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = special_tokens_dict
        self.vocab_start = vocab_start

    def train(self, text , vocab_size, verbose = False):
        vocab_start_size = len(start_vocab)
        num_merges  = vocab_size - vocab_start_size
        print(f'Vocab_size training on :{vocab_size}')
        print(f'Length of characters : {len(characters)}')
        print(f'Length of Special tokens:{len(special_tokens)}')
        print(f'Length of functioal_grps : {len(functional_groups)}')
        print(f'Length of elements: { len(elements)}')
        print(f'Length of elements: { len(elements)}')
        print(f'Length of misc: {len(misc)}')
        print(f'Length of vocab_start: {len(start_vocab)}')
        print(f'Length of extended special_tokens: { len(special_tokens_)}') 
        
        # converting to encodes in charcater_vocab
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = [list(encode_character(ch)) for ch in text_chunks]
        
        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int ,int )--> int
        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            stats = {}
            for chunk_ids in ids:
                get_stats(chunk_ids, stats)
            # find the pair with the higshest count
            pair = max(stats, key = stats.get)
            # mint a new token: assign it the next available id
            idx = 368 + i
            # replace all occurrences of pair in ids with idx
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            vocab_inverse[pair[0] + pair[1]] = idx
            # print 
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        self.merges = merges # used in encode 
        self.vocab = vocab # used in decode
        self.vocab_inverse = vocab_inverse

    def decode(self, ids):
        #given ids (list of integers), return Python string
        psmile = ''.join(self.vocab[idx] for idx in ids)
        return psmile
    
    def _encode_chunk(self, chunk_encodes):
        # return token ids
        ids = chunk_encodes
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids
    
    def encode_ordinary(self, psmile_chunk):
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        chunks = re.findall(self.compiled_pattern, psmile_chunk)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in chunks:
            chunk_ids = self._encode_chunk(encode_character(chunk))
            ids.extend(chunk_ids)
        return ids
    
    def encode(self, psmile):
        special = self.special_tokens
        # otherwise, we have to be careful with potential special tokens in text
        # we handle special tokens by splitting the text
        # based on the occurrence of any exact match with any of the special tokens
        # we can use re.split for this. note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, psmile)
        special_chunks_filtered = list(filter(None, special_chunks))
        # now all the special characters are separated from the rest of the text
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for part in special_chunks_filtered:
            if part in special:
                # this is a special token, encode it separately as a special case
                ids.append(self.vocab_inverse[part])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_ordinary(part))
        return ids
        
        



