import unicodedata

# helper functions 

def get_stats(ids, counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): # Pythonic way to iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
  """ 
  input (ids, pair to replace, id to replace with) ===> ouput new ids 
  In the list of ints (ids), replace all consecutive occurences of pair with the new token idx
  """
  newids = []
  i = 0
  while i < len(ids):
    # if we are not at the very last position AND the pair matches, replace it
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
      newids.append(idx)
      i += 2
    else:
      newids.append(ids[i])
      i += 1
  return newids


def replace_control_characters(s: str) -> str:
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s



# -------------------------------------------------------------------
# the base Tokenizer class

class Tokenizer:
    """ Base class for Tokenizer """

    def __init__(self):
        # default : vocab consist of all characters, no merges no patterns
        self.merges = {}
        self.pattern = "" #str
        self.special_tokens = {} #str ---> int
        self.vocab_start = {}
        self.vocab = self._build_vocab() # int--> bytes
        self.vocab_inverse = {}
        

    def train(self, psmiles, vocab_size, verbose = False):
        # Tokenizer can train a vocabulary of size vocab_size from text
        raise NotImplementedError

    def encode(self, psmile):
        # Tokenizer can encode a psmile into a list of integers using vocab
        raise NotImplementedError
    
    def decode(self, ids):
        # Tokenizer can decode a list of integers into a psmile using vocab
        raise NotImplementedError

    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        # special toekns are already in vocab start
        vocab = self.vocab_start.copy()
        for (p0, p1), idx in self.merges.items():
            vocab[int(idx)] = vocab[int(p0)] + vocab[int(p1)]
        return vocab

    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        # write the model: to be used in load() later
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # write the version, pattern and merges, that's all that's needed
            f.write("bpe v1\n")
            f.write(f"{self.pattern}\n")
            # write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # write the vocab_start, first the number of them, then each one 
            f.write(f"{len(self.vocab_start)}\n")
            for idx, vocab in self.vocab_start.items():
                f.write(f"{idx} {vocab}\n")
            # the merges dict
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")

    def load(self, model_file):
        """Inverse of save() but only for the model file"""
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        vocab_start ={}
        
        #index where the merges start
        idx = 368
        with open(model_file, 'r', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "bpe v1"
            # read the pattern
            self.pattern = f.readline().strip()
            # read the special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx) 
            # read the vocab_start
            num_vocab_start = int(f.readline().strip())
            for _ in range(num_vocab_start):
                idx_, vocab_ = f.readline().strip().split()
                vocab_start[int(idx_)] = vocab_
            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab_start = vocab_start
        self.vocab = self._build_vocab()
        self.vocab_inverse = {v:k for k,v in self.vocab.items()}
