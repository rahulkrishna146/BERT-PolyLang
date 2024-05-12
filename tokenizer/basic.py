
from .base import Tokenizer, get_stats, merge


class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, psmiles, vocab_size, verbose = False):
        # expecting psmiles as list of list (tokens)
        l = 47  # vocab start would be nearly 47
        
        assert vocab_size >= l 
        num_merges = vocab_size - l

        # find all unique characters
        text_all = ''.join(psmiles)
        chars = sorted(list(set(text_all)))
        # vocab_start is ids of unique chars 
        vocab_start = [ord(char) for char in chars]

        # converting to encodes in utf-8
        id_list = []
        for psmile in psmiles:
            tokens = psmile.encode("utf-8")
            tokens = list(map(int, tokens))
            id_list.append(tokens)

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in vocab_start}
        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            stats = get_stats(id_list)
            # find the pair with the highest count
            pair = max(stats, key = stats.get)
            idx = 256 + i
            id_list = merge(id_list, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        self.merges = merges # used in encode 
        self.vocab = vocab # used in decode
        self.vocab_start = vocab_start
        

        
    def decode(self, ids):
        # given ids (list of integers), return Python string
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        # given a string text, return the token ids
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats_single(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge_single(ids, pair, idx)
        return ids
    