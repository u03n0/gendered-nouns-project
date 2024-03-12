from sklearn.model_selection import train_test_split
from src.models.utils import get_data_from_df
from random import shuffle



class DataGenerator:
    def __init__(self, data, pad_token='<pad>', unk_token='<unk>'):

        self.pad_token = pad_token
        self.unk_token = unk_token

        self.input_idx2sym,self.input_sym2idx   = self.vocabulary(data, False)# False
        self.output_idx2sym,self.output_sym2idx = self.vocabulary(data) # true?

        nouns, genders = get_data_from_df(data)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(nouns, genders, test_size=0.2)

        self.train_size = len(self.X_train)
        self.test_size = len(self.X_test)

    def generate_batches(self, batch_size, validation=False):

        if validation:
            X = self.X_test
            y = self.y_test
        else:
            X = self.X_train
            y = self.y_train

        assert(len(X) == len(y))

        N = len(X)
        idxes = list(range(N))

        # Data ordering
        shuffle(idxes)
        idxes.sort(key=lambda idx: len(X[idx]))

        # batch generation
        bstart = 0
        while bstart < N:
            bend = min(bstart + batch_size,N)
            batch_idxes = idxes[bstart:bend]
            batch_len = max(len(X[idx]) for idx in batch_idxes)
            Xpad = [self.pad_sequence(X[idx], batch_len) for idx in batch_idxes]
            #   save_padded_words('../data/eval/padded_fr', Xpad)
            seqX = [self.code_sequence(x, self.input_sym2idx) for x in Xpad]
            seqY = [self.output_sym2idx[y[idx]] for idx in batch_idxes]

            assert(len(seqX) == len(seqY))
            yield (seqX,seqY)
            bstart += batch_size

    def vocabulary(self, data, labels=True):

        nouns, genders = get_data_from_df(data) # false (not reversed)
        
        if labels:
            sym2idx = {sym: idx for idx, sym in enumerate(set(genders))}
        else:
            unique_chars = set(char for noun in nouns for char in noun)
            sym2idx = {sym: idx for idx, sym in enumerate(unique_chars)}
            sym2idx[self.unk_token] = len(sym2idx)
            sym2idx[self.pad_token] = len(sym2idx)

        idx2sym = [sym for sym in sym2idx.keys()]

        return idx2sym, sym2idx
    
    def pad_sequence(self, sequence, pad_size):
        # returns a list of the characters in the sequence with additional pad tokens to match pad_size if needed
        return list(sequence) + [self.pad_token] * (pad_size - len(sequence))

    def code_sequence(self, charseq, encodingmap):
        # charseq is a sequence of chars
        return [encodingmap[char] if char in encodingmap 
                else encodingmap[self.unk_token] for char in charseq]

    def decode_sequence(self, idxseq, decodingmap):
        # idxseq is a list of integers
        return [decodingmap[idx] for idx in idxseq]
