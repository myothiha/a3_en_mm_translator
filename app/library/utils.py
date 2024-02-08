import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import pyidaungsu as pds

SRC_LANGUAGE = 'en'
TRG_LANGUAGE = 'my'

# Define special symbols and indices
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<sos>', '<eos>']

# Myanmar word tokenizer.
def mmtokenizer(sentence):
    return pds.tokenize(sentence, form="word")

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids):
    return torch.cat((torch.tensor([SOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

def get_text_transform(token_transform, vocab_transform):
    # src and trg language text transforms to convert raw strings into tensors indices
    text_transform = {}
    for ln in [SRC_LANGUAGE, TRG_LANGUAGE]:
        text_transform[ln] = sequential_transforms(token_transform[ln], # Tokenization
                                                vocab_transform[ln], # Numericalization
                                                tensor_transform) # Add BOS/EOS and create tensor
        
    return text_transform
