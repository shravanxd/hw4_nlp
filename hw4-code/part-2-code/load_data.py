import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.encoder_inputs, self.decoder_inputs, self.decoder_targets = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        # Load natural language queries
        nl_path = os.path.join(data_folder, f'{split}.nl')
        nl_queries = load_lines(nl_path)
        
        # Tokenize encoder inputs (natural language queries)
        encoder_inputs = []
        for query in nl_queries:
            # Add prefix for text-to-SQL task
            input_text = f"translate English to SQL: {query}"
            encoded = tokenizer(input_text, return_tensors='pt', add_special_tokens=True)
            encoder_inputs.append(encoded['input_ids'].squeeze(0))
        
        # For test set, we don't have SQL targets
        if split == 'test':
            return encoder_inputs, None, None
        
        # Load SQL queries for train/dev
        sql_path = os.path.join(data_folder, f'{split}.sql')
        sql_queries = load_lines(sql_path)
        
        # Tokenize decoder inputs and targets
        decoder_inputs = []
        decoder_targets = []
        
        for sql in sql_queries:
            # Tokenize the SQL query
            encoded = tokenizer(sql, return_tensors='pt', add_special_tokens=True)
            tokens = encoded['input_ids'].squeeze(0)
            
            # Decoder input: start with pad token (T5 uses pad as decoder start)
            decoder_input = tokens[:-1]  # All tokens except last
            decoder_target = tokens[1:]   # All tokens except first (shifted by 1)
            
            decoder_inputs.append(decoder_input)
            decoder_targets.append(decoder_target)
        
        return encoder_inputs, decoder_inputs, decoder_targets
    
    def __len__(self):
        return len(self.encoder_inputs)

    def __getitem__(self, idx):
        if self.split == 'test':
            return self.encoder_inputs[idx]
        else:
            return self.encoder_inputs[idx], self.decoder_inputs[idx], self.decoder_targets[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_inputs, decoder_inputs, decoder_targets = zip(*batch)
    
    # Pad encoder inputs
    encoder_ids = pad_sequence(encoder_inputs, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    
    # Pad decoder inputs and targets
    decoder_inputs_padded = pad_sequence(decoder_inputs, batch_first=True, padding_value=PAD_IDX)
    decoder_targets_padded = pad_sequence(decoder_targets, batch_first=True, padding_value=PAD_IDX)
    
    # Initial decoder input (just the first token, typically PAD_IDX for T5)
    initial_decoder_inputs = torch.full((len(batch), 1), PAD_IDX, dtype=torch.long)
    
    return encoder_ids, encoder_mask, decoder_inputs_padded, decoder_targets_padded, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_inputs = batch
    
    # Pad encoder inputs
    encoder_ids = pad_sequence(encoder_inputs, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    
    # Initial decoder input (PAD_IDX for T5)
    initial_decoder_inputs = torch.full((len(batch), 1), PAD_IDX, dtype=torch.long)
    
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x