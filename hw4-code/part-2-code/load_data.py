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

    def __init__(self, data_folder, split, use_schema_enhancement=False):
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
        self.use_schema_enhancement = use_schema_enhancement
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        
        # Load schema if using enhancement
        if use_schema_enhancement:
            self.schema_text = self._load_schema(data_folder)
        else:
            self.schema_text = None
            
        self.encoder_inputs, self.decoder_inputs, self.decoder_targets = self.process_data(data_folder, split, self.tokenizer)

    def _load_schema(self, data_folder):
        """Load and format database schema information from flight_database.schema file."""
        import json
        schema_path = os.path.join(data_folder, 'flight_database.schema')
        
        # Load the schema JSON file
        with open(schema_path, 'r') as f:
            schema_data = json.load(f)
        
        # Extract table definitions from "ents" section
        tables = schema_data.get('ents', {})
        
        # Format schema as: Table: table_name (column1, column2, ...) | Table: ...
        schema_parts = []
        for table_name, columns in sorted(tables.items()):
            column_names = sorted(columns.keys())
            column_str = ', '.join(column_names)
            schema_parts.append(f"Table: {table_name} ({column_str})")
        
        schema_info = ' | '.join(schema_parts)
        return schema_info

    def process_data(self, data_folder, split, tokenizer):
        # Load natural language queries
        nl_path = os.path.join(data_folder, f'{split}.nl')
        nl_queries = load_lines(nl_path)
        
        # Tokenize encoder inputs (natural language queries)
        encoder_inputs = []
        for query in nl_queries:
            if self.use_schema_enhancement:
                # Enhanced format: Question: ... Schema: ... Answer:
                input_text = f"Question: {query} Schema: {self.schema_text} Answer:"
            else:
                # Original format
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
            # Add END token to SQL for training (helps model learn when to stop)
            sql_with_end = sql + ' END'
            
            # Tokenize the SQL query with END
            encoded = tokenizer(sql_with_end, return_tensors='pt', add_special_tokens=True)
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

def get_dataloader(batch_size, split, use_schema_enhancement=False):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split, use_schema_enhancement=use_schema_enhancement)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size, use_schema_enhancement=False):
    train_loader = get_dataloader(batch_size, "train", use_schema_enhancement)
    dev_loader = get_dataloader(test_batch_size, "dev", use_schema_enhancement)
    test_loader = get_dataloader(test_batch_size, "test", use_schema_enhancement)
    
    return train_loader, dev_loader, test_loader


def extract_sql_before_end(generated_text):
    """
    Extract SQL query from generated text by getting everything before END token.
    If END token is not found, return the full generated text.
    
    Args:
        generated_text: The generated SQL query (potentially with END token)
    
    Returns:
        SQL query with END token and everything after it removed
    """
    if 'END' in generated_text:
        # Split by END and take only the first part
        sql_query = generated_text.split('END')[0].strip()
    else:
        # If no END token, return as is
        sql_query = generated_text.strip()
    
    return sql_query

def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x