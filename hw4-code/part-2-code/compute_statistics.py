"""
Script to compute data statistics for Q4 of the assignment.
This will help you fill in Tables 1 and 2 in the report.
"""

import os
from transformers import T5TokenizerFast
import numpy as np

def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def compute_statistics(data_folder, split, tokenizer, preprocess=False):
    """
    Compute statistics for the given split.
    
    Args:
        data_folder: Path to data folder
        split: 'train' or 'dev'
        tokenizer: T5 tokenizer
        preprocess: If True, apply preprocessing (add task prefix)
    """
    # Load data
    nl_path = os.path.join(data_folder, f'{split}.nl')
    sql_path = os.path.join(data_folder, f'{split}.sql')
    
    nl_queries = load_lines(nl_path)
    sql_queries = load_lines(sql_path)
    
    # Apply preprocessing if needed
    if preprocess:
        nl_queries = [f"translate English to SQL: {q}" for q in nl_queries]
    
    # Tokenize
    nl_tokenized = [tokenizer.encode(q) for q in nl_queries]
    sql_tokenized = [tokenizer.encode(q) for q in sql_queries]
    
    # Compute statistics
    num_examples = len(nl_queries)
    mean_nl_length = np.mean([len(tokens) for tokens in nl_tokenized])
    mean_sql_length = np.mean([len(tokens) for tokens in sql_tokenized])
    
    # Vocabulary size (unique tokens)
    nl_vocab = set()
    for tokens in nl_tokenized:
        nl_vocab.update(tokens)
    
    sql_vocab = set()
    for tokens in sql_tokenized:
        sql_vocab.update(tokens)
    
    return {
        'num_examples': num_examples,
        'mean_nl_length': mean_nl_length,
        'mean_sql_length': mean_sql_length,
        'nl_vocab_size': len(nl_vocab),
        'sql_vocab_size': len(sql_vocab)
    }

def main():
    data_folder = 'data'
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    print("=" * 60)
    print("Table 1: Statistics BEFORE preprocessing")
    print("=" * 60)
    
    for split in ['train', 'dev']:
        stats = compute_statistics(data_folder, split, tokenizer, preprocess=False)
        print(f"\n{split.upper()} SET:")
        print(f"  Number of examples: {stats['num_examples']}")
        print(f"  Mean sentence length: {stats['mean_nl_length']:.2f}")
        print(f"  Mean SQL query length: {stats['mean_sql_length']:.2f}")
        print(f"  Vocabulary size (natural language): {stats['nl_vocab_size']}")
        print(f"  Vocabulary size (SQL): {stats['sql_vocab_size']}")
    
    print("\n" + "=" * 60)
    print("Table 2: Statistics AFTER preprocessing")
    print("=" * 60)
    print(f"Model name: google-t5/t5-small")
    
    for split in ['train', 'dev']:
        stats = compute_statistics(data_folder, split, tokenizer, preprocess=True)
        print(f"\n{split.upper()} SET:")
        print(f"  Mean sentence length: {stats['mean_nl_length']:.2f}")
        print(f"  Mean SQL query length: {stats['mean_sql_length']:.2f}")
        print(f"  Vocabulary size (natural language): {stats['nl_vocab_size']}")
        print(f"  Vocabulary size (SQL): {stats['sql_vocab_size']}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
