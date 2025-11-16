# T5 Text-to-SQL Training Guide

## Implementation Summary

This implementation includes:

1. **load_data.py**: Complete data loading and preprocessing
   - T5Dataset class for loading and tokenizing data
   - Collate functions for dynamic padding
   - Support for train/dev/test splits

2. **t5_utils.py**: Model utilities
   - Model initialization (finetune or from scratch)
   - Checkpoint saving and loading
   - Optimizer and scheduler setup (already provided)

3. **train_t5.py**: Training and evaluation
   - Training loop (already provided)
   - Evaluation loop with metric computation
   - Test inference for generating predictions

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Fine-tune T5 Model
```bash
python train_t5.py \
    --finetune \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --max_n_epochs 10 \
    --patience_epochs 3 \
    --batch_size 16 \
    --test_batch_size 32 \
    --scheduler_type cosine \
    --num_warmup_epochs 1 \
    --experiment_name my_ft_experiment
```

### 3. Train from Scratch (optional)
```bash
python train_t5.py \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --max_n_epochs 20 \
    --patience_epochs 5 \
    --batch_size 16 \
    --test_batch_size 32 \
    --scheduler_type cosine \
    --num_warmup_epochs 2 \
    --experiment_name my_scratch_experiment
```

### 4. Evaluate on Dev Set
```bash
python evaluate.py \
    --predicted_sql results/t5_ft_my_ft_experiment_dev.sql \
    --predicted_records records/t5_ft_my_ft_experiment_dev.pkl \
    --development_sql data/dev.sql \
    --development_records records/ground_truth_dev.pkl
```

## Key Hyperparameters to Tune

1. **Learning Rate**: Start with 5e-5 for finetuning, 1e-4 for scratch
2. **Batch Size**: 16-32 depending on GPU memory
3. **Max Epochs**: 10-20 with early stopping
4. **Scheduler**: Cosine with warmup typically works well
5. **Weight Decay**: 0.01 for regularization

## Data Processing Details

The implementation includes:
- Task prefix: "translate English to SQL: " for encoder input
- T5Tokenizer from 'google-t5/t5-small'
- Dynamic padding in collate functions
- Proper decoder input shifting for teacher forcing

## Expected Performance

For full credit, achieve ≥ 65 F1 on test set:
- Start with baseline hyperparameters
- Monitor dev set performance during training
- Use early stopping to prevent overfitting
- Experiment with beam search (num_beams=4 is default)

## Optional Improvements

To push performance higher:
1. Freeze encoder layers and only finetune decoder
2. Experiment with different learning rates per layer
3. Add data augmentation or preprocessing
4. Try different beam search parameters
5. Use larger beam size (but slower inference)

## File Outputs

After training:
- `results/t5_ft_<experiment>_dev.sql` - Dev predictions
- `results/t5_ft_<experiment>_test.sql` - Test predictions  
- `records/t5_ft_<experiment>_dev.pkl` - Dev records
- `records/t5_ft_<experiment>_test.pkl` - Test records
- `checkpoints/ft_experiments/<experiment>/` - Model checkpoints
