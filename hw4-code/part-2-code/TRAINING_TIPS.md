# Training Tips to Reach F1 ≥ 0.65

## Problem Analysis
Your training shows classic overfitting:
- Epoch 10: F1 = 0.57 ✅ (good start)
- Epoch 15: F1 = 0.38 ❌ (dramatic drop - overfitting kicks in)
- Epoch 35: F1 = 0.41 ⚠️ (slow recovery)

## Key Fixes Implemented

### 1. **Label Smoothing (NEW)** ⭐
- Added `--label_smoothing 0.1` (default)
- Prevents model from being overconfident on training data
- Helps generalization to dev set

### 2. **Dropout (NEW)** ⭐
- Added `--dropout 0.1` (default)
- Applied to all model layers during fine-tuning
- Critical for preventing overfitting

### 3. **Weight Decay**
- Changed default from `0.0` → `0.01`
- Provides L2 regularization
- Use `--weight_decay 0.01` explicitly

### 4. **Evaluation Frequency**
- Changed from every 5 epochs → **every 1 epoch**
- Catches peak performance before overfitting
- More data for debugging

### 5. **Gradient Clipping**
- Already set to `--max_grad_norm 1.0`
- Prevents gradient explosion
- Keeps training stable

## Recommended Training Commands

### Option 1: Conservative (RECOMMENDED) 🌟
Best chance to reach 0.65 F1 without overfitting:

```bash
python train_t5.py \
  --finetune \
  --learning_rate 3e-5 \
  --weight_decay 0.01 \
  --dropout 0.1 \
  --label_smoothing 0.1 \
  --batch_size 16 \
  --test_batch_size 16 \
  --max_n_epochs 30 \
  --eval_every_n_epochs 1 \
  --patience_epochs 5 \
  --scheduler_type cosine \
  --num_warmup_epochs 2 \
  --max_grad_norm 1.0 \
  --num_beams 4 \
  --max_gen_length 128 \
  --check_gradients \
  --experiment_name conservative_3e5_wd01_drop01_ls01
```

**Why this works:**
- Lower LR (3e-5): More stable, less overfitting
- Cosine scheduler: Smooth decay, better than linear
- Warmup: 2 epochs to stabilize
- Beam search (4): Better generation quality
- Shorter max_length (128): SQL queries don't need 512 tokens
- Early stopping (patience=5): Stops at peak

### Option 2: Aggressive (if Conservative fails)
Higher learning rate with strong regularization:

```bash
python train_t5.py \
  --finetune \
  --learning_rate 5e-5 \
  --weight_decay 0.02 \
  --dropout 0.15 \
  --label_smoothing 0.15 \
  --batch_size 16 \
  --test_batch_size 16 \
  --max_n_epochs 25 \
  --eval_every_n_epochs 1 \
  --patience_epochs 4 \
  --scheduler_type cosine \
  --num_warmup_epochs 1 \
  --max_grad_norm 1.0 \
  --num_beams 4 \
  --max_gen_length 128 \
  --experiment_name aggressive_5e5_wd02_drop15_ls15
```

### Option 3: With Schema Enhancement
If you want to try schema information:

```bash
python train_t5.py \
  --finetune \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --dropout 0.1 \
  --label_smoothing 0.1 \
  --batch_size 8 \
  --test_batch_size 8 \
  --max_n_epochs 30 \
  --eval_every_n_epochs 1 \
  --patience_epochs 5 \
  --scheduler_type cosine \
  --num_warmup_epochs 2 \
  --max_grad_norm 1.0 \
  --num_beams 4 \
  --max_gen_length 256 \
  --use_schema_enhancement \
  --experiment_name schema_2e5_wd01_drop01_ls01
```

**Note:** Schema adds ~500 tokens, so batch_size=8 to fit in memory

## What Changed in Code

### train_t5.py
1. Added `--dropout` argument (default: 0.1)
2. Added `--label_smoothing` argument (default: 0.1)
3. Changed `--eval_every_n_epochs` default from 5 → 1
4. Changed `--weight_decay` default from 0.0 → 0.01
5. Applied label smoothing to CrossEntropyLoss

### t5_utils.py
1. Apply dropout to model during initialization
2. Set dropout_rate in model.config
3. Apply dropout to all Dropout layers

## Monitoring During Training

Watch for these patterns:

### Good Signs ✅
- F1 steadily increases from 0.3 → 0.5 → 0.6+
- Train loss decreases smoothly
- Dev F1 follows train F1 closely (not diverging)
- Gradient norms stay < 5.0

### Warning Signs ⚠️
- F1 spikes then crashes (overfitting)
- Dev F1 peaks then drops >5% (stop training!)
- Gradient norms > 10 frequently
- 100% error rate on dev set

### Stop Training When:
- Dev F1 ≥ 0.65 (mission accomplished!)
- Patience reached (early stopping)
- Dev F1 drops >10% from peak (severe overfitting)

## Debugging Tips

### If F1 is still too low (< 0.5):
1. **Check SQL syntax errors**: Look at the sample predictions
2. **Reduce learning rate**: Try 2e-5 or 1e-5
3. **Increase num_beams**: Try 8 or 10
4. **Train longer**: Maybe 40-50 epochs with lower LR

### If F1 drops after epoch 10:
1. **Increase dropout**: Try 0.15 or 0.2
2. **Increase label_smoothing**: Try 0.15 or 0.2
3. **Reduce learning rate**: Try 2e-5
4. **Use early stopping**: patience=3 or 4

### If training is unstable:
1. **Check gradients**: Use `--check_gradients`
2. **Reduce learning rate**: Halve it
3. **Increase warmup**: Try 3-5 epochs
4. **Reduce batch size**: Try 8 instead of 16

## Expected Timeline

With proper hyperparameters:
- **Epoch 1-5**: F1 = 0.20-0.40 (learning basics)
- **Epoch 6-15**: F1 = 0.40-0.60 (improving)
- **Epoch 15-25**: F1 = 0.60-0.65+ (peak performance)
- **After peak**: Early stopping kicks in

Total training time on GPU: ~30-60 minutes

## Quick Comparison

| Hyperparameter | Your Old Settings | New Defaults | Why Changed |
|----------------|-------------------|--------------|-------------|
| weight_decay | 0.0 | 0.01 | Add regularization |
| dropout | N/A | 0.1 | Prevent overfitting |
| label_smoothing | N/A | 0.1 | Better generalization |
| learning_rate | 1e-4 | 5e-5 | More stable (suggest 3e-5) |
| scheduler | linear | cosine | Smoother decay |
| eval_every | 5 | 1 | Catch peak early |
| num_beams | 1 | 1 | Suggest 4 for quality |
| max_gen_length | 512 | 512 | Suggest 128 (SQL is short) |

## Final Recommendation

**Start with Option 1 (Conservative)** - it has the best chance of reaching 0.65+ F1:
- Learning rate: 3e-5
- Weight decay: 0.01
- Dropout: 0.1
- Label smoothing: 0.1
- Cosine scheduler
- Beam search: 4
- Eval every epoch
- Early stopping: patience=5

This should give you stable training that reaches 0.65+ by epoch 15-20.

Good luck! 🚀
