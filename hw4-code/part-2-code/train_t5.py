import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb
from transformers import GenerationConfig, T5Tokenizer
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records
from eval_utils import eval_epoch as eval_epoch_util

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")
    
    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=0,
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=0,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=0,
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)

    # Freezing options (for fine-tuning)
    parser.add_argument('--freeze_all_encoder_layers', action='store_true',
                        help='Freeze all encoder layers')
    parser.add_argument('--freeze_all_decoder_layers', action='store_true',
                        help='Freeze all decoder layers')
    parser.add_argument('--freeze_encoder_n_layers', type=int, default=0,
                        help='Freeze the bottom N encoder layers')
    parser.add_argument('--freeze_decoder_n_layers', type=int, default=0,
                        help='Freeze the bottom N decoder layers')
    parser.add_argument('--freeze_embeddings', action='store_true',
                        help='Freeze shared embeddings (and tied lm_head)')

    # Logging
    parser.add_argument('--log_every', type=int, default=200,
                        help='How many training steps between progress prints')

    # Generation and reranking options
    parser.add_argument('--num_beams', type=int, default=1,
                        help='Number of beams for beam search (1 = greedy)')
    parser.add_argument('--num_candidates', type=int, default=4,
                        help='Number of candidates to generate and rerank')
    parser.add_argument('--max_gen_length', type=int, default=256,
                        help='Maximum generation length for SQL queries')
    parser.add_argument('--rerank_by_execution', action='store_true',
                        help='Rerank candidates by executing them on the database')
    parser.add_argument('--use_schema_enhancement', action='store_true',
                        help='Use enhanced input with database schema information and Answer: pattern')
    parser.add_argument('--eval_every_n_epochs', type=int, default=1,
                        help='Evaluate on dev set every N epochs (default: 1 = every epoch)')

    args = parser.parse_args()
    return args

def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1
    epochs_since_improvement = 0

    model_type = 'ft' if args.finetune else 'scr'
    experiment_name = args.experiment_name

    # Single run directory to hold everything for this run
    run_dir = os.path.join('runs', f'{model_type}_experiments', experiment_name)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    results_dir = os.path.join(run_dir, 'results')
    records_dir = os.path.join(run_dir, 'records')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(records_dir, exist_ok=True)

    # Attach for checkpoint saving/loading
    args.checkpoint_dir = ckpt_dir
    # Dev outputs for this run (define early so we can print them below)
    model_sql_path = os.path.join(results_dir, 'dev.sql')
    model_record_path = os.path.join(records_dir, 'dev.pkl')
    print("\n=== Run setup ===")
    print(f"Mode: {'fine-tune' if args.finetune else 'scratch'}")
    print(f"Experiment: {experiment_name}")
    print(f"Run dir: {run_dir}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Dev outputs -> SQL: {model_sql_path} | Records: {model_record_path}")
    print(f"Test outputs -> will be saved under: {results_dir} / {records_dir}")
    print(f"Batch sizes: train={args.batch_size}, dev={args.test_batch_size}")
    print("=================\n")

    # Ground-truth files (fixed locations from starter)
    gt_sql_path = os.path.join('data', 'dev.sql')
    gt_record_path = os.path.join('records', 'ground_truth_dev.pkl')

    # Ensure ground-truth dev records exist (compute once if missing)
    if not os.path.exists(gt_record_path):
        from utils import read_queries, compute_records
        print(f"Ground-truth records not found at {gt_record_path}. Computing once from {gt_sql_path} ...")
        gt_qs = read_queries(gt_sql_path)
        gt_recs, gt_errs = compute_records(gt_qs)
        os.makedirs(os.path.dirname(gt_record_path), exist_ok=True)
        import pickle
        with open(gt_record_path, 'wb') as f:
            pickle.dump((gt_recs, gt_errs), f)
        err_count = sum(1 for e in gt_errs if e)
        print(f"Saved GT records to {gt_record_path} (errors: {err_count}/{len(gt_errs)})")
    for epoch in range(args.max_n_epochs):
        # Report LR at epoch start
        current_lr = optimizer.param_groups[0]['lr'] if optimizer.param_groups else -1
        print(f"Epoch {epoch}: starting, learning rate={current_lr:.6f}")

        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        # Evaluate based on user-specified frequency (or on the last epoch)
        should_evaluate = (epoch % args.eval_every_n_epochs == 0) or (epoch == args.max_n_epochs - 1)
        
        if should_evaluate:
            print(f"Running dev evaluation at epoch {epoch}...")
            eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(args, model, dev_loader,
                                                                             gt_sql_path, model_sql_path,
                                                                             gt_record_path, model_record_path)
            print(f"Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
            print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")
        else:
            print(f"Skipping dev evaluation at epoch {epoch} (evaluating every {args.eval_every_n_epochs} epochs)")
            # Use previous best values for tracking
            eval_loss, record_f1, record_em, sql_em, error_rate = 0.0, best_f1, 0.0, 0.0, 0.0

        if args.use_wandb:
            result_dict = {
                'train/loss' : tr_loss,
                'train/lr' : current_lr,
            }
            # Only log dev metrics when we actually evaluated
            if should_evaluate:
                result_dict.update({
                    'dev/loss' : eval_loss,
                    'dev/record_f1' : record_f1,
                    'dev/record_em' : record_em,
                    'dev/sql_em' : sql_em,
                    'dev/error_rate' : error_rate,
                })
            wandb.log(result_dict, step=epoch)
            
            # Only save artifacts when we actually evaluated
            if should_evaluate:
                try:
                    wandb.save(model_sql_path)
                    wandb.save(model_record_path)
                except Exception:
                    pass

        # Only update best model when we actually evaluated
        if should_evaluate:
            # Ignore F1 scores of 1.0 as they're likely false positives from SQL error edge cases
            if record_f1 >= 0.99999:
                print(f"Epoch {epoch}: Ignoring suspicious F1 score of {record_f1:.6f} (likely false positive)")
                epochs_since_improvement += 1
                # Save as best if we have no best model yet (prevent crash later)
                if best_f1 < 0:
                    print(f"Epoch {epoch}: No previous best model, saving this one to prevent crash")
                    best_f1 = 0.01  # Small non-suspicious value
                    epochs_since_improvement = 0
            elif record_f1 > best_f1:
                best_f1 = record_f1
                epochs_since_improvement = 0
                print(f"Epoch {epoch}: New best Record F1 = {best_f1:.4f} — saving best model")
            else:
                epochs_since_improvement += 1
        else:
            # Don't count skipped evaluations toward patience
            print(f"Epoch {epoch}: Skipped evaluation, patience counter unchanged ({epochs_since_improvement})")

        save_model(args.checkpoint_dir, model, best=False)
        if should_evaluate and epochs_since_improvement == 0:
            save_model(args.checkpoint_dir, model, best=True)

        # Only check patience when we actually evaluated
        if should_evaluate and epochs_since_improvement >= args.patience_epochs:
            print(f"Early stopping: no improvement for {args.patience_epochs} evaluations")
            break

def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()
    step = 0
    running_loss = 0.0
    running_tokens = 0

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        logits = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
        )['logits']

        non_pad = decoder_targets != PAD_IDX
        loss = criterion(logits[non_pad], decoder_targets[non_pad])
        loss.backward()
        optimizer.step()
        if scheduler is not None: 
            scheduler.step()

        with torch.no_grad():
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            running_loss += loss.item() * num_tokens
            running_tokens += num_tokens
            step += 1

            if args.log_every > 0 and step % args.log_every == 0:
                avg = running_loss / running_tokens if running_tokens > 0 else 0.0
                print(f"  Step {step}: avg token-CE loss over last {args.log_every} steps = {avg:.4f}")
                running_loss = 0.0
                running_tokens = 0

    return total_loss / total_tokens
        
def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    '''
    Reuses eval_utils.eval_epoch for generation/metrics and also computes CE loss on dev set.
    Returns: avg_loss, record_f1, record_em, sql_em, error_rate
    '''
    model.eval()

    # 1) Compute CE loss on dev
    total_loss = 0.0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(dev_loader, desc="Eval loss"):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)

            logits = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            )["logits"]

            non_pad = decoder_targets != PAD_IDX
            loss = criterion(logits[non_pad], decoder_targets[non_pad])
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = (total_loss / total_tokens) if total_tokens > 0 else 0.0

    # 2) Use eval_utils to generate predictions; then save and score with provided scripts
    # Use SQL-optimized tokenizer if available, otherwise default
    from transformers import T5TokenizerFast
    sql_tokenizer_path = "./sql_optimized_tokenizer"
    if os.path.exists(sql_tokenizer_path):
        print("🚀 Using SQL-optimized tokenizer for evaluation")
        tokenizer = T5TokenizerFast.from_pretrained(sql_tokenizer_path)
    else:
        print("📊 Using default tokenizer for evaluation")
        tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-small')
    f1_from_util, predictions = eval_epoch_util(
        model=model,
        dataloader=dev_loader,
        tokenizer=tokenizer,
        device=DEVICE,
        generation_max_length=getattr(args, 'max_gen_length', 256),
        num_beams=getattr(args, 'num_beams', 1),
        num_candidates=getattr(args, 'num_candidates', 4) if getattr(args, 'rerank_by_execution', False) else 1,
        rerank_by_execution=getattr(args, 'rerank_by_execution', False),
        return_predictions=True,
    )

    # Save queries to disk and compute records for downstream metrics
    save_queries_and_records(predictions, model_sql_path, model_record_path)

    # Compute EM/F1 with official helper (uses GT SQL + GT records if provided)
    sql_em, record_em, record_f1, error_msgs = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_path, model_record_path
    )

    error_count = sum(1 for msg in error_msgs if msg)
    error_rate = error_count / len(error_msgs) if error_msgs else 0.0

    # 3) Optionally log a sample table of predictions vs targets to wandb
    if getattr(args, 'use_wandb', False):
        try:
            import random
            # Load NL and GT SQL for the dev set (order-aligned with predictions)
            from load_data import load_lines
            dev_nl = load_lines(os.path.join('data', 'dev.nl'))
            dev_sql = load_lines(os.path.join('data', 'dev.sql'))
            n = min(20, len(predictions), len(dev_sql), len(dev_nl))
            # Pick first n or random n samples
            indices = list(range(len(predictions)))
            random.shuffle(indices)
            indices = indices[:n]

            table = wandb.Table(columns=[
                'id', 'nl', 'target_sql', 'pred_sql', 'sql_em', 'error_msg'
            ])
            for i in indices:
                gt = dev_sql[i]
                pred = predictions[i]
                sql_match = int(gt == pred)
                err = error_msgs[i] if i < len(error_msgs) else ""
                table.add_data(i, dev_nl[i], gt, pred, sql_match, err)
            wandb.log({'dev/samples': table})
        except Exception as e:
            print(f"wandb table logging skipped: {e}")

    return avg_loss, record_f1, record_em, sql_em, error_rate
        
def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''
    Reuses eval_utils.eval_epoch to generate predictions on test set, then saves SQL and records.
    '''
    model.eval()
    
    # Use SQL-optimized tokenizer if available, otherwise default
    from transformers import T5TokenizerFast
    sql_tokenizer_path = "./sql_optimized_tokenizer"
    if os.path.exists(sql_tokenizer_path):
        print("🚀 Using SQL-optimized tokenizer for test inference")
        tokenizer = T5TokenizerFast.from_pretrained(sql_tokenizer_path)
    else:
        print("📊 Using default tokenizer for test inference")
        tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-small')

    # Generate only; F1 will be None since no targets in test loader
    _, predictions = eval_epoch_util(
        model=model,
        dataloader=test_loader,
        tokenizer=tokenizer,
        device=DEVICE,
        generation_max_length=getattr(args, 'max_gen_length', 256),
        num_beams=getattr(args, 'num_beams', 1),
        num_candidates=getattr(args, 'num_candidates', 4) if getattr(args, 'rerank_by_execution', False) else 1,
        rerank_by_execution=getattr(args, 'rerank_by_execution', False),
        return_predictions=True,
    )

    # Save SQL and execute to records for submission
    save_queries_and_records(predictions, model_sql_path, model_record_path)
    
    print(f"Test inference completed. Generated {len(predictions)} predictions.")
    print(f"Saved test files: {model_sql_path}, {model_record_path}")
    
    # Upload to wandb if enabled
    if hasattr(args, 'use_wandb') and args.use_wandb:
        try:
            print("Uploading test files to wandb...")
            import wandb
            wandb.save(model_sql_path)
            print("Uploaded test.sql to wandb")
            wandb.save(model_record_path)
            print("Uploaded test.pkl to wandb")
            print(f"✓ All test files uploaded to wandb successfully")
        except Exception as e:
            print(f"wandb test file upload failed: {e}")
    
    print("Test inference function completed.")

    return predictions

def main():
    # Get key arguments
    args = get_args()
    if args.use_wandb:
        # Recommended: Using wandb (or tensorboard) for result logging can make experimentation easier
        setup_wandb(args)

    # Load the data and the model
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    # Train 
    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Evaluate
    model = load_model_from_checkpoint(args, best=True)
    model.eval()
    
    # Dev set
    experiment_name = args.experiment_name
    model_type = 'ft' if args.finetune else 'scr'
    run_dir = os.path.join('runs', f'{model_type}_experiments', experiment_name)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    results_dir = os.path.join(run_dir, 'results')
    records_dir = os.path.join(run_dir, 'records')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(records_dir, exist_ok=True)
    # Ensure subsequent loads know where to look
    args.checkpoint_dir = ckpt_dir

    # Dev evaluation is already done during training, no need to repeat

    # Test set
    model_sql_path = os.path.join(results_dir, 'test.sql')
    model_record_path = os.path.join(records_dir, 'test.pkl')
    test_inference(args, model, test_loader, model_sql_path, model_record_path)

if __name__ == "__main__":
    main()