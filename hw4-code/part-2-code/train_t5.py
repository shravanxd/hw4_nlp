import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb
from transformers import GenerationConfig
from load_data import load_t5_data, extract_sql_before_end
from utils import compute_metrics, save_queries_and_records

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
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=1,
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=50,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=10,
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")
    
    parser.add_argument('--eval_every_n_epochs', type=int, default=5,
                        help="Evaluate on dev set every N epochs (default: 5)")

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

    # Generation options
    parser.add_argument('--num_beams', type=int, default=1,
                        help='Number of beams for beam search (1 = greedy)')
    parser.add_argument('--max_gen_length', type=int, default=512,
                        help='Maximum generation length for SQL queries')
    
    # Data enhancement
    parser.add_argument('--use_schema_enhancement', action='store_true',
                        help='Use enhanced input format with Question:/Schema:/Answer: pattern')

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
    # Dev outputs for this run
    model_sql_path = os.path.join(results_dir, 'dev.sql')
    model_record_path = os.path.join(records_dir, 'dev.pkl')
    
    print("\n=== Run setup ===")
    print(f"Mode: {'fine-tune' if args.finetune else 'scratch'}")
    print(f"Experiment: {experiment_name}")
    print(f"Run dir: {run_dir}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Dev outputs -> SQL: {model_sql_path} | Records: {model_record_path}")
    print(f"Batch sizes: train={args.batch_size}, dev={args.test_batch_size}")
    print("=================\n")

    # Ground-truth files (fixed locations from starter)
    gt_sql_path = 'data/dev.sql'
    gt_record_path = 'records/ground_truth_dev.pkl'

    # Ensure ground-truth dev records exist (compute once if missing)
    if not os.path.exists(gt_record_path):
        from utils import read_queries, compute_records
        print(f"Ground-truth records not found at {gt_record_path}. Computing once from {gt_sql_path} ...")
        gt_qs = read_queries(gt_sql_path)
        print(f"Loaded {len(gt_qs)} ground truth SQL queries")
        print(f"Executing ground truth queries to generate records...")
        gt_recs, gt_errs = compute_records(gt_qs)
        os.makedirs(os.path.dirname(gt_record_path), exist_ok=True)
        import pickle
        with open(gt_record_path, 'wb') as f:
            pickle.dump((gt_recs, gt_errs), f)
        err_count = sum(1 for e in gt_errs if e)
        print(f"✅ Ground truth execution complete:")
        print(f"   Total queries: {len(gt_qs)}")
        print(f"   Successful: {len(gt_qs) - err_count}")
        print(f"   Failed: {err_count}")
        if err_count > 0:
            print(f"   Error rate: {100*err_count/len(gt_qs):.1f}%")
        print(f"Saved GT records to {gt_record_path}")
    else:
        # Load and report on existing GT records
        import pickle
        with open(gt_record_path, 'rb') as f:
            gt_recs, gt_errs = pickle.load(f)
        err_count = sum(1 for e in gt_errs if e)
        print(f"\n✅ Ground Truth Statistics:")
        print(f"   Total queries: {len(gt_recs)}")
        print(f"   Successfully executed: {len(gt_recs) - err_count}")
        print(f"   Execution failed: {err_count}")
        if err_count > 0:
            print(f"   Ground truth error rate: {100*err_count/len(gt_recs):.1f}%")
        else:
            print(f"   Ground truth error rate: 0.0%")
        print()
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
        else:
            print(f"Skipping dev evaluation at epoch {epoch} (evaluating every {args.eval_every_n_epochs} epochs)")
            # Use previous values for non-evaluation epochs
            eval_loss, record_f1, record_em, sql_em, error_rate = 0.0, best_f1, 0.0, 0.0, 0.0
        
        # Load GT statistics for comparison (only when evaluating)
        if should_evaluate:
            import pickle
            with open(gt_record_path, 'rb') as f:
                gt_recs, gt_errs = pickle.load(f)
            gt_err_count = sum(1 for e in gt_errs if e)
            gt_success = len(gt_recs) - gt_err_count
            
            # Calculate detailed error statistics for model
            num_queries = 466  # dev set size
            num_errors = int(error_rate * num_queries)
            num_success = num_queries - num_errors
            
            print(f"\nEpoch {epoch} Results:")
            print(f"=" * 70)
            print(f"Dev loss: {eval_loss:.4f} | Record F1: {record_f1:.4f} | Record EM: {record_em:.4f} | SQL EM: {sql_em:.4f}")
            print(f"\n📊 Ground Truth SQL Execution:")
            print(f"   Total queries: {len(gt_recs)}")
            print(f"   Successfully executed: {gt_success}/{len(gt_recs)}")
            print(f"   Execution failed: {gt_err_count}/{len(gt_recs)}")
            print(f"   Ground truth error rate: {100*gt_err_count/len(gt_recs):.1f}%")
            print(f"\n📊 Model SQL Execution:")
            print(f"   Total queries generated: {num_queries}")
            print(f"   Successfully executed: {num_success}/{num_queries}")
            print(f"   Execution failed: {num_errors}/{num_queries}")
            print(f"   Model error rate: {error_rate*100:.1f}%")
            print(f"=" * 70 + "\n")

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
            # Ignore F1 scores of 1.0 as they're likely false positives
            if record_f1 >= 0.99999:
                print(f"Epoch {epoch}: Ignoring suspicious F1 score of {record_f1:.6f} (likely false positive)")
                epochs_since_improvement += 1
                if best_f1 < 0:
                    best_f1 = 0.01  # Prevent crash on first epoch
                    epochs_since_improvement = 0
            elif record_f1 > best_f1:
                best_f1 = record_f1
                epochs_since_improvement = 0
                print(f"Epoch {epoch}: New best Record F1 = {best_f1:.4f} — saving best model")
            else:
                epochs_since_improvement += 1
        else:
            # Don't count skipped evaluations toward patience
            pass

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

    return total_loss / total_tokens
        
def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    '''
    You must implement the evaluation loop to be using during training. We recommend keeping track
    of the model loss on the SQL queries, the metrics compute_metrics returns (save_queries_and_records should be helpful)
    and the model's syntax error rate. 

    To compute non-loss metrics, you will need to perform generation with the model. Greedy decoding or beam search
    should both provide good results. If you find that this component of evaluation takes too long with your compute,
    we found the cross-entropy loss (in the evaluation set) to be well (albeit imperfectly) correlated with F1 performance.
    '''
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()
    
    generated_queries = []
    
    from transformers import T5TokenizerFast
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    with torch.no_grad():
        for encoder_input, encoder_mask, decoder_input, decoder_targets, initial_decoder_input in tqdm(dev_loader):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)
            
            # Compute loss
            logits = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            )['logits']
            
            non_pad = decoder_targets != PAD_IDX
            loss = criterion(logits[non_pad], decoder_targets[non_pad])
            
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            
            # Generate predictions
            outputs = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=getattr(args, 'max_gen_length', 512),
                num_beams=1,
                early_stopping=True
            )
            
            # Decode generated outputs and extract SQL before END token
            for output in outputs:
                decoded = tokenizer.decode(output, skip_special_tokens=True)
                # Extract SQL before END token
                sql_query = extract_sql_before_end(decoded)
                generated_queries.append(sql_query)
    
    avg_loss = total_loss / total_tokens
    
    # Save generated queries and compute metrics
    save_queries_and_records(generated_queries, model_sql_path, model_record_path)
    
    sql_em, record_em, record_f1, error_msgs = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_path, model_record_path
    )
    
    # Calculate error rate
    error_count = sum(1 for msg in error_msgs if msg != "")
    error_rate = error_count / len(error_msgs) if len(error_msgs) > 0 else 0
    
    # Print detailed generation statistics
    print(f"\n{'='*70}")
    print(f"📊 Generation Statistics for Current Evaluation:")
    print(f"{'='*70}")
    print(f"Total queries generated: {len(generated_queries)}")
    print(f"Successfully executed: {len(error_msgs) - error_count}/{len(error_msgs)}")
    print(f"Execution failed: {error_count}/{len(error_msgs)}")
    print(f"Model error rate: {error_rate*100:.1f}%")
    print(f"{'='*70}")
    
    # Show sample errors if any
    if error_count > 0 and error_count <= 5:
        print(f"\n⚠️  All {error_count} errors:")
        for i, msg in enumerate(error_msgs):
            if msg != "":
                print(f"\n   Query {i}:")
                print(f"   Generated: {generated_queries[i][:120]}...")
                print(f"   Error: {msg[:100]}")
    elif error_count > 5:
        print(f"\n⚠️  Sample of first 3 errors (out of {error_count} total):")
        shown = 0
        for i, msg in enumerate(error_msgs):
            if msg != "" and shown < 3:
                print(f"\n   Query {i}:")
                print(f"   Generated: {generated_queries[i][:120]}...")
                print(f"   Error: {msg[:100]}")
                shown += 1
    else:
        print(f"\n✅ All queries executed successfully!")
    
    print(f"{'='*70}\n")

    return avg_loss, record_f1, record_em, sql_em, error_rate
        
def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''
    You must implement inference to compute your model's generated SQL queries and its associated 
    database records. Implementation should be very similar to eval_epoch.
    '''
    model.eval()
    generated_queries = []
    
    from transformers import T5TokenizerFast
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    with torch.no_grad():
        for encoder_input, encoder_mask, initial_decoder_input in tqdm(test_loader):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            
            # Generate predictions
            outputs = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=getattr(args, 'max_gen_length', 128),
                num_beams=getattr(args, 'num_beams', 4),
                early_stopping=True
            )
            
            # Decode generated outputs and extract SQL before END token
            for output in outputs:
                decoded = tokenizer.decode(output, skip_special_tokens=True)
                # Extract SQL before END token
                sql_query = extract_sql_before_end(decoded)
                generated_queries.append(sql_query)
    
    # Save generated queries and compute records
    save_queries_and_records(generated_queries, model_sql_path, model_record_path)
    print(f"Saved test predictions to {model_sql_path} and {model_record_path}")

def main():
    # Get key arguments
    args = get_args()
    
    # Validate arguments
    if args.max_n_epochs == 0:
        print("Warning: max_n_epochs is 0. No training will be performed.")
        print("Set --max_n_epochs to a positive value (e.g., 10) to train the model.")
        print("\nExample usage:")
        print("python train_t5.py --finetune --max_n_epochs 10 --experiment_name my_experiment")
        return
    
    if args.use_wandb:
        # Recommended: Using wandb (or tensorboard) for result logging can make experimentation easier
        setup_wandb(args)

    # Load the data and the model
    train_loader, dev_loader, test_loader = load_t5_data(
        args.batch_size, 
        args.test_batch_size,
        use_schema_enhancement=args.use_schema_enhancement
    )
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    # Train 
    if args.max_n_epochs > 0:
        train(args, model, train_loader, dev_loader, optimizer, scheduler)
        # Evaluate - load best checkpoint after training
        model = load_model_from_checkpoint(args, best=True)
    
    model.eval()
    
    # Setup directories for final evaluation
    experiment_name = args.experiment_name
    model_type = 'ft' if args.finetune else 'scr'
    run_dir = os.path.join('runs', f'{model_type}_experiments', experiment_name)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    results_dir = os.path.join(run_dir, 'results')
    records_dir = os.path.join(run_dir, 'records')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(records_dir, exist_ok=True)
    args.checkpoint_dir = ckpt_dir

    # Dev evaluation is already done during training
    print("\n=== Final Test Inference ===")

    # Test set
    model_sql_path = os.path.join(results_dir, 'test.sql')
    model_record_path = os.path.join(records_dir, 'test.pkl')
    test_inference(args, model, test_loader, model_sql_path, model_record_path)

if __name__ == "__main__":
    main()
