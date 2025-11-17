import os

import torch

import transformers
from transformers import T5ForConditionalGeneration, T5Config
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import wandb

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def setup_wandb(args):
    """Initialize Weights & Biases tracking with better error handling."""
    if not getattr(args, 'use_wandb', False):
        print("wandb disabled (--use_wandb not set)")
        return None

    try:
        mode = os.environ.get("WANDB_MODE", "online")
        project = os.environ.get("WANDB_PROJECT", "hw4-text-to-sql")
        run_name = f"{'ft' if args.finetune else 'scr'}-{args.experiment_name}"

        run = wandb.init(
            project=project,
            name=run_name,
            config=dict(vars(args)),
            mode=mode,
            reinit=True,
        )
        print(f"✅ wandb initialized: project={project}, name={run_name}")
        if run and hasattr(run, 'url'):
            print(f"🔗 wandb run: {run.url}")
        return run
    except Exception as e:
        print(f"⚠️ wandb init failed ({e}); running in offline mode.")
        os.environ["WANDB_MODE"] = "offline"
        run = wandb.init(project="hw4-text-to-sql", name=run_name, config=dict(vars(args)), mode="offline", reinit=True)
        return run

def initialize_model(args):
    '''
    Helper function to initialize the model. You should be either finetuning
    the pretrained model associated with the 'google-t5/t5-small' checkpoint
    or training a T5 model initialized with the 'google-t5/t5-small' config
    from scratch.
    '''
    if args.finetune:
        print("Initializing T5 model for fine-tuning...")
        model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-small')
        
        # Apply dropout if specified (helps prevent overfitting)
        dropout_rate = getattr(args, 'dropout', 0.1)
        if dropout_rate > 0:
            model.config.dropout_rate = dropout_rate
            # Apply to all dropout layers in the model
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = dropout_rate
            print(f"Applied dropout rate: {dropout_rate}")
        
        # Apply optional parameter freezing for fine-tuning
        apply_freezing(args, model)
    else:
        print("Initializing T5 model from scratch...")
        config = T5Config.from_pretrained('google-t5/t5-small')
        # Apply dropout to config for from-scratch training
        dropout_rate = getattr(args, 'dropout', 0.1)
        if dropout_rate > 0:
            config.dropout_rate = dropout_rate
            print(f"Set config dropout rate: {dropout_rate}")
        model = T5ForConditionalGeneration(config)
    
    model.to(DEVICE)
    print(f"Model moved to device: {DEVICE}")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model

def _freeze_module(module):
    """Helper to freeze all parameters in a module."""
    for p in module.parameters():
        p.requires_grad = False

def apply_freezing(args, model: T5ForConditionalGeneration):
    """Apply selective freezing to T5 components based on CLI args.
    
    Args considered:
      - freeze_all_encoder_layers
      - freeze_all_decoder_layers
      - freeze_encoder_n_layers (bottom N)
      - freeze_decoder_n_layers (bottom N)
      - freeze_embeddings
    """
    # Freeze embeddings (shared between encoder/decoder and tied with lm_head)
    if getattr(args, 'freeze_embeddings', False):
        if hasattr(model, 'shared'):
            _freeze_module(model.shared)
            print("Froze shared embeddings")
        if hasattr(model, 'lm_head') and model.lm_head.weight is model.shared.weight:
            model.lm_head.weight.requires_grad = False

    # Encoder freezing
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'block'):
        num_enc = len(model.encoder.block)
        if getattr(args, 'freeze_all_encoder_layers', False):
            for blk in model.encoder.block:
                _freeze_module(blk)
            if hasattr(model.encoder, 'final_layer_norm'):
                _freeze_module(model.encoder.final_layer_norm)
            print(f"Froze all {num_enc} encoder layers")
        else:
            n = min(max(getattr(args, 'freeze_encoder_n_layers', 0), 0), num_enc)
            if n > 0:
                for i in range(n):
                    _freeze_module(model.encoder.block[i])
                print(f"Froze bottom {n} encoder layers")

    # Decoder freezing
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'block'):
        num_dec = len(model.decoder.block)
        if getattr(args, 'freeze_all_decoder_layers', False):
            for blk in model.decoder.block:
                _freeze_module(blk)
            if hasattr(model.decoder, 'final_layer_norm'):
                _freeze_module(model.decoder.final_layer_norm)
            print(f"Froze all {num_dec} decoder layers")
        else:
            n = min(max(getattr(args, 'freeze_decoder_n_layers', 0), 0), num_dec)
            if n > 0:
                for i in range(n):
                    _freeze_module(model.decoder.block[i])
                print(f"Froze bottom {n} decoder layers")

def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass

def save_model(checkpoint_dir, model, best):
    """Save model checkpoint with config for easier loading."""
    mkdir(checkpoint_dir)
    
    if best:
        save_path = os.path.join(checkpoint_dir, 'best_model.pt')
        print(f"Saving best model to {save_path}")
    else:
        save_path = os.path.join(checkpoint_dir, 'latest_model.pt')
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model.config,
    }, save_path)

def load_model_from_checkpoint(args, best):
    """Load model from checkpoint with fallback directory logic."""
    model_type = 'ft' if args.finetune else 'scr'
    # Prefer explicit directory from args (set by training code)
    checkpoint_dir = getattr(args, 'checkpoint_dir', None)
    
    if not checkpoint_dir:
        # Try runs/ structure first (new layout)
        candidate = os.path.join('runs', f'{model_type}_experiments', args.experiment_name, 'checkpoints')
        if os.path.isdir(candidate):
            checkpoint_dir = candidate
        else:
            # Fall back to legacy checkpoints/ structure
            checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    
    if best:
        checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
    else:
        checkpoint_path = os.path.join(checkpoint_dir, 'latest_model.pt')
    
    print(f"Loading model from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Initialize model with same config
    if args.finetune:
        model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-small')
    else:
        model = T5ForConditionalGeneration(checkpoint['model_config'])
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    
    return model

def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler

def initialize_optimizer(args, model):
    decay_parameters = get_parameter_names(model, transformers.pytorch_utils.ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999)
        )
    else:
        pass

    return optimizer
        
def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        raise NotImplementedError

def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

