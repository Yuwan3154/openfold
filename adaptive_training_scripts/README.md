# Adaptive Training Scripts

Clean, comprehensive implementation for training adaptive weighting between original Evoformer blocks and pre-trained replacement blocks.

## Overview

This system enables training a model that dynamically weighs between:
- Original Evoformer blocks (pre-trained on full data)
- Replacement blocks (simpler, pre-trained separately)

The adaptive weight is predicted per-block based on the MSA representation: `w = sigmoid(linear(mean_pool(msa[..., 0, :, :])))`.

## Key Features

✅ **Weight Loading**: Supports both PyTorch (.pt) and JAX (.npz) checkpoints  
✅ **Template Compatibility**: Loads PTM weights with templates, trains without templates in data  
✅ **Single Sequence Mode**: Trains efficiently with single sequences (no MSA/templates)  
✅ **Replace Loss**: Encourages use of replacement blocks during training  
✅ **Structure Logging**: Logs predicted structures to wandb periodically  
✅ **Adaptive Metrics**: Tracks adaptive weights per block and statistics  
✅ **All Features**: Preserves all optional features (wandb, gradient accumulation, etc.)

## File Structure

```
adaptive_training_scripts/
├── __init__.py                  # Package initialization
├── run_adaptive_training.py     # Main entry point (run this)
├── adaptive_model.py            # Model loading & block replacement
├── adaptive_wrapper.py          # PyTorch Lightning training wrapper
├── utils.py                     # Data preparation utilities
└── README.md                    # This file
```

## Quick Start

### 1. Prepare Configuration

Create a YAML config file (or use existing one):

```yaml
# Example: AFdistill/configs/my_config.yaml
csv_path: data/af2rank_single/af2rank_single_set_single_tms_05.csv
pdb_dir: data/af2rank_single/pdb
weights_path: openfold/openfold/resources/openfold_params/finetuning_ptm_2.pt
trained_models_dir: AFdistill/pretrain_full
output_dir: AFdistill/my_adaptive_training

experiment_name: my_adaptive_training
linear_type: full
replace_loss_scaler: 1.0

batch_size: 1
learning_rate: 0.001
max_epochs: 500
train_epoch_len: 128
validation_fraction: 0.1
grad_accum_steps: 2

wandb: true
wandb_project: af2distill
wandb_entity: your-entity
log_structure_every_k_epoch: 5
```

### 2. Test with Dry Run

```bash
cd /home/jupyter-chenxi

python openfold/adaptive_training_scripts/run_adaptive_training.py \
    --config AFdistill/configs/my_config.yaml \
    --gpus 1 \
    --max_epochs 1 \
    --train_epoch_len 2 \
    --dry_run
```

### 3. Run Training

```bash
python openfold/adaptive_training_scripts/run_adaptive_training.py \
    --config AFdistill/configs/my_config.yaml
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--config` | Path to YAML config file | *Required* |
| `--gpus` | Override number of GPUs | From config |
| `--max_epochs` | Override max epochs | From config |
| `--train_epoch_len` | Override train epoch length | From config |
| `--dry_run` | Prepare but don't train | False |

## How It Works

### 1. Model Loading

```python
# Creates base model with finetuning_no_templ_ptm preset
config = model_config("finetuning_no_templ_ptm", train=True, low_prec=True)
model = AlphaFold(config)

# Loads PTM weights (with strict=False to allow template mismatches)
load_weights(model, "finetuning_ptm_2.pt")
```

### 2. Block Replacement

```python
# Replaces Evoformer blocks with adaptive versions
for block_idx in available_blocks:
    replacement_block = load_pretrained_replacement_block(...)
    weight_predictor = AdaptiveWeightPredictor(c_m=256)
    adaptive_block = AdaptiveEvoformerBlock(
        original_block, replacement_block, weight_predictor
    )
    model.evoformer.blocks[block_idx] = adaptive_block
```

### 3. Training

```python
# Freezes all parameters except:
# - Adaptive weight predictors
# - Replacement blocks

# Training loss:
loss = main_loss + replace_loss_scaler * mean(adaptive_weights)
```

## Weight Loading Details

### PyTorch Checkpoints (.pt)

- Uses `model.load_state_dict(state_dict, strict=False)`
- Missing template keys are expected and ignored
- Logs warnings for other missing/unexpected keys

### JAX Parameters (.npz)

- Uses OpenFold's `import_jax_weights_` function
- Requires model created with compatible config preset
- Automatically detects version from filename (e.g., "model_2_ptm")

## Configuration Options

### Required

- `csv_path`: CSV with chain list (natives_rcsb column)
- `pdb_dir`: Directory with structure files (.cif/.pdb)
- `weights_path`: Pre-trained weights (.pt or .npz)
- `trained_models_dir`: Pre-trained replacement blocks
- `output_dir`: Output directory for checkpoints/logs

### Model

- `linear_type`: Type of replacement blocks ("full", "diagonal", "affine")
- `replace_loss_scaler`: Weight for replace loss (default: 1.0)

### Training

- `max_epochs`: Maximum epochs
- `train_epoch_len`: Steps per epoch
- `batch_size`: Batch size (usually 1)
- `learning_rate`: Learning rate (default: 1e-3)
- `validation_fraction`: Validation split fraction (default: 0.1)
- `grad_accum_steps`: Gradient accumulation steps (default: 1)

### Logging

- `experiment_name`: Experiment name
- `wandb`: Enable wandb logging (default: false)
- `wandb_project`: Wandb project name
- `wandb_entity`: Wandb entity/username
- `log_structure_every_k_epoch`: Structure logging frequency (0=disabled)

## Output Structure

```
output_dir/
├── checkpoints/
│   └── best_model.ckpt          # Best model checkpoint
├── chain_lists/
│   ├── all_chains.txt           # All available chains
│   ├── training_chains.txt      # Training split
│   └── validation_chains.txt    # Validation split
├── alignments/                   # Minimal alignment files
│   ├── 1abc.a3m
│   └── ...
└── lightning_logs/               # PyTorch Lightning logs
    └── version_X/
```

## Metrics Logged

### Training/Validation

- `loss`: Total loss (main + replace)
- `replace_loss`: Raw replace loss (mean adaptive weight)
- `replace_loss_scaled`: Scaled replace loss
- All OpenFold sub-losses (fape, plddt, tm, etc.)

### Adaptive Metrics

- `mean_adaptive_weight`: Mean weight across blocks
- `adaptive_weight_std`: Standard deviation
- `adaptive_weight_min`: Minimum weight
- `adaptive_weight_max`: Maximum weight

### Validation Metrics

- `lddt_ca`: CA-only lDDT score
- `drmsd_ca`: CA-only DRMSD
- `alignment_rmsd`: RMSD after superimposition
- `gdt_ts`: GDT-TS score
- `gdt_ha`: GDT-HA score

## Troubleshooting

### Import Errors

Make sure you're in the openfold directory:
```bash
cd /home/jupyter-chenxi/openfold
python adaptive_training_scripts/run_adaptive_training.py ...
```

### Missing Weights

Check that:
- PTM checkpoint exists at `weights_path`
- Pre-trained replacement blocks exist in `trained_models_dir/block_XX/linear_type/best_model.ckpt`

### Template Errors

This system uses `finetuning_no_templ_ptm` preset, which should avoid template-related errors. If you still see template errors, check that:
- Config has `use_templates: False`
- Template model is disabled in config

### NCCL vs Gloo

For single-GPU training, gloo is automatically used. For multi-GPU, specify in config:
```yaml
distributed_backend: gloo  # or nccl
```

## Credits

Built on top of:
- OpenFold (AlphaFold implementation)
- PyTorch Lightning (training framework)
- Pre-trained replacement blocks (from block_replacement_scripts)

