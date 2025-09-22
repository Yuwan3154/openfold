# Evoformer Replacement Block Training Pipeline

This pipeline implements a comprehensive approach to training and evaluating replacement blocks for OpenFold's Evoformer blocks, with the goal of creating adaptive models that can dynamically choose between expensive full Evoformer computation and cheaper replacement blocks.

## Overview

The pipeline consists of four main steps:

1. **Data Collection**: Collect input/output pairs for all 48 Evoformer blocks using recycle 0 values
2. **Replacement Training**: Train 48 separate replacement blocks with 3 different linear layer types
3. **Model Evaluation**: Evaluate replacement blocks in the full model using TM scores vs ground truth
4. **Adaptive Training**: Train an adaptive weighting model that learns when to use replacement vs original blocks

## Pipeline Components

### Step 1: Data Collection (`collect_block_data.py`)

Collects input/output pairs for all 48 Evoformer blocks during forward passes.

**Key Features:**
- Uses forward hooks to capture block inputs and outputs
- Processes proteins from CSV file with 80/20 train/val split
- Uses recycle 0 values only (first iteration)
- Saves data separately for each block position

**Usage:**
```bash
python openfold/collect_block_data.py \
  --csv_path data/af2rank_single/af2rank_single_set_single_tms_07.csv \
  --pdb_dir data/af2rank_single/pdb \
  --weights_path params/params_model_2_ptm.npz \
  --output_dir replacement_block_data \
  --max_proteins 50  # Optional: limit for testing
```

### Step 2: Replacement Training (`train_replacement_blocks.py`)

Trains 48 separate replacement blocks (one for each Evoformer position) using the collected data.

**Key Features:**
- Tests 3 linear layer types: `full`, `diagonal`, `affine`
- Uses PyTorch Lightning for training
- Implements early stopping and learning rate scheduling
- Compares convergence speed and validation loss

**Linear Layer Types:**
- **Full**: Standard linear layers (`nn.Linear`)
- **Diagonal**: Diagonal weight matrices (parameter-efficient)
- **Affine**: Simple affine transformations (most efficient)

**Usage:**
```bash
python openfold/train_replacement_blocks.py \
  --data_dir replacement_block_data \
  --output_dir trained_replacement_models \
  --hidden_dim 256 \
  --batch_size 4 \
  --max_epochs 50 \
  --learning_rate 1e-3 \
  --test_blocks 23 24  # Optional: test specific blocks only
```

### Step 3: Model Evaluation (`evaluate_replacement_blocks.py`)

Evaluates trained replacement blocks in the full OpenFold model using TM scores against ground truth structures.

**Key Features:**
- Tests replacement blocks in actual model performance
- Compares against "block removed" baseline
- Uses USalign for TM score calculation
- Evaluates all block positions and linear types

**Usage:**
```bash
python openfold/evaluate_replacement_blocks.py \
  --csv_path data/af2rank_single/af2rank_single_set_single_tms_07.csv \
  --pdb_dir data/af2rank_single/pdb \
  --weights_path params/params_model_2_ptm.npz \
  --trained_models_dir trained_replacement_models \
  --output_dir evaluation_results \
  --hidden_dim 256
```

### Step 4: Adaptive Training (`train_adaptive_weighting.py`)

Trains an adaptive OpenFold model where each block output is a weighted combination of original and replacement blocks.

**Key Features:**
- Weight prediction: `w = sigmoid(mean_pool(linear(single_representation)))`
- 48 separate weight predictors (one per block)
- Replace loss: `mean(all_block_weights)` to encourage replacement usage
- Initializes replacement blocks with best weights from Step 2

**Architecture:**
```
For each Evoformer block i:
  evo_output_i, pair_output_i = original_block_i(inputs)
  repl_output_i, repl_pair_i = replacement_block_i(inputs)
  weight_i = sigmoid(linear_i(mean_pool(single_repr)))
  
  final_output_i = weight_i * evo_output_i + (1 - weight_i) * repl_output_i
  final_pair_i = weight_i * pair_output_i + (1 - weight_i) * repl_pair_i

Total Loss = AlphaFold_Loss + λ * mean(all_weights)
```

**Usage:**
```bash
python openfold/train_adaptive_weighting.py \
  --csv_path data/af2rank_single/af2rank_single_set_single_tms_07.csv \
  --pdb_dir data/af2rank_single/pdb \
  --weights_path params/params_model_2_ptm.npz \
  --trained_models_dir trained_replacement_models \
  --output_dir adaptive_weighting_results \
  --batch_size 1 \
  --max_epochs 10 \
  --replace_loss_weight 0.1
```

## Complete Pipeline Runner (`run_replacement_pipeline.py`)

Orchestrates the entire pipeline with a single command and configuration file.

**Create default config:**
```bash
python openfold/run_replacement_pipeline.py --create_config
```

**Run full pipeline:**
```bash
python openfold/run_replacement_pipeline.py --config replacement_pipeline_config.yaml
```

**Run specific steps:**
```bash
python openfold/run_replacement_pipeline.py --config config.yaml --steps 1 2  # Only data collection and training
```

## Configuration

Example configuration file (`replacement_pipeline_config.yaml`):

```yaml
# Data paths (relative to home directory)
csv_path: data/af2rank_single/af2rank_single_set_single_tms_07.csv
pdb_dir: data/af2rank_single/pdb
weights_path: params/params_model_2_ptm.npz
base_dir: replacement_block_pipeline

# Training parameters
hidden_dim: 256
batch_size: 4
max_epochs: 50
learning_rate: 0.001
weight_decay: 0.0001
num_workers: 4

# Adaptive weighting parameters
adaptive_batch_size: 1
adaptive_max_epochs: 10
adaptive_learning_rate: 0.0001
replace_loss_weight: 0.1

# Testing parameters (optional)
max_proteins: null  # Set to small number for testing
test_blocks: null   # Set to [23, 24] for testing specific blocks

# Which steps to run
steps: [1, 2, 3, 4]  # All steps
```

## Expected Results

### Step 2 Results
- Training summary showing convergence for different linear types
- Best performing linear type per block position
- Validation loss comparison across architectures

### Step 3 Results
- TM score comparison: replacement vs removed vs original
- Performance analysis by block position
- Identification of blocks that work well with replacements

### Step 4 Results
- Adaptive model that learns when to use replacements
- Weight prediction analysis showing model confidence
- Computational cost reduction while maintaining accuracy

## Key Insights Expected

1. **Block Importance**: Some Evoformer blocks may be more critical than others
2. **Linear Type Efficiency**: Comparison of full vs diagonal vs affine transformations
3. **Adaptive Behavior**: Model learning to use cheap replacements when confident
4. **Speed vs Accuracy**: Trade-off analysis for different configurations

## File Structure

```
replacement_block_pipeline/
├── block_data/
│   ├── block_00/
│   │   ├── train/ (*.pkl files)
│   │   └── val/ (*.pkl files)
│   ├── block_01/
│   └── ...
├── trained_models/
│   ├── block_00/
│   │   ├── full/
│   │   ├── diagonal/
│   │   └── affine/
│   └── ...
├── evaluation_results/
│   ├── predictions/
│   └── analysis/
└── adaptive_weighting/
    ├── checkpoints/
    └── logs/
```

## Dependencies

- PyTorch Lightning
- OpenFold codebase
- USalign (for TM score calculation)
- Standard ML libraries (numpy, pandas, matplotlib, seaborn)

## Research Applications

This pipeline enables research into:
- **Adaptive computation**: Learning when to use expensive vs cheap operations
- **Model compression**: Finding minimal architectures that maintain performance
- **Block importance**: Understanding which parts of Evoformer are most critical
- **Efficiency optimization**: Reducing computational cost while preserving accuracy

The results can inform future model architectures and training strategies for large-scale protein folding models.
