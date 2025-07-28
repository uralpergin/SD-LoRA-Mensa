"""
Hyperparameter Search for Mensa LoRA Training using Optuna with BOHB
"""

import argparse
import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
import sys
import os

# Import the train function from lora_train
sys.path.append('.')
from lora_train import train


def objective(trial):
    """Optuna objective function for hyperparameter optimization"""
    
    # Sample hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
    lora_r = trial.suggest_categorical('lora_r', [2, 4, 8, 16, 32, 48, 64])
    lora_alpha = trial.suggest_categorical('lora_alpha', [8, 16, 32, 48, 64, 96, 128])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.3)
    lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.3)
    
    # Create experiment name under hyperparameter search folder
    experiment_name = f"hyperparameter_search_new_data/trial_{trial.number}"
    
    try:
        # Call train function with sampled hyperparameters
        best_loss = train(
            dataset_csv=args.dataset_csv,
            experiment_name=experiment_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=learning_rate,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            lora_dropout=lora_dropout,
            duplicate=1,  # Use minimal duplication for faster training
            save_steps=10,  # Save less frequently during search
            concept_token="<mensafood>",
            pretrained_model="CompVis/stable-diffusion-v1-4",
            resolution=512
        )
        
        return best_loss
        
    except Exception as e:
        print(f"[ERROR] Trial {trial.number} failed: {e}")
        raise optuna.TrialPruned()


def main():
    global args
    parser = argparse.ArgumentParser(description='Hyperparameter Search for Mensa LoRA Training')
    parser.add_argument('--dataset_csv', required=True, help='Path to dataset CSV file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=6, help='Training batch size')
    parser.add_argument('--n_trials', type=int, default=30, help='Number of trials')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("         OPTUNA HYPERPARAMETER SEARCH")
    print("=" * 60)
    print(f"[SEARCH] Dataset: {args.dataset_csv}")
    print(f"[SEARCH] Epochs per trial: {args.epochs}")
    print(f"[SEARCH] Batch size: {args.batch_size}")
    print(f"[SEARCH] Number of trials: {args.n_trials}")
    print("-" * 60)
    
    # Configure Optuna study with BOHB (Hyperband + TPE)
    sampler = TPESampler(seed=42)
    pruner = HyperbandPruner(
        min_resource=1,
        max_resource=50,
        reduction_factor=3
    )
    
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        pruner=pruner,
        study_name='mensa_lora_hyperopt'
    )
    
    print("[SEARCH] Starting hyperparameter optimization...")
    
    # Run optimization
    study.optimize(objective, n_trials=args.n_trials)
    
    # Print results
    print("-" * 60)
    print("[RESULTS] Hyperparameter search complete!")
    print(f"[RESULTS] Best trial: {study.best_trial.number}")
    print(f"[RESULTS] Best loss: {study.best_value:.4f}")
    print("[RESULTS] Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    main()
