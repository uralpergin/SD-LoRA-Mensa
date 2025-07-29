"""
Hyperparameter Search for Mensa LoRA Training using Optuna with BOHB
"""

import argparse
import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances
import joblib
import torch
import gc
import sys
import os

# Import the train function from lora_train
sys.path.append('.')
from lora_train import train


def objective(trial):
    """Optuna objective function for hyperparameter optimization"""
    
    # Sample hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
    lora_r = trial.suggest_categorical('lora_r', [2, 4, 8, 16, 32])
    lora_alpha = trial.suggest_categorical('lora_alpha', [8, 16, 32, 48, 64, 96])
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
            resolution=512,
            trial=trial
        )
        
        return best_loss
        
    except optuna.TrialPruned:
        print(f"[PRUNED] Trial {trial.number} was pruned.")
        raise  # Re-raise pruning exceptions as-is
    except Exception as e:
        print(f"[ERROR] Trial {trial.number} crashed: {e}")
        raise optuna.TrialPruned()
    finally:
        # delete everything that holds GPU memory
        for name in ("unet","vae","text_encoder","optimizer",
                    "dataloader","lr_scheduler","accelerator"):
            if name in locals():
                del locals()[name]
        torch.cuda.empty_cache()
        gc.collect()



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
    
    # Configure BOHB (Hyperband + TPE) components
    sampler = TPESampler(seed=42)
    pruner = HyperbandPruner(
        min_resource=1,
        max_resource=args.epochs,
        reduction_factor=3  # try with different values like 3 or 4
    )
    
    # Create study with persistent storage
    study = optuna.create_study(
        storage="sqlite:///optuna_mensa.db",
        study_name="mensa_lora_bohb",
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )
    
    
    print("[SEARCH] Starting hyperparameter optimization...")
    
    # Run optimization
    study.optimize(objective,
                n_trials=args.n_trials,
                n_jobs=1,
                catch=(Exception,),
                gc_after_trial=True
                )

    # Save study and generate reports
    os.makedirs("logs", exist_ok=True)
    joblib.dump(study, "logs/optuna_study.pkl")
    print("[LOG] Study saved to logs/optuna_study.pkl")

    # Print results
    print("-" * 60)
    print("[RESULTS] Hyperparameter search complete!")
    print(f"[RESULTS] Best trial: {study.best_trial.number}")
    print(f"[RESULTS] Best loss: {study.best_value:.4f}")
    print("[RESULTS] Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    print("=" * 60)
    
    # Generate interactive Plotly reports
    opt_history_fig = plot_optimization_history(study)
    opt_history_fig.write_html("logs/optimization_history.html")
    print("[LOG] Optimization history saved to logs/optimization_history.html")
    
    param_importance_fig = plot_param_importances(study)
    param_importance_fig.write_html("logs/param_importances.html")
    print("[LOG] Parameter importances saved to logs/param_importances.html") 

if __name__ == "__main__":
    main()
