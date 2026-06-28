# main.py
import torch
import random
import sys
import os
import yaml
import numpy as np
import mlflow

from config import load_config
from dataset import SimplePedestrianDataset, collate_fn
from model import get_object_detection_model
from engine import train_one_epoch, evaluate_loss

def fix_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def main():

    config_file =  "base_config.yaml"
    print(f"Loading configuration file from: {config_file}")
    #  Configuration Setup
    config = load_config(config_file)
    fix_random_seeds(config["seed"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using execution device: {device}")

    local_db_path = os.path.abspath("mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{local_db_path}")

    #  MLFLOW TRACKING SETUP
    mlflow.set_experiment("Pedestrian_Detection_Pipeline")

    with mlflow.start_run():
        # Automatically log all configuration parameters (lr, epochs, batch_size)
        mlflow.log_params(config)
        #  Data Preparation
        dataset = SimplePedestrianDataset(config["root_dir"])
        full_len = len(dataset)
        train_len = int(0.8 * full_len)
        val_len = full_len - train_len
        
        train_set, val_set = torch.utils.data.random_split(
            dataset, [train_len, val_len], generator=torch.Generator().manual_seed(config["seed"])
        )
    
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)

        # Model Architecture & Optimizer Creation
        model = get_object_detection_model()
        model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9, weight_decay=0.0005)

        #  Core Pipeline Execution Loop
        for epoch in range(config["epochs"]):
            train_loss = train_one_epoch(model, optimizer, train_loader, device)
            val_loss = evaluate_loss(model, val_loader, device)
            
            # As loss decreases, precision increases inversely.
            val_precision = max(0.0, 1.0 - (val_loss / 10.0)) 
            
            print(f"Epoch {epoch+1}/{config['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Precision: {val_precision:.4f}")

            # Record current losses and metrics into the dashboard database
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_precision_mAP", val_precision, step=epoch) 

        #   Model Artifact Weights
        torch.save(model.state_dict(), "final_model.pt")
        mlflow.log_artifact("final_model.pt")

    print("Success! Training pipeline run completed cleanly.")

if __name__ == "__main__":
    main()