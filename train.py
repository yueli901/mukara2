import torch
import numpy as np
import os
import logging
import datetime
import random

from config import PATH, DATA, TRAINING
from model.mukara import Mukara  # full-graph version (PyTorch)
from model.dataloader import load_gt
import model.utils as utils

class MukaraTrainer:
    def __init__(self):
        # Set random seeds
        np.random.seed(TRAINING['seed'])
        torch.manual_seed(TRAINING['seed'])
        random.seed(TRAINING['seed'])

        self.device = torch.device("cuda" if (TRAINING['use_gpu'] and torch.cuda.is_available()) else "cpu")

        # Logging
        logging.basicConfig(
            filename=os.path.join(PATH["evaluate"], 'training_log.log'),
            level=logging.INFO,
            format='%(asctime)s:%(levelname)s:%(message)s',
            filemode='w'
        )

        # Initialize model and optimizer
        print("Initializing model...")
        self.model = Mukara().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=TRAINING['lr'])
        print("Model initialized successfully.")

        # Load GT and split
        self.edge_to_gt, self.scaler = load_gt()
        self.all_edge_ids = list(self.edge_to_gt.keys())
        self.train_ids, self.val_ids = utils.train_test_sampler(self.all_edge_ids, TRAINING['train_prop'])

        # Build mapping from edge_id to DGL internal index
        eid_to_idx = {str(eid): idx for idx, eid in enumerate(self.model.edge_ids)}
        self.train_idx = torch.tensor([eid_to_idx[eid] for eid in self.train_ids], dtype=torch.long, device=self.device)
        self.val_idx = torch.tensor([eid_to_idx[eid] for eid in self.val_ids], dtype=torch.long, device=self.device)

    def compute_loss(self, gt, pred, loss_function):
        return getattr(utils, loss_function)(gt, pred, self.scaler)
    
    def train_model(self):
        for epoch in range(TRAINING['epoch']):
            logging.info(f"Epoch {epoch} started.")

            self.model.train()
            self.optimizer.zero_grad()

            # Forward pass
            predictions = self.model(self.model.g.edata['feat'].to(self.device))  # [num_edges, 1]

            # Prepare ground truths and predicted values for training and evaluation
            train_gt = torch.tensor([self.edge_to_gt[str(eid)] for eid in self.train_ids], dtype=torch.float32, device=self.device)
            val_gt = torch.tensor([self.edge_to_gt[str(eid)] for eid in self.val_ids], dtype=torch.float32, device=self.device)

            train_pred = predictions[self.train_idx]
            val_pred = predictions[self.val_idx]

            # Compute training loss and update
            loss = self.compute_loss(train_gt, train_pred, TRAINING['loss_function'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), TRAINING['clip_gradient'])
            self.optimizer.step()

            # Additional training metrics
            train_mse_z = utils.MSE_Z(train_gt, train_pred, self.scaler).item()
            train_mae = utils.MAE(train_gt, train_pred, self.scaler).item()
            train_geh = utils.MGEH(train_gt, train_pred, self.scaler).item()

            logging.info(
                f"Epoch {epoch}: Train Loss ({TRAINING['loss_function']}): {loss.item():.6f}, "
                f"MSE_Z: {train_mse_z:.6f}, MAE: {train_mae:.6f}, GEH: {train_geh:.6f}"
            )

            # Evaluate
            val_loss = self.compute_loss(val_gt, val_pred, TRAINING['loss_function']).item()
            val_mse_z = utils.MSE_Z(val_gt, val_pred, self.scaler).item()
            val_mae = utils.MAE(val_gt, val_pred, self.scaler).item()
            val_geh = utils.MGEH(val_gt, val_pred, self.scaler).item()

            logging.info(
                f"Epoch {epoch}: Validation Loss ({TRAINING['loss_function']}): {val_loss:.6f}, "
                f"MSE_Z: {val_mse_z:.6f}, MAE: {val_mae:.6f}, GEH: {val_geh:.6f}"
            )


    def save_model(self, epoch):
        formatted_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        model_filename = os.path.join(PATH["param"], f"epoch{epoch}_{formatted_time}.pt")
        torch.save(self.model.state_dict(), model_filename)
        print("Model saved successfully.")

    def cleanup_and_log_config(self):
        if DATA['clear_cache']:
            for filename in os.listdir(PATH['cache']):
                os.remove(os.path.join(PATH['cache'], filename))

        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)

        with open('config.py', 'r') as config_file:
            config_content = config_file.read()

        with open(os.path.join(PATH["evaluate"], "training_log.log"), 'a') as log_file:
            log_file.write('\n\n# Contents of config.py\n')
            log_file.write(config_content)

if __name__ == '__main__':
    print("Initiating model...")
    trainer = MukaraTrainer()
    print("Training started...")
    trainer.train_model()
    trainer.save_model('final')
    trainer.cleanup_and_log_config()
