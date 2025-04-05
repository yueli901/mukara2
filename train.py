import torch
import numpy as np
import os
import logging
import datetime
import random

from config import PATH, DATA, TRAINING
from model.mukara import Mukara  # subgraph-based PyTorch+DGL model
from model.dataloader import load_gt
import model.utils as utils

class MukaraTrainer:
    def __init__(self):
        # Set random seeds
        np.random.seed(TRAINING['seed'])
        torch.manual_seed(TRAINING['seed'])
        random.seed(TRAINING['seed'])
        
        self.device = torch.device("cuda" if (TRAINING['use_gpu'] and torch.cuda.is_available()) else "cpu")
        print(self.device)

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
        self.model.z_shift = (-self.scaler.mean / (self.scaler.std + 1e-8)).item()
        self.all_edge_ids = list(self.edge_to_gt.keys())
        self.train_ids, self.val_ids = utils.train_test_sampler(self.all_edge_ids, TRAINING['train_prop'])

    def compute_loss(self, gt, pred, loss_function):
        return getattr(utils, loss_function)(gt, pred, self.scaler)

    def train_model(self):
        for epoch in range(TRAINING['epoch']):
            logging.info(f"Epoch {epoch} started.")
            self.model.train()
            random.shuffle(self.train_ids)

            total_loss = 0.0
            train_preds, train_gts = [], []

            for step, edge_id in enumerate(self.train_ids):
                gt = self.edge_to_gt[edge_id]
                pred = self.model(edge_id)
                if pred is None or pred.numel() == 0:
                    continue

                gt_tensor = torch.tensor([gt], dtype=torch.float32, device=self.device)
                pred = pred.to(self.device)

                loss = self.compute_loss(gt_tensor, pred, TRAINING['loss_function'])
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), TRAINING['clip_gradient'])
                self.optimizer.step()

                train_preds.append(pred.detach())
                train_gts.append(gt_tensor)
                total_loss += loss.item()

                mse_z = utils.MSE_Z(gt_tensor, pred, self.scaler).item()
                mae = utils.MAE(gt_tensor, pred, self.scaler).item()
                geh = utils.MGEH(gt_tensor, pred, self.scaler).item()

                real_gt = self.scaler.inverse_transform(gt_tensor).item()
                real_pred = self.scaler.inverse_transform(pred).item()
                logging.info(
                    f"Epoch {epoch} Step {step}: Train Edge {edge_id}, MSE_Z: {mse_z:.6f}, MAE: {mae:.2f}, GEH: {geh:.2f}, "
                    f"GT: {real_gt:.2f}, Pred: {real_pred:.2f}"
                )

                # Evaluation interval
                if (step + 1) % TRAINING['eval_interval'] == 0:
                    self.evaluate_random_sample(epoch, step)

            # Final evaluation at end of epoch
            self.evaluate_random_sample(epoch, 'end')

    def evaluate_random_sample(self, epoch, step):
        self.model.eval()
        with torch.no_grad():
            sample_train = random.sample(self.train_ids, min(TRAINING['eval_sample'], len(self.train_ids)))
            sample_val = random.sample(self.val_ids, min(TRAINING['eval_sample'], len(self.val_ids)))

            for split, sample_ids in [('Train', sample_train), ('Validation', sample_val)]:
                preds, gts = [], []
                for edge_id in sample_ids:
                    gt = self.edge_to_gt[edge_id]
                    pred = self.model(edge_id)
                    if pred is None or pred.numel() == 0:
                        continue
                    preds.append(pred)
                    gts.append(torch.tensor([gt], dtype=torch.float32, device=self.device))

                if preds:
                    preds = torch.cat(preds)
                    gts = torch.cat(gts)

                    mse_z = utils.MSE_Z(gts, preds, self.scaler).item()
                    mae = utils.MAE(gts, preds, self.scaler).item()
                    geh = utils.MGEH(gts, preds, self.scaler).item()

                    logging.info(
                        f"Epoch {epoch} Step {step} {split} Eval: MSE_Z: {mse_z:.6f}, MAE: {mae:.6f}, GEH: {geh:.6f}"
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