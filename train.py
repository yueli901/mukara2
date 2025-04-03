import tensorflow as tf
import numpy as np
import os
import logging
import datetime
import random

from config import PATH, DATA, TRAINING
from model.mukara import Mukara  # full-graph version
from model.dataloader import load_gt
import model.utils as utils


class MukaraTrainer:
    def __init__(self):
        # Set random seeds
        np.random.seed(TRAINING['seed'])
        tf.random.set_seed(TRAINING['seed'])
        random.seed(TRAINING['seed'])

        if not TRAINING['use_gpu']:
            tf.config.set_visible_devices([], 'GPU')

        # Logging
        logging.basicConfig(
            filename=os.path.join(PATH["evaluate"], 'training_log.log'),
            level=logging.INFO,
            format='%(asctime)s:%(levelname)s:%(message)s',
            filemode='w'
        )

        # Initialize model and optimizer
        print("Initializing model...")
        self.model = Mukara()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=TRAINING['lr'])
        print("Model initialized successfully.")

        # Load GT and split
        self.edge_to_gt, self.scaler = load_gt()
        self.all_edge_ids = list(self.edge_to_gt.keys())
        self.train_ids, self.val_ids = utils.train_test_sampler(self.all_edge_ids, TRAINING['train_prop'])

        # Build masks
        eid_to_idx = {eid: idx for idx, eid in enumerate(self.model.edge_ids)}
        self.train_mask = tf.convert_to_tensor([eid_to_idx[eid] for eid in self.train_ids], dtype=tf.int32)
        self.val_mask = tf.convert_to_tensor([eid_to_idx[eid] for eid in self.val_ids], dtype=tf.int32)

    def compute_loss(self, gt, pred, loss_function):
        return getattr(utils, loss_function)(gt, pred, self.scaler)

    def train_model(self):
        for epoch in range(TRAINING['epoch']):
            logging.info(f"Epoch {epoch} started.")

            with tf.GradientTape() as tape:
                predictions = self.model(self.model.g.edata['feat'])  # Full graph prediction, shape [num_edges, 1]

                train_gt = tf.convert_to_tensor([self.edge_to_gt[str(eid)] for eid in self.train_ids], dtype=tf.float32)
                train_pred = tf.gather(predictions, self.train_mask)

                loss = self.compute_loss(train_gt, train_pred, TRAINING['loss_function'])

            grads = tape.gradient(loss, self.model.trainable_variables)
            grads = [tf.clip_by_value(g, -TRAINING['clip_gradient'], TRAINING['clip_gradient']) for g in grads]
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            logging.info(f"Epoch {epoch}: Train Loss: {loss.numpy():.6f}")

            if epoch % TRAINING['eval_interval'] == 0 or epoch == TRAINING['epoch'] - 1:
                val_loss = self.evaluate_model()
                logging.info(f"Epoch {epoch}: Validation Loss: {val_loss:.6f}")

        self.save_model(epoch)

    def evaluate_model(self):
        predictions = self.model()
        val_gt = tf.convert_to_tensor([self.edge_to_gt[str(eid)] for eid in self.val_ids], dtype=tf.float32)
        val_pred = tf.gather(predictions, self.val_mask)
        val_loss = self.compute_loss(val_gt, val_pred, TRAINING['loss_function'])
        return val_loss.numpy()

    def save_model(self, epoch):
        formatted_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        model_filename = os.path.join(PATH["param"], f"epoch{epoch}_{formatted_time}.weights.h5")
        self.model.save_weights(model_filename)
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
