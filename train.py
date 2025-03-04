import tensorflow as tf
import numpy as np
import os
import logging
import datetime
import random

from config import PATH, DATA, TRAINING
from model.Mukara import Mukara
from model.dataloader import load_grid_cells, load_gt
import model.utils as utils


class MukaraTrainer:
    def __init__(self):
        """
        Initializes the MukaraTrainer with model, optimizer, and logging configuration.
        """
        # Set random seeds for reproducibility
        np.random.seed(TRAINING['seed'])
        tf.random.set_seed(TRAINING['seed'])
        random.seed(TRAINING['seed'])

        # Disable GPUs if required
        if TRAINING['use_gpu'] == False:
            tf.config.set_visible_devices([], 'GPU')

        # Configure logging
        logging.basicConfig(
            filename=os.path.join(PATH["evaluate"], 'training_log.log'),
            level=logging.INFO,
            format='%(asctime)s:%(levelname)s:%(message)s',
            filemode='w'
        )

        # Initialize model and optimizer
        self.model = Mukara()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=TRAINING['lr'])

        self.id_to_gt, self.scaler = load_gt()
        self.train_ids, self.test_ids = utils.train_test_sampler(
            list(self.id_to_gt.keys()), TRAINING['train_prop']
        )

    def compute_loss(self, gt, pred, traffic_loss_function):
        """
        Computes the total loss with multiple components.
        """
        traffic_loss = getattr(utils, traffic_loss_function)(gt, pred, self.scaler)
        
        return traffic_loss

    def train_model(self):
        """
        Training loop for the Mukara model.
        Evaluates the model every `TRAINING['eval_interval']` sensors trained.
        """
        for epoch in range(TRAINING['epoch']):
            random.shuffle(self.train_ids)

            for i, sensor_id in enumerate(self.train_ids, start=1):
                with tf.GradientTape() as tape:
                    pred = self.model(sensor_id)
                    gt = tf.convert_to_tensor(self.id_to_gt[sensor_id], dtype=tf.float32)

                    # Compute total loss
                    loss = self.compute_loss(gt, pred, TRAINING['loss_function'])

                # Compute and apply gradients
                grads = tape.gradient(loss, self.model.trainable_variables)
                grads = [tf.clip_by_value(grad, -TRAINING['clip_gradient'], TRAINING['clip_gradient']) for grad in grads]
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                print(f"Training complete for epoch {epoch}, sensor {i}/{len(self.train_ids)}.")

                # Evaluate model periodically
                if i % TRAINING['eval_interval'] == 0 or i == len(self.train_ids):
                    train_loss = self.evaluate_model(self.train_ids)
                    test_loss = self.evaluate_model(self.test_ids)

                    logging.info(f"Epoch {epoch}, Sensor {i}: Train Loss: {self.format_loss(train_loss)}")
                    logging.info(f"Epoch {epoch}, Sensor {i}: Valid Loss: {self.format_loss(test_loss)}")

            # Save model weights
            self.save_model()

    def evaluate_model(self, ids):
        """
        Evaluates the model on a subset of sensor IDs and logs all loss components.
        """
        sampled_ids = random.sample(ids, TRAINING['eval_samples'])
        loss = {}

        for metric in TRAINING['eval_metrics']:
            loss[metric] = 0.0

        for sensor_id in sampled_ids:
            pred = self.model(sensor_id)
            gt = tf.convert_to_tensor(self.id_to_gt[sensor_id], dtype=tf.float32)

            # Compute traffic loss using different evaluation metrics
            for metric in TRAINING['eval_metrics']:
                traffic_loss = self.compute_loss(gt, pred, metric)
                loss[metric] += traffic_loss.numpy()

        # Normalize by the number of evaluated samples
        num_samples = len(sampled_ids)
        for key in loss.keys():
            loss[key] /= num_samples

        return loss

    @staticmethod
    def format_loss(loss_dict):
        """
        Formats the loss dictionary into a readable string with .6f precision.
        """
        return ", ".join([f"{key}: {value:.6f}" for key, value in loss_dict.items()])

    def save_model(self):
        """
        Saves the model weights.
        """
        formatted_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        model_filename = os.path.join(PATH["param"], f"{formatted_time}.h5")
        self.model.save_weights(model_filename)
        print("Model saved successfully.")

    def cleanup_and_log_config(self):
        """
        Clears the cache if enabled and logs the configuration.
        """
        if DATA['clear_cache']:
            for filename in os.listdir(PATH['cache']):
                os.remove(os.path.join(PATH['cache'], filename))

        # Close logging handlers
        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)

        # Append model configuration to log file
        with open('config.py', 'r') as config_file:
            config_content = config_file.read()

        with open(os.path.join(PATH["evaluate"], "training_log.log"), 'a') as log_file:
            log_file.write('\n\n# Contents of config.py\n')
            log_file.write(config_content)


if __name__ == '__main__':
    trainer = MukaraTrainer()
    trainer.train_model()
    trainer.save_model()
    print(trainer.model.summary())
    trainer.cleanup_and_log_config()