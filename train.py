import tensorflow as tf
# To disable all GPUs
tf.config.set_visible_devices([], 'GPU')

import numpy as np
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
import os
import logging
import datetime
import random

from config import PATH, DATA, TRAINING, MODEL
import importlib
imported_module = importlib.import_module(f"model.{MODEL['model']}")
Model = getattr(imported_module, "Model")
from model.dataloader import get_gt, get_static_features, process_sensors
import model.utils as utils

np.random.seed(TRAINING['seed'])
tf.random.set_seed(TRAINING['seed'])
random.seed(TRAINING['seed'])

# Set logging
# current_time = datetime.datetime.now()
# formatted_time = current_time.strftime('%Y%m%d-%H%M%S')
logging.basicConfig(filename=os.path.join(PATH["evaluate"],'training_log.log'), level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s', filemode='w')

# Define a function for the training loop
def train_model(mukara, id_to_gt, scaler, train_ids, test_ids):
    optimizer = tf.keras.optimizers.Adam(learning_rate=TRAINING['lr'])
    for epoch in range(TRAINING['epoch']):
        random.shuffle(train_ids)
        for i in range(0, len(train_ids), TRAINING['batch_size']):
            batch_ids = train_ids[i : i+TRAINING['batch_size']]
            with tf.GradientTape() as tape:
                batch_pred = mukara(batch_ids) 
                batch_gt = tf.convert_to_tensor([id_to_gt[batch_id] for batch_id in batch_ids], dtype=tf.float32)
                loss = compute_loss(
                    batch_gt, batch_pred, scaler, TRAINING['loss_function']
                )
            
            grads = tape.gradient(loss, mukara.trainable_variables)
            grads = [tf.clip_by_value(grad, -TRAINING['clip_gradient'], TRAINING['clip_gradient']) for grad in grads]
            optimizer.apply_gradients(zip(grads, mukara.trainable_variables))

            # Evaluate model and get the loss as a dictionary
            if i % 10 == 0:
                train_loss = evaluate_model(mukara, scaler, train_ids)
                test_loss = evaluate_model(mukara, scaler, test_ids)

                train_log_message = f"Epoch {epoch}, Step {i}, Train Loss: " + ", ".join(
                    [f"{key}: {value:.2f}" for key, value in train_loss.items()]
                )
                logging.info(train_log_message)

                test_log_message = f"Epoch {epoch}, Step {i}, Valid Loss: " + ", ".join(
                    [f"{key}: {value:.2f}" for key, value in test_loss.items()]
                )
                logging.info(test_log_message)

            if epoch == TRAINING['epoch']-1 and i == TRAINING['step']:
                return


def compute_loss(gt, pred, scaler, loss_function):
    """
    label shape (b,) tf, one batch only
    pred shape (b,) tf, one batch only
    train_test_index shape (batch_size,) numpy boolean
    """
    loss = getattr(utils, loss_function)(gt, pred, scaler)
    return loss


def evaluate_model(mukara, scaler, ids):
    ids = random.sample(ids, TRAINING['eval_samples'])
    loss = {metric: 0.0 for metric in TRAINING['eval_metrics']}
    pred = mukara(ids)
    gt = tf.convert_to_tensor([id_to_gt[_id] for _id in ids], dtype=tf.float32)
    for metric in TRAINING['eval_metrics']:
        loss[metric] = compute_loss(gt, pred, scaler, metric)
    return loss

# Main
if __name__ == '__main__':
    np.random.seed(TRAINING['seed'])
    tf.random.set_seed(TRAINING['seed'])
    random.seed(TRAINING['seed'])

    # add timestamp for model name
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime('%Y%m%d-%H%M%S')
    filename = f"{formatted_time}.h5"
    name = os.path.join(PATH["param"], filename)

    # initialize model
    mukara = Model() 

    # Data setup
    id_to_gt, scaler = get_gt()
    grid_static_features = get_static_features()
    train_ids, test_ids = utils.train_test_sampler(list(id_to_gt.keys()), TRAINING['train_prop'])
    process_sensors(id_to_gt, grid_static_features)

    # Training
    train_model(mukara, id_to_gt, scaler, train_ids, test_ids)

    # Save model parameters
    mukara.save_weights(name)
    print(mukara.summary())
    print("Model saved successfully.")

    if DATA['clear_cache']:
        for filename in os.listdir(PATH['roi_cache']):
            os.remove(os.path.join(PATH['roi_cache'], filename))

    # log file
    for handler in logging.root.handlers[:]:
        handler.close()
        logging.root.removeHandler(handler)

    # copy the current model config to the end of log file
    with open('config.py', 'r') as config_file:
        config_content = config_file.read()

    with open(os.path.join(PATH["evaluate"], f"training_log.log"), 'a') as log_file:
        log_file.write('\n\n# Contents of config.py\n')
        log_file.write(config_content)