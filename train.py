import tensorflow as tf
import numpy as np
import os
import logging
import datetime
import random

from config import PATH, DATA, TRAINING, MODEL
import importlib
from model import Mukara
from model.dataloader import load_gt
import model.utils as utils

# Set random seeds for reproducibility
np.random.seed(TRAINING['seed'])
tf.random.set_seed(TRAINING['seed'])
random.seed(TRAINING['seed'])

# Disable GPUs if required
tf.config.set_visible_devices([], 'GPU')

# Configure logging
logging.basicConfig(
    filename=os.path.join(PATH["evaluate"], 'training_log.log'),
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s',
    filemode='w'
)

def train_model(mukara, id_to_gt, scaler, train_ids, test_ids):
    """
    Training loop for the Mukara model with batch learning and multiple loss functions.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=TRAINING['lr'])
    
    train_dataset = tf.data.Dataset.from_tensor_slices(train_ids)
    train_dataset = train_dataset.shuffle(len(train_ids)).batch(TRAINING['batch_size'])
    
    for epoch in range(TRAINING['epoch']):
        for batch_ids in train_dataset:
            with tf.GradientTape() as tape:
                batch_pred, pixel_embeddings, cluster_embeddings, cluster_assignments = mukara(batch_ids) 
                batch_gt = tf.convert_to_tensor([id_to_gt[batch_id.numpy()] for batch_id in batch_ids], dtype=tf.float32)
                
                # Compute total loss with multi-loss function
                loss, _ = compute_loss(
                    batch_gt, batch_pred, pixel_embeddings, cluster_embeddings, cluster_assignments, scaler, TRAINING['loss_function']
                )
            
            # Compute and apply gradients
            grads = tape.gradient(loss, mukara.trainable_variables)
            grads = [tf.clip_by_value(grad, -TRAINING['clip_gradient'], TRAINING['clip_gradient']) for grad in grads]
            optimizer.apply_gradients(zip(grads, mukara.trainable_variables))
            
        train_loss = evaluate_model(mukara, scaler, train_ids)
        test_loss = evaluate_model(mukara, scaler, test_ids)
        
        logging.info(f"Epoch {epoch}, Train Loss: {format_loss(train_loss)}")
        logging.info(f"Epoch {epoch}, Valid Loss: {format_loss(test_loss)}")

def compute_loss(gt, pred, pixel_embeddings, cluster_embeddings, cluster_assignments, scaler, traffic_loss_function):
    """
    Computes the total loss with multiple loss components.
    """
    traffic_loss = getattr(utils, traffic_loss_function)(gt, pred, scaler)

    o_pixel_embeddings, d_pixel_embeddings = pixel_embeddings
    o_cluster_assignments, d_cluster_assignments = cluster_assignments
    o_cluster_embeddings, d_cluster_embeddings = cluster_embeddings
    
    compactness_loss = (
        utils.compute_compactness_loss(o_pixel_embeddings, o_cluster_embeddings, o_cluster_assignments) +
        utils.compute_compactness_loss(d_pixel_embeddings, d_cluster_embeddings, d_cluster_assignments)
    ) / 2
    
    separation_loss = (
        utils.compute_separation_loss(o_cluster_embeddings) +
        utils.compute_separation_loss(d_cluster_embeddings)
    ) / 2
    
    balance_loss = (
        utils.compute_balance_loss(o_cluster_assignments) +
        utils.compute_balance_loss(d_cluster_assignments)
    ) / 2
    
    total_loss = (
        TRAINING['lambda_traffic'] * traffic_loss +
        TRAINING['lambda_compact'] * compactness_loss +
        TRAINING['lambda_separate'] * separation_loss +
        TRAINING['lambda_balance'] * balance_loss
    )
    
    return total_loss, traffic_loss

def evaluate_model(mukara, scaler, ids):
    """
    Evaluates the model on a subset of IDs.
    """
    sampled_ids = random.sample(ids, TRAINING['eval_samples'])
    loss = {metric: 0.0 for metric in TRAINING['eval_metrics']}
    pred, cluster_assignments, cluster_similarity, cluster_usage = mukara(sampled_ids)
    gt = tf.convert_to_tensor([id_to_gt[_id] for _id in sampled_ids], dtype=tf.float32)
    
    for metric in TRAINING['eval_metrics']:
        _, loss[metric] = compute_loss(gt, pred, cluster_assignments, cluster_similarity, cluster_usage, scaler, metric)
    return loss

def format_loss(loss_dict):
    """
    Formats the loss dictionary into a readable string.
    """
    return ", ".join([f"{key}: {value:.2f}" for key, value in loss_dict.items()])

if __name__ == '__main__':
    # Timestamp for model name
    formatted_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    model_filename = os.path.join(PATH["param"], f"{formatted_time}.h5")
    
    # Initialize model
    mukara = Mukara() 
    
    # Load data
    id_to_gt, scaler = load_gt()
    train_ids, test_ids = utils.train_test_sampler(list(id_to_gt.keys()), TRAINING['train_prop'])
    
    # Train model
    train_model(mukara, id_to_gt, scaler, train_ids, test_ids)
    
    # Save model parameters
    mukara.save_weights(model_filename)
    print(mukara.summary())
    print("Model saved successfully.")
    
    # Clear cache if enabled
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
