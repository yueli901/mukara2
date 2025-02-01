import tensorflow as tf
from config import TRAINING
import random

# Set random seeds for reproducibility
tf.random.set_seed(TRAINING['seed'])
random.seed(TRAINING['seed'])

class Scaler:
    def __init__(self, data):
        """
        Standard scaler that normalizes data while ignoring NaN values.
        """
        valid_data = tf.where(tf.math.is_nan(data), tf.zeros_like(data), data)
        count = tf.reduce_sum(tf.cast(~tf.math.is_nan(data), tf.float32))
        
        self.mean = tf.reduce_sum(valid_data) / count
        self.std = tf.sqrt(tf.reduce_sum(tf.square(valid_data - self.mean)) / count)
        
        tf.print("Mean:", self.mean, "Standard Deviation:", self.std)

    def transform(self, data):
        return tf.where(tf.math.is_nan(data), data, (data - self.mean) / (self.std + 1e-8))
    
    def inverse_transform(self, data):
        return tf.where(tf.math.is_nan(data), data, data * self.std + self.mean)


def MSE_z(label_z, pred_z, scaler):
    return tf.reduce_mean(tf.square(label_z - pred_z))

def MAE(label_z, pred_z, scaler):
    pred = scaler.inverse_transform(pred_z)
    label = scaler.inverse_transform(label_z)
    return tf.reduce_mean(tf.abs(label - pred))

def MSE(label_z, pred_z, scaler):
    pred = scaler.inverse_transform(pred_z)
    label = scaler.inverse_transform(label_z)
    return tf.reduce_mean(tf.square(label - pred))

def MGEH(label_z, pred_z, scaler):
    pred = scaler.inverse_transform(pred_z)
    label = scaler.inverse_transform(label_z)
    gehs = tf.sqrt(2 * tf.square(pred - label) / (tf.abs(pred) + label + 1e-8))
    return tf.reduce_mean(gehs)


def train_test_sampler(sensor_ids, true_probability):
    """
    Splits sensor IDs into training and test sets based on probability.
    """
    random.shuffle(sensor_ids)
    train_ids = [sensor_id for sensor_id in sensor_ids if random.random() < true_probability]
    test_ids = [sensor_id for sensor_id in sensor_ids if sensor_id not in train_ids]
    return train_ids, test_ids


def compute_compactness_loss(pixel_embeddings, cluster_embeddings, cluster_assignments):
    """
    Encourages pixel embeddings within the same cluster to be similar.
    
    Args:
        pixel_embeddings (tf.Tensor): (B, N, D) pixel embeddings.
        cluster_embeddings (tf.Tensor): (B, K, D) cluster embeddings.
        cluster_assignments (tf.Tensor): (B, N, K) cluster assignment weights.
    
    Returns:
        tf.Tensor: (B,) Compactness loss per batch.
    """
    expanded_pixel_embeddings = tf.expand_dims(pixel_embeddings, 2)  # (B, N, 1, D)
    expanded_cluster_centroids = tf.expand_dims(cluster_embeddings, 1)  # (B, 1, K, D)
    squared_differences = tf.square(expanded_pixel_embeddings - expanded_cluster_centroids)  # (B, N, K, D)
    weighted_distances = tf.reduce_sum(cluster_assignments[..., None] * squared_differences, axis=(1, 2, 3))
    return tf.reduce_mean(weighted_distances, axis=0)  # Averaged per batch


def compute_separation_loss(cluster_embeddings):
    """
    Encourages cluster embeddings to be distinct.
    
    Args:
        cluster_embeddings (tf.Tensor): (B, K, D) cluster embeddings.
    
    Returns:
        tf.Tensor: (B,) Separation loss per batch.
    """
    normalized_embeddings = tf.nn.l2_normalize(cluster_embeddings, axis=-1)  # (B, K, D)
    cluster_similarity = tf.matmul(normalized_embeddings, normalized_embeddings, transpose_b=True)  # (B, K, K)
    identity_matrix = tf.eye(tf.shape(cluster_similarity)[1], batch_shape=[tf.shape(cluster_similarity)[0]])
    off_diagonal_similarity = cluster_similarity * (1 - identity_matrix)
    return tf.reduce_mean(off_diagonal_similarity, axis=[1, 2])  # Averaged per batch


def compute_balance_loss(cluster_assignments):
    """
    Ensures that clusters are used evenly across a batch.
    
    Args:
        cluster_assignments (tf.Tensor): (B, N, K) cluster assignment probabilities.
    
    Returns:
        tf.Tensor: (B,) Balance loss per batch.
    """
    cluster_usage = tf.reduce_sum(cluster_assignments, axis=1)  # (B, K)
    mean_usage = tf.reduce_mean(cluster_usage, axis=1, keepdims=True)  # (B, 1)
    return tf.reduce_mean(tf.square(cluster_usage - mean_usage), axis=1)  # Averaged per batch
