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


def MSE_Z(label_z, pred_z, scaler):
    mse_z = tf.reduce_mean(tf.square(label_z - pred_z))
    print(f"GT_Z: {label_z}, Pred_Z: {pred_z}, MSE_Z: {mse_z}")
    return mse_z

def MAE(label_z, pred_z, scaler):
    pred = scaler.inverse_transform(pred_z)
    label = scaler.inverse_transform(label_z)
    mae = tf.reduce_mean(tf.abs(label - pred))
    print(f"GT: {label}, Pred: {pred}, MAE: {mae}")
    return mae

def MSE(label_z, pred_z, scaler):
    pred = scaler.inverse_transform(pred_z)
    label = scaler.inverse_transform(label_z)
    mse = tf.reduce_mean(tf.square(label - pred))
    print(f"GT: {label}, Pred: {pred}, MSE: {mse}")
    return mse

def MGEH(label_z, pred_z, scaler):
    pred = scaler.inverse_transform(pred_z)
    label = scaler.inverse_transform(label_z)
    gehs = tf.sqrt(2 * tf.square(pred - label) / (tf.abs(pred) + label + 1e-8))
    mgeh = tf.reduce_mean(gehs)
    print(f"GT: {label}, Pred: {pred}, GEH: {mgeh}")
    return mgeh

def train_test_sampler(edge_ids, true_probability):
    """
    Splits edge IDs into training and test sets based on probability.
    """
    random.shuffle(edge_ids)
    train_ids = [edge_id for edge_id in edge_ids if random.random() < true_probability]
    test_ids = [edge_id for edge_id in edge_ids if edge_id not in train_ids]
    return train_ids, test_ids

def compute_feature_compact_loss(o_pixel_embeddings, d_pixel_embeddings, 
                                 o_cluster_embeddings, d_cluster_embeddings, 
                                 o_cluster_assignments, d_cluster_assignments):
    """
    Ensures that pixel feature embeddings within the same cluster are similar.

    Args:
        o_pixel_embeddings (tf.Tensor): (N_o, D) pixel embeddings for origin.
        d_pixel_embeddings (tf.Tensor): (N_d, D) pixel embeddings for destination.
        o_cluster_embeddings (tf.Tensor): (K, D) cluster embeddings for origin.
        d_cluster_embeddings (tf.Tensor): (K, D) cluster embeddings for destination.
        o_cluster_assignments (tf.Tensor): (N_o, K) cluster assignment probabilities for origin.
        d_cluster_assignments (tf.Tensor): (N_d, K) cluster assignment probabilities for destination.

    Returns:
        tf.Tensor: Scalar feature compactness loss.
    """
    def compactness_loss(pixel_embeddings, cluster_embeddings, cluster_assignments):
        """
        Computes the compactness loss for a given set of pixel and cluster embeddings.
        """
        expanded_pixel_embeddings = tf.expand_dims(pixel_embeddings, 1)  # (N, 1, D)
        expanded_cluster_embeddings = tf.expand_dims(cluster_embeddings, 0)  # (1, K, D)
        squared_differences = tf.square(expanded_pixel_embeddings - expanded_cluster_embeddings)  # (N, K, D)
        weighted_distances = tf.reduce_mean(tf.expand_dims(cluster_assignments, axis=-1) * squared_differences)  # (1,)
        return weighted_distances

    o_loss = compactness_loss(o_pixel_embeddings, o_cluster_embeddings, o_cluster_assignments)
    d_loss = compactness_loss(d_pixel_embeddings, d_cluster_embeddings, d_cluster_assignments)

    return (o_loss + d_loss) / 2  # Average over origin and destination


def compute_spatial_compact_loss(o_pixel_coordinates, d_pixel_coordinates, 
                                 o_cluster_assignments, d_cluster_assignments):
    """
    Ensures that pixels within the same cluster are spatially compact.

    Args:
        o_indices (list): List of pixel indices in the origin region.
        d_indices (list): List of pixel indices in the destination region.
        o_cluster_embeddings (tf.Tensor): (K, D) cluster embeddings for origin.
        d_cluster_embeddings (tf.Tensor): (K, D) cluster embeddings for destination.
        o_cluster_assignments (tf.Tensor): (N_o, K) cluster assignment probabilities for origin.
        d_cluster_assignments (tf.Tensor): (N_d, K) cluster assignment probabilities for destination.
        grid_cells (gpd.GeoDataFrame): GeoDataFrame containing the pixel grid.

    Returns:
        tf.Tensor: Scalar spatial compactness loss.
    """
    def spatial_loss(cluster_assignments, pixel_coords):
        """
        Computes the spatial compactness loss using weighted geographic centroids.
        """
        # Compute cluster centroids
        weighted_sum = tf.matmul(cluster_assignments, pixel_coords, transpose_a=True)  # (K, 2)
        sum_weights = tf.reduce_sum(cluster_assignments, axis=0)  # (K,)
        sum_weights = tf.expand_dims(sum_weights, axis=-1)  # (K, 1)
        cluster_centroids = weighted_sum / (sum_weights + 1e-6)  # (K, 2)

        # Compute squared distances from each pixel to its assigned cluster centroid
        expanded_pixel_coords = tf.expand_dims(pixel_coords, 1)  # (N, 1, 2)
        expanded_cluster_centroids = tf.expand_dims(cluster_centroids, 0)  # (1, K, 2)
        squared_distances = tf.square(expanded_pixel_coords - expanded_cluster_centroids)  # (N, K, 2)
        weighted_distances = tf.reduce_mean(tf.expand_dims(cluster_assignments, axis=-1) * squared_distances)  # (1,)

        return weighted_distances

    o_loss = spatial_loss(o_cluster_assignments, o_pixel_coordinates)
    d_loss = spatial_loss(d_cluster_assignments, d_pixel_coordinates)

    return (o_loss + d_loss) / 2  # Average over origin and destination