import tensorflow as tf
import numpy as np
from config import MODEL
from model.dataloader import load_grid_cells, get_pixel_coordinates

class CustomAttentionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(CustomAttentionLayer, self).__init__()
        
        self.grid_cells = load_grid_cells()

        # Custom Multi-Head Attention ?

        # Layer normalization
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Feed Forward Network (FFN)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(MODEL['embedding_dim'] * 4, activation='relu'),
            tf.keras.layers.Dense(MODEL['embedding_dim'])
        ])

        # MLPs for learnable G and H
        self.G_MLP = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='relu')
        ])

        self.H_MLP = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='relu')
        ])
        
    def call(self, sensor_coords, query, key, q_assignments, k_assignments, q_indices, k_indices, q_isochrone_classes, k_isochrone_classes):
        """
        Custom Self-Attention and Cross-Attention Mechanism with Geometric Encoding.

        Args:
            sensor_coords (tf.Tensor): (2,) sensor x, y coordinates.
            query, key (tf.Tensor): (K, D) cluster embeddings.
            q_assignments, k_assignments (tf.Tensor): (N, K) cluster assignment probabilities.
            q_indices, k_indices (list): Indices of pixels in the grid.
            q_isochrone_classes, k_isochrone_classes (tf.Tensor): (N,) pixel isochrone class values (1-12).

        Returns:
            tf.Tensor: (K, D) updated cluster embeddings after attention.
        """

        # Compute soft-weighted centroid for clusters
        q_cluster_centroids = self.compute_cluster_centroids(q_assignments, q_indices)  # (K, 2)
        k_cluster_centroids = self.compute_cluster_centroids(k_assignments, k_indices)  # (K, 2)

        # Compute pairwise distance vectors and their norms
        r_kk_prime = q_cluster_centroids[:, None, :] - k_cluster_centroids[None, :, :]  # (K, K, 2)
        r_ks = q_cluster_centroids - sensor_coords  # (K, 2)
        r_k_prime_s = k_cluster_centroids - sensor_coords  # (K, 2)
        ### potential normalizing issue
        norm_r_kk_prime = tf.norm(r_kk_prime, axis=-1)  # (K, K)
        norm_r_ks = tf.norm(r_ks, axis=-1)  # (K,)
        norm_r_k_prime_s = tf.norm(r_k_prime_s, axis=-1)  # (K,)

        # Compute cluster-level isochrone embeddings
        d_k = self.compute_cluster_isochrone_encoding(q_assignments, q_isochrone_classes)  # (K, 12)
        d_k_prime = self.compute_cluster_isochrone_encoding(k_assignments, k_isochrone_classes)  # (K, 12)

        # Compute learnable geometric attention scaling factor
        R_kk_prime = self.geometric_attention_factor(d_k, d_k_prime, norm_r_kk_prime, norm_r_ks, norm_r_k_prime_s)

        # Compute Multi-Head Attention (scaled before softmax)
        attention_logits = self.compute_scaled_attention(query, key)  # (K, K)
        attention_logits = attention_logits * R_kk_prime  # Apply geometric scaling before softmax

        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(attention_logits, axis=-1)  # (K, K)

        # Compute new attention output
        attention_output = tf.matmul(attention_weights, key)  # (K, D)

        # Residual connection and LayerNorm
        query = self.norm1(query + attention_output)

        # Feed Forward Network
        ffn_output = self.ffn(query)

        # Second Residual connection and LayerNorm
        query = self.norm2(query + ffn_output)

        return query

    def compute_scaled_attention(self, query, key):
        """
        Compute scaled dot-product attention logits.

        Args:
            query, key (tf.Tensor): (K, D) cluster embeddings.

        Returns:
            tf.Tensor: (K, K) attention logits.
        """
        d_k = tf.cast(tf.shape(query)[-1], tf.float32)  # Scaling factor
        return tf.matmul(query, key, transpose_b=True) / tf.sqrt(d_k)  # (K, K)

    def compute_cluster_centroids(self, cluster_assignments, indices):
        """
        Compute soft-weighted cluster centroids in 2D space.

        Args:
            cluster_assignments (tf.Tensor): (N, K) cluster assignment probabilities.
            indices (list): Pixel grid indices.

        Returns:
            tf.Tensor: (K, 2) cluster centroids in x, y coordinates.
        """
        grid_coords = get_pixel_coordinates(self.grid_cells, indices)  # (N, 2)
        weighted_sum = tf.matmul(cluster_assignments, grid_coords, transpose_a=True)  # (K, 2)
        sum_weights = tf.reduce_sum(cluster_assignments, axis=0, keepdims=True)  # (1, K)
        return weighted_sum / (sum_weights + 1e-6)  # (K, 2)

    def compute_cluster_isochrone_encoding(self, cluster_assignments, pixel_isochrone_classes):
        """
        Compute soft-weighted cluster-level isochrone encoding.

        Args:
            cluster_assignments (tf.Tensor): (N, K) cluster assignment probabilities.
            pixel_isochrone_classes (tf.Tensor): (N,) pixel isochrone classes (1-12).

        Returns:
            tf.Tensor: (K, 12) normalized isochrone class distribution.
        """
        pixel_isochrone_classes = tf.one_hot(pixel_isochrone_classes - 1, depth=12)  # (N, 12)
        weighted_isochrone = tf.matmul(cluster_assignments, pixel_isochrone_classes, transpose_a=True)  # (K, 12)
        return weighted_isochrone / tf.reduce_sum(weighted_isochrone, axis=-1, keepdims=True)  # Normalize across isochrone classes

    def geometric_attention_factor(self, d_k, d_k_prime, norm_r_kk_prime, norm_r_ks, norm_r_k_prime_s):
        """
        Compute the geometric-aware scaling factor for attention using MLP.

        Args:
            d_k, d_k_prime (tf.Tensor): (K, 12) soft-clustered isochrone class encodings.
            norm_r_kk_prime, norm_r_ks, norm_r_k_prime_s (tf.Tensor): Norms of spatial distances.

        Returns:
            tf.Tensor: (K, K) geometric attention scaling factor.
        """
        G = self.G_MLP(tf.stack([d_k[:, None, :], d_k_prime[None, :, :]], axis=-1))  # (K, K, 1)
        H = self.H_MLP(tf.stack([norm_r_kk_prime, norm_r_ks[:, None], norm_r_k_prime_s[None, :]], axis=-1))  # (K, K, 1)
        return tf.squeeze(G * H, axis=-1)  # (K, K)
