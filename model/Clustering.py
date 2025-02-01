import tensorflow as tf
from config import MODEL

class LearnableClustering(tf.keras.layers.Layer):
    def __init__(self):
        super(LearnableClustering, self).__init__()
        
        self.num_clusters = MODEL['num_clusters']
        self.embedding_dim = MODEL['embedding_dim']
        
        # MLP to project pixel embeddings into cluster space
        self.cluster_projection = tf.keras.layers.Dense(self.embedding_dim, activation='relu')
        
        # Learnable MLP for cluster assignment
        self.assignment_mlp = tf.keras.layers.Dense(self.num_clusters, activation='softmax')
    
    def compute_cluster_assignments(self, pixel_embeddings):
        """
        Compute soft cluster assignments for a **single sensor** using a learnable MLP.

        Args:
            pixel_embeddings (tf.Tensor): (N, D) pixel embeddings for one sensor.

        Returns:
            tf.Tensor: (N, K) cluster assignment probabilities.
        """
        projected_pixels = self.cluster_projection(pixel_embeddings)  # (N, D)
        cluster_assignments = self.assignment_mlp(projected_pixels)  # (N, K)
        
        return cluster_assignments  # (N, K)
    
    def compute_cluster_embeddings(self, pixel_embeddings, cluster_assignments):
        """
        Compute cluster embeddings for a **single sensor** as a weighted sum of pixel embeddings.

        Args:
            pixel_embeddings (tf.Tensor): (N, D) pixel embeddings for one sensor.
            cluster_assignments (tf.Tensor): (N, K) soft cluster assignment probabilities.

        Returns:
            tf.Tensor: (K, D) updated cluster embeddings.
        """
        weighted_sum = tf.matmul(cluster_assignments, pixel_embeddings, transpose_a=True)  # (K, D)
        sum_weights = tf.reduce_sum(cluster_assignments, axis=0, keepdims=True)  # (1, K)
        return weighted_sum / (sum_weights + 1e-6)  # (K, D) - Avoid division by zero
    
    def call(self, batch_pixel_embeddings):
        """
        Process **multiple sensors** by looping through them.

        Args:
            batch_pixel_embeddings (list of tf.Tensor): ([B], N, D)

        Returns:
            tuple: 
                - list of ([B], K, D) cluster embeddings, one per sensor.
                - list of ([B], N, K) cluster assignments, one per sensor.
        """
        batch_cluster_embeddings = []
        batch_cluster_assignments = []
        
        for pixel_embeddings in batch_pixel_embeddings:  # Process each sensor independently
            cluster_assignments = self.compute_cluster_assignments(pixel_embeddings)  # (N, K)
            cluster_embeddings = self.compute_cluster_embeddings(pixel_embeddings, cluster_assignments)  # (K, D)

            batch_cluster_embeddings.append(cluster_embeddings)
            batch_cluster_assignments.append(cluster_assignments)

        return batch_cluster_embeddings, batch_cluster_assignments