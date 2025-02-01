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
        Compute soft cluster assignments using a learnable MLP mapping.
        
        Args:
            projected_pixels (tf.Tensor): (B, N, D) projected pixel embeddings.
        
        Returns:
            tf.Tensor: (B, N, K) cluster assignment probabilities.
        """
        projected_pixels = self.cluster_projection(pixel_embeddings)  # Project to cluster space (B, N, D)
        return self.assignment_mlp(projected_pixels)  # (B, N, K)
    
    def compute_cluster_embeddings(self, pixel_embeddings, cluster_assignments):
        """
        Compute cluster embeddings as a weighted sum of pixel embeddings.
        
        Args:
            pixel_embeddings (tf.Tensor): (B, N, D) pixel embeddings.
            cluster_assignments (tf.Tensor): (B, N, K) soft cluster assignment weights.
        
        Returns:
            tf.Tensor: (B, K, D) updated cluster embeddings.
        """
        weighted_sum = tf.matmul(cluster_assignments, pixel_embeddings, transpose_a=True)  # (B, K, D)
        sum_weights = tf.reduce_sum(cluster_assignments, axis=1)  # (B, K)
        sum_weights = tf.expand_dims(sum_weights, axis=-1)  # (B, K, 1)
        return weighted_sum / (sum_weights + 1e-6)  # Avoid division by zero
    
    def call(self, pixel_embeddings):
        """
        Learn dynamic cluster embeddings from pixel embeddings.
        
        Args:
            pixel_embeddings (tf.Tensor): (B, N, D) pixel embeddings.
        
        Returns:
            tuple: (B, K, D) cluster embeddings,
                   (B, N, K) cluster assignments.
        """
        cluster_assignments = self.compute_cluster_assignments(pixel_embeddings)  # Soft assignments (B, N, K)
        
        # Compute cluster embeddings as weighted sum of assigned pixels
        cluster_embeddings = self.compute_cluster_embeddings(pixel_embeddings, cluster_assignments)
        
        return cluster_embeddings, cluster_assignments
