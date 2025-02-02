import tensorflow as tf
from config import MODEL

class SelfAttentionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(SelfAttentionLayer, self).__init__()
                
        # Multi-head self-attention layer
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=MODEL['num_attention_heads'], key_dim=MODEL['embedding_dim'])

        # Layer normalization and residual connection
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Feed Forward Network (FFN)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(MODEL['embedding_dim'] * 4, activation='relu'),
            tf.keras.layers.Dense(MODEL['embedding_dim'])
        ])
        
    def call(self, cluster_embeddings):
        """
        Apply self-attention within O/D clusters.

        Args:
            cluster_embeddings (tf.Tensor): (K, D) cluster embeddings.

        Returns:
            tf.Tensor: (K, D) updated cluster embeddings.
        """
        # Expand dimension to match api input dim
        if len(cluster_embeddings.shape) != 3:  
            cluster_embeddings = tf.expand_dims(cluster_embeddings, axis=0) # (1, K, D)

        query, key, value = cluster_embeddings, cluster_embeddings, cluster_embeddings
        
        # Multi-head self-attention
        attention_output = self.attention(query=query, key=key, value=value)  # (1, K, D)  

        # Residual connection and LayerNorm
        cluster_embeddings = self.norm1(cluster_embeddings + attention_output)

        # Feed Forward Network
        ffn_output = self.ffn(cluster_embeddings)

        # Second Residual connection and LayerNorm
        cluster_embeddings = self.norm2(cluster_embeddings + ffn_output)

        assert cluster_embeddings.shape == (1, MODEL['num_clusters'], MODEL['embedding_dim']), "Shape mismatch!"

        return cluster_embeddings


class CrossAttentionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(CrossAttentionLayer, self).__init__()
                
        # Multi-head cross-attention layer
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=MODEL['num_attention_heads'], key_dim=MODEL['embedding_dim'])

        # Layer normalization and residual connection
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Feed Forward Network (FFN)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(MODEL['embedding_dim'] * 4, activation='relu'),
            tf.keras.layers.Dense(MODEL['embedding_dim'])
        ])
        
    def call(self, A_cluster_embeddings, B_cluster_embeddings):
        """
        Apply cross-attention between O and D clusters.

        Args:
            A_cluster_embeddings (tf.Tensor): (K, D) cluster embeddings for origin/destination.
            B_cluster_embeddings (tf.Tensor): (K, D) cluster embeddings for destination/origin.

        Returns:
            tf.Tensor: (K, D) updated cluster A embeddings.
        """

        query, key, value = A_cluster_embeddings, B_cluster_embeddings, B_cluster_embeddings # must be (1, K, D)

        # Compute attention
        attention_output = self.attention(query=query, key=key, value=value)  # (1, K, D)

        # Residual connection and LayerNorm
        A_cluster_embeddings = self.norm1(A_cluster_embeddings + attention_output)

        # Feed Forward Network
        ffn_output = self.ffn(A_cluster_embeddings)

        # Second Residual connection and LayerNorm
        A_cluster_embeddings = self.norm2(A_cluster_embeddings + ffn_output)

        assert A_cluster_embeddings.shape == (1, MODEL['num_clusters'], MODEL['embedding_dim']), "Shape mismatch!"

        return A_cluster_embeddings