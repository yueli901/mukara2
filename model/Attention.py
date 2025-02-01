import tensorflow as tf
from config import MODEL

class SelfAttentionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(SelfAttentionLayer, self).__init__()
                
        # Multi-head self-attention layer
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=MODEL['num_attention_heads'], key_dim=MODEL['embedding_dim'])
        
        # Layer normalization and residual connection
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(MODEL['embedding_dim'] * 4, activation='relu'),
            tf.keras.layers.Dense(MODEL['embedding_dim'])
        ])
        
    def call(self, cluster_embeddings):
        """
        Apply self-attention within O/D clusters.
        
        Args:
            cluster_embeddings (tf.Tensor): (B, K, D) cluster embeddings.
        
        Returns:
            tf.Tensor: (B, K, D) updated cluster embeddings.
        """
        attention_output = self.attention(
            query=cluster_embeddings, key=cluster_embeddings, value=cluster_embeddings
        )
        
        # Add & Normalize (Residual Connection)
        cluster_embeddings = self.norm(cluster_embeddings + attention_output)
        
        # Feed Forward Network
        ffn_output = self.ffn(cluster_embeddings)
        cluster_embeddings = self.norm(cluster_embeddings + ffn_output)
        
        return cluster_embeddings


class CrossAttentionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(CrossAttentionLayer, self).__init__()
                
        # Multi-head cross-attention layer
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=MODEL['num_attention_heads'], key_dim=MODEL['embedding_dim'])
        
        # Layer normalization and residual connection
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(MODEL['embedding_dim'] * 4, activation='relu'),
            tf.keras.layers.Dense(MODEL['embedding_dim'])
        ])
        
    def call(self, A_cluster_embeddings, B_cluster_embeddings):
        """
        Apply cross-attention between O and D clusters.
        
        Args:
            A_cluster_embeddings (tf.Tensor): (B, K, D) cluster embeddings for origin/destination.
            B_cluster_embeddings (tf.Tensor): (B, K, D) cluster embeddings for destination/origin.
        
        Returns:
            tf.Tensor: (B, K, D) updated cluster embeddings.
        """
        attention_output = self.attention(
            query=A_cluster_embeddings, key=B_cluster_embeddings, value=B_cluster_embeddings
        )
        
        # Add & Normalize (Residual Connection)
        A_cluster_embeddings = self.norm(A_cluster_embeddings + attention_output)
        
        # Feed Forward Network
        ffn_output = self.ffn(A_cluster_embeddings)
        A_cluster_embeddings = self.norm(A_cluster_embeddings + ffn_output)
        
        return A_cluster_embeddings
