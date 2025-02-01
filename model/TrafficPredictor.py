import tensorflow as tf
from config import MODEL

class TrafficPredictor(tf.keras.layers.Layer):
    def __init__(self):
        super(TrafficPredictor, self).__init__()
        
        self.pooling_method = MODEL['pooling_method']  # Options: 'mean', 'sum', 'attention'
        
        # Attention pooling (if selected)
        if self.pooling_method == 'attention':
            self.attention_weights = tf.keras.layers.Dense(1, activation='softmax')
        
        # Final prediction layer
        self.output_layer = tf.keras.layers.Dense(1, activation=None)  # Traffic volume output
        
    def pooling(self, cluster_embeddings):
        """
        Apply pooling method to aggregate cluster embeddings into a single vector.
        
        Args:
            cluster_embeddings (tf.Tensor): (B, K, D) cluster embeddings for a batch of sensors.
        
        Returns:
            tf.Tensor: (B, D) pooled representation.
        """
        if self.pooling_method == 'mean':
            return tf.reduce_mean(cluster_embeddings, axis=1)  # (B, D)
        elif self.pooling_method == 'sum':
            return tf.reduce_sum(cluster_embeddings, axis=1)  # (B, D)
        elif self.pooling_method == 'attention':
            attention_scores = tf.nn.softmax(self.attention_weights(cluster_embeddings), axis=1)  # (B, K, 1)
            return tf.reduce_sum(attention_scores * cluster_embeddings, axis=1)  # (B, D)
        else:
            raise ValueError("Invalid pooling method")
    
    def call(self, cluster_embeddings):
        """
        Predict traffic volume from final cluster embeddings.
        
        Args:
            cluster_embeddings (tf.Tensor): (B, K, embedding_dim) cluster embeddings for a batch of sensors.
        
        Returns:
            tf.Tensor: (B, 1) predicted traffic volume.
        """
        pooled_output = self.pooling(cluster_embeddings)  # (B, D)
        traffic_volume = self.output_layer(pooled_output)  # (B, 1)
        return traffic_volume
