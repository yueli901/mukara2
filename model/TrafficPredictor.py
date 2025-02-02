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
            cluster_embeddings (tf.Tensor): (K, D) cluster embeddings for a **single sensor**.
        
        Returns:
            tf.Tensor: (D,) pooled representation.
        """
        if self.pooling_method == 'mean':
            return tf.reduce_mean(cluster_embeddings, axis=0)  # (D,)
        elif self.pooling_method == 'sum':
            return tf.reduce_sum(cluster_embeddings, axis=0)  # (D,)
        elif self.pooling_method == 'attention':
            attention_scores = tf.nn.softmax(self.attention_weights(cluster_embeddings), axis=0)  # (K, 1)
            return tf.reduce_sum(attention_scores * cluster_embeddings, axis=0)  # (D,)
        else:
            raise ValueError("Invalid pooling method")
    
    def call(self, o_cluster_embeddings, d_cluster_embeddings):
        """
        Predict traffic volume from final O and D cluster embeddings for a single sensor.

        Args:
            o_cluster_embeddings (tf.Tensor): (K, D) cluster embeddings for the origin.
            d_cluster_embeddings (tf.Tensor): (K, D) cluster embeddings for the destination.

        Returns:
            tf.Tensor: Scalar predicted traffic volume.
        """
        assert o_cluster_embeddings.shape == (MODEL['num_clusters'], MODEL['embedding_dim']), "Shape mismatch!"

        # Apply pooling separately to O and D cluster embeddings
        o_pooled = self.pooling(o_cluster_embeddings)  # (D,)
        d_pooled = self.pooling(d_cluster_embeddings)  # (D,)

        # Concatenate O and D representations
        combined_representation = tf.concat([o_pooled, d_pooled], axis=-1)  # (2D,)

        # Predict traffic volume
        traffic_volume = self.output_layer(tf.expand_dims(combined_representation, axis=0))  # (1,)
        
        return tf.squeeze(traffic_volume)  # Scalar output