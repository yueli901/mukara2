import tensorflow as tf
from config import MODEL  # Import hyperparameter settings

from model.PixelEmbedding import PixelEmbedding
from model.Clustering import LearnableClustering
from model.Attention import SelfAttentionLayer, CrossAttentionLayer
from model.TrafficPredictor import TrafficPredictor

class Mukara(tf.keras.Model):
    def __init__(self):
        super(Mukara, self).__init__()
        
        # Define model components
        self.pixel_embedding = PixelEmbedding()
        self.cluster_module = LearnableClustering()
        self.self_attention_layers = [SelfAttentionLayer() for _ in range(MODEL['num_attention_layers'])]
        self.cross_attention_layers = [CrossAttentionLayer() for _ in range(MODEL['num_attention_layers'])]
        self.traffic_predictor = TrafficPredictor()
        
    def call(self, sensor_id):
        """
        Forward pass of the Traffic Volume Prediction Model.
        
        Args:
            sensor_id (str): Single sensor identifier.
        
        Returns:
            tuple: Predicted traffic volume (1,),
                   Pixel embeddings (N, D)
                   Cluster embeddings (K, D),
                   Cluster assignments (N, K).
        """
        # Compute pixel embeddings (N, D)
        o_pixel_embeddings, d_pixel_embeddings = self.pixel_embedding(sensor_id)
        
        # Compute cluster embeddings dynamically for O and D (K, D) and (N, K)
        o_cluster_embeddings, o_cluster_assignments = self.cluster_module(o_pixel_embeddings)
        d_cluster_embeddings, d_cluster_assignments = self.cluster_module(d_pixel_embeddings)
        
        # Apply interleaved attention: self → cross → self → cross
        for self_layer, cross_layer in zip(self.self_attention_layers, self.cross_attention_layers):
            o_cluster_embeddings = self_layer(o_cluster_embeddings)  # Self-attention for O
            d_cluster_embeddings = self_layer(d_cluster_embeddings)  # Self-attention for D
            o_cluster_embeddings = cross_layer(o_cluster_embeddings, d_cluster_embeddings)  # Cross-attention
            d_cluster_embeddings = cross_layer(d_cluster_embeddings, o_cluster_embeddings)  # Cross-attention
        
        o_cluster_embeddings = tf.squeeze(o_cluster_embeddings) # from (1, K, D) to (K, D)
        d_cluster_embeddings = tf.squeeze(d_cluster_embeddings)

        # Predict traffic volume using pooled O and D cluster embeddings (1,)
        traffic_volume = self.traffic_predictor(o_cluster_embeddings, d_cluster_embeddings)
        
        return traffic_volume, (o_pixel_embeddings, d_pixel_embeddings), (o_cluster_embeddings, d_cluster_embeddings), (o_cluster_assignments, d_cluster_assignments)
