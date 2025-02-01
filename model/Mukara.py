import tensorflow as tf
from config import MODEL  # Import hyperparameter settings

from PixelEmbedding import PixelEmbedding
from Clustering import LearnableClustering
from Attention import SelfAttentionLayer, CrossAttentionLayer
from TrafficPredictor import TrafficPredictor

class TrafficVolumeModel(tf.keras.Model):
    def __init__(self):
        super(TrafficVolumeModel, self).__init__()
        
        # Define model components
        self.pixel_embedding = PixelEmbedding()
        self.cluster_module = LearnableClustering()
        self.self_attention_layers = [SelfAttentionLayer() for _ in range(MODEL['num_attention_layers'])]
        self.cross_attention_layers = [CrossAttentionLayer() for _ in range(MODEL['num_attention_layers'])]
        self.traffic_predictor = TrafficPredictor()
        
    def call(self, sensor_ids):
        """
        Forward pass of the Traffic Volume Prediction Model.
        
        Args:
            sensor_ids (tf.Tensor): (B,) Batch of sensor identifiers.
        
        Returns:
            tuple: Predicted traffic volume (B, 1),
                   Pixel embeddings (B, N, D),
                   Cluster embeddings (B, K, D),
                   Cluster assignments (B, N, K).
        """
        # Compute pixel embeddings (B, N, D)
        o_pixel_embeddings, d_pixel_embeddings = self.pixel_embedding(sensor_ids)
        
        # Compute cluster embeddings dynamically for O and D (B, K, D) and (B, N, K)
        o_cluster_embeddings, o_cluster_assignments = self.cluster_module(o_pixel_embeddings)
        d_cluster_embeddings, d_cluster_assignments = self.cluster_module(d_pixel_embeddings)
        
        # Apply interleaved attention: self → cross → self → cross
        for self_layer, cross_layer in zip(self.self_attention_layers, self.cross_attention_layers):
            o_cluster_embeddings = self_layer(o_cluster_embeddings)  # Self-attention for O
            d_cluster_embeddings = self_layer(d_cluster_embeddings)  # Self-attention for D
            o_cluster_embeddings = cross_layer(o_cluster_embeddings, d_cluster_embeddings)  # Cross-attention
            d_cluster_embeddings = cross_layer(d_cluster_embeddings, o_cluster_embeddings)  # Cross-attention
        
        # Predict traffic volume using pooled O and D cluster embeddings (B, 1)
        traffic_volume = self.traffic_predictor(o_cluster_embeddings, d_cluster_embeddings)
        
        return traffic_volume, (o_pixel_embeddings, d_pixel_embeddings), (o_cluster_embeddings, d_cluster_embeddings), (o_cluster_assignments, d_cluster_assignments)
