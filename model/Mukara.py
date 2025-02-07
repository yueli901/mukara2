import tensorflow as tf
from config import MODEL  # Import hyperparameter settings

from model.PixelEmbedding import PixelEmbedding
from model.Clustering import LearnableClustering
from model.Attention import CustomAttentionLayer
from model.TrafficPredictor import TrafficPredictor
from model.dataloader import get_sensor_coordinates

class Mukara(tf.keras.Model):
    def __init__(self):
        super(Mukara, self).__init__()
        
        # Define model components
        self.pixel_embedding = PixelEmbedding()
        self.o_cluster_module = LearnableClustering()
        self.d_cluster_module = LearnableClustering()
        self.o_self_attention_layers = [CustomAttentionLayer() for _ in range(MODEL['num_attention_layers'])]
        self.d_self_attention_layers = [CustomAttentionLayer() for _ in range(MODEL['num_attention_layers'])]
        self.od_cross_attention_layers = [CustomAttentionLayer() for _ in range(MODEL['num_attention_layers'])]
        self.do_cross_attention_layers = [CustomAttentionLayer() for _ in range(MODEL['num_attention_layers'])]
        self.traffic_predictor = TrafficPredictor()
        
    def call(self, sensor_id):
        """
        Forward pass of the Traffic Volume Prediction Model.
        
        Args:
            sensor_id (str): Single sensor identifier.

        Intermediates:
            o_pixel_embeddings / d_pixel_embeddings: (N, D)
            o_indices / d_indices: list of grid indices
            o_isochrone_classes / d_isochrone_classes: (N,) with values from 1-12
            o_cluster_embeddings / d_cluster_embeddings: (K, D)
            o_cluster_assignments / d_cluster_assignments: (N, K)
            sensor_coords: (2,)

        
        Returns:
            tuple: Predicted traffic volume (1,),
                   Pixel embeddings (N, D)
                   Cluster embeddings (K, D),
                   Cluster assignments (N, K).
        """
        # Compute pixel embeddings (N, D)
        o_pixel_embeddings, d_pixel_embeddings, o_indices, d_indices, o_isochrone_classes, d_isochrone_classes = self.pixel_embedding(sensor_id)
        
        # Compute cluster embeddings (K, D) and cluster_assignments (N, K) dynamically for O and D
        o_cluster_embeddings, o_cluster_assignments = self.o_cluster_module(o_pixel_embeddings)
        d_cluster_embeddings, d_cluster_assignments = self.d_cluster_module(d_pixel_embeddings)
        
        # Apply interleaved attention: self → cross → self → cross
        sensor_coords = get_sensor_coordinates(sensor_id)
        for o_self_layer, d_self_layer, od_cross_layer, do_cross_layer in zip(self.o_self_attention_layers, self.d_self_attention_layers, self.od_cross_attention_layers, self.od_cross_attention_layers):
            o_cluster_embeddings = o_self_layer(sensor_coords, o_cluster_embeddings, o_cluster_embeddings, o_cluster_assignments, o_cluster_assignments, o_indices, o_indices, o_isochrone_classes, o_isochrone_classes)  # Self-attention for O
            d_cluster_embeddings = d_self_layer(sensor_coords, d_cluster_embeddings, d_cluster_embeddings, d_cluster_assignments, d_cluster_assignments, d_indices, d_indices, d_isochrone_classes, d_isochrone_classes)  # Self-attention for D
            o_cluster_embeddings_new = od_cross_layer(sensor_coords, o_cluster_embeddings, d_cluster_embeddings, o_cluster_assignments, d_cluster_assignments, o_indices, d_indices, o_isochrone_classes, d_isochrone_classes)  # Cross-attention OD
            d_cluster_embeddings_new = do_cross_layer(sensor_coords, d_cluster_embeddings, o_cluster_embeddings, d_cluster_assignments, o_cluster_assignments, d_indices, o_indices, d_isochrone_classes, o_isochrone_classes)  # Cross-attention DO
            o_cluster_embeddings = o_cluster_embeddings_new
            d_cluster_embeddings = d_cluster_embeddings_new

        # Predict traffic volume using pooled O and D cluster embeddings (1,)
        traffic_volume = self.traffic_predictor(o_cluster_embeddings, d_cluster_embeddings)
        
        return traffic_volume, (o_indices, d_indices, o_pixel_embeddings, d_pixel_embeddings, o_cluster_embeddings, d_cluster_embeddings, o_cluster_assignments, d_cluster_assignments)
