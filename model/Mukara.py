import tensorflow as tf
from config import MODEL  # Import hyperparameter settings

from model.PixelEmbedding import PixelEmbedding
from model.Clustering import FixedPatchClustering
from model.Attention import CustomAttentionLayer
from model.TrafficPredictor import TrafficPredictor
from model.dataloader import get_sensor_coordinates, load_grid_cells

class Mukara(tf.keras.Model):
    def __init__(self):
        super(Mukara, self).__init__()

        self.grid_cells = load_grid_cells()

        # Define model components
        self.pixel_embedding = PixelEmbedding('pixel_embeddings', self.grid_cells)
        self.o_cluster_module = FixedPatchClustering('o_clustering', self.grid_cells)
        self.d_cluster_module = FixedPatchClustering('d_clustering', self.grid_cells)
        self.o_self_attention_layers = [CustomAttentionLayer(f'o_self_attention_layer_{i}', self.grid_cells) for i in range(MODEL['num_attention_layers'])]
        self.d_self_attention_layers = [CustomAttentionLayer(f'd_self_attention_layer_{i}', self.grid_cells) for i in range(MODEL['num_attention_layers'])]
        self.od_cross_attention_layers = [CustomAttentionLayer(f'od_cross_attention_layer_{i}', self.grid_cells) for i in range(MODEL['num_attention_layers'])]
        self.do_cross_attention_layers = [CustomAttentionLayer(f'do_cross_attention_layer_{i}', self.grid_cells) for i in range(MODEL['num_attention_layers'])]
        self.traffic_predictor = TrafficPredictor('traffic_predictor')
        
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
        sensor_coords = get_sensor_coordinates(sensor_id)

        # Compute pixel embeddings (N, D)
        o_pixel_embeddings, d_pixel_embeddings, o_indices, d_indices, o_isochrone_classes, d_isochrone_classes = self.pixel_embedding(sensor_id)

        # Compute cluster embeddings (K, D) and cluster_assignments (N, K) dynamically for O and D
        o_cluster_embeddings, o_cluster_centroids, o_cluster_isochrone_encodings = self.o_cluster_module(o_pixel_embeddings, o_indices, o_isochrone_classes)
        d_cluster_embeddings, d_cluster_centroids, d_cluster_isochrone_encodings = self.d_cluster_module(d_pixel_embeddings, d_indices, d_isochrone_classes)

        # Apply interleaved attention: self → cross → self → cross
        for o_self_layer, d_self_layer, od_cross_layer, do_cross_layer in zip(self.o_self_attention_layers, self.d_self_attention_layers, self.od_cross_attention_layers, self.do_cross_attention_layers):
            o_cluster_embeddings = o_self_layer(sensor_coords, o_cluster_embeddings, o_cluster_embeddings, o_cluster_centroids, o_cluster_centroids, o_cluster_isochrone_encodings, o_cluster_isochrone_encodings)  # Self-attention for O
            d_cluster_embeddings = d_self_layer(sensor_coords, d_cluster_embeddings, d_cluster_embeddings, d_cluster_centroids, d_cluster_centroids, d_cluster_isochrone_encodings, d_cluster_isochrone_encodings)  # Self-attention for D
            o_cluster_embeddings_new = od_cross_layer(sensor_coords, o_cluster_embeddings, d_cluster_embeddings, o_cluster_centroids, d_cluster_centroids, o_cluster_isochrone_encodings, d_cluster_isochrone_encodings)  # Cross-attention OD
            d_cluster_embeddings_new = do_cross_layer(sensor_coords, d_cluster_embeddings, o_cluster_embeddings, d_cluster_centroids, o_cluster_centroids, d_cluster_isochrone_encodings, o_cluster_isochrone_encodings)  # Cross-attention DO
            o_cluster_embeddings = o_cluster_embeddings_new
            d_cluster_embeddings = d_cluster_embeddings_new

        # Predict traffic volume using pooled O and D cluster embeddings (1,)
        traffic_volume = self.traffic_predictor(o_cluster_embeddings, d_cluster_embeddings)
        print(f"{o_cluster_embeddings.shape[0]} o clusters, {d_cluster_embeddings.shape[0]} d clusters.")
        
        return traffic_volume