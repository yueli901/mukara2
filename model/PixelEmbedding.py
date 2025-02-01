import tensorflow as tf
import json
import os
import numpy as np
from config import MODEL, PATH, DATA  # Import hyperparameter settings and cache path
from dataloader import load_grid_features, load_grid_cells, get_sensor_coordinates, get_polygon
from shapely.geometry import shape

class PixelEmbedding(tf.keras.layers.Layer):
    def __init__(self):
        super(PixelEmbedding, self).__init__()
        
        self.grid_cells = load_grid_cells()
        self.grid_features = load_grid_features()
        
        # MLP for feature embedding
        self.feature_mlp = tf.keras.layers.Dense(MODEL['embedding_dim'], activation='relu')
        
        # MLP for absolute 2D positional encoding
        self.position_mlp = tf.keras.layers.Dense(MODEL['embedding_dim'], activation='relu')

    def call(self, sensor_ids):
        """
        Compute final pixel embeddings for a batch of sensors by combining feature, isochrone, and positional encodings.
        Load cache if already computed.
        
        Args:
            sensor_ids (tf.Tensor): (B,) Batch of sensor identifiers.
        
        Returns:
            tf.Tensor: O pixel embeddings (B, N, D),
            tf.Tensor: D pixel embeddings (B, N, D)
        """
        batch_o_embeddings, batch_d_embeddings = [], []
        
        for sensor_id in sensor_ids:
            sensor_id = sensor_id.numpy().decode('utf-8')
            sensor_cache_dir = os.path.join(PATH['cache'], sensor_id)
            
            if not os.path.exists(sensor_cache_dir):
                os.makedirs(sensor_cache_dir)
            
            index_path = os.path.join(sensor_cache_dir, "index.json")
            feature_cache = os.path.join(sensor_cache_dir, "features.npy")
            position_cache = os.path.join(sensor_cache_dir, "position.npy")
            o_isochrone_cache = os.path.join(sensor_cache_dir, "isochrone_o.npy")
            d_isochrone_cache = os.path.join(sensor_cache_dir, "isochrone_d.npy")
            
            if os.path.exists(index_path):
                with open(index_path, 'r') as f:
                    cache_data = json.load(f)
                o_indices = cache_data['o_indices']
                d_indices = cache_data['d_indices']
                
                raw_features = np.load(feature_cache)
                raw_absolute_position_encoding = np.load(position_cache)
                raw_isochrone_encoding_o = np.load(o_isochrone_cache)
                raw_isochrone_encoding_d = np.load(d_isochrone_cache)
            else:
                o_polygon = get_polygon(sensor_id, 'o')
                d_polygon = get_polygon(sensor_id, 'd')
                
                o_indices, d_indices, union_indices = self.compute_pixel_indices(o_polygon, d_polygon)
                raw_features = tf.gather_nd(self.grid_features, indices=union_indices).numpy()
                
                sensor_coords = get_sensor_coordinates(sensor_id)
                raw_absolute_position_encoding = self.compute_raw_absolute_position_encoding(sensor_coords, union_indices)
                
                raw_isochrone_encoding_o = self.compute_raw_isochrone_encoding(sensor_id, o_indices, 'o')
                raw_isochrone_encoding_d = self.compute_raw_isochrone_encoding(sensor_id, d_indices, 'd')
                
                np.save(feature_cache, raw_features)
                np.save(position_cache, raw_absolute_position_encoding)
                np.save(o_isochrone_cache, raw_isochrone_encoding_o)
                np.save(d_isochrone_cache, raw_isochrone_encoding_d)
                
                cache_data = {'o_indices': o_indices, 'd_indices': d_indices}
                with open(index_path, 'w') as f:
                    json.dump(cache_data, f)
            
            feature_embedding = self.feature_mlp(tf.convert_to_tensor(raw_features, dtype=tf.float32))
            absolute_position_embedding = self.position_mlp(tf.convert_to_tensor(raw_absolute_position_encoding, dtype=tf.float32))
            
            o_positions = [union_indices.index(i) for i in o_indices]
            d_positions = [union_indices.index(i) for i in d_indices]
            
            o_feature_embedding = tf.gather(feature_embedding, o_positions)
            d_feature_embedding = tf.gather(feature_embedding, d_positions)
            
            o_absolute_position_embedding = tf.gather(absolute_position_embedding, o_positions)
            d_absolute_position_embedding = tf.gather(absolute_position_embedding, d_positions)
            
            o_isochrone_embedding = tf.convert_to_tensor(raw_isochrone_encoding_o, dtype=tf.float32)
            d_isochrone_embedding = tf.convert_to_tensor(raw_isochrone_encoding_d, dtype=tf.float32)
            
            o_pixel_embeddings = o_feature_embedding + o_isochrone_embedding + o_absolute_position_embedding
            d_pixel_embeddings = d_feature_embedding + d_isochrone_embedding + d_absolute_position_embedding
            
            batch_o_embeddings.append(o_pixel_embeddings)
            batch_d_embeddings.append(d_pixel_embeddings)
        
        return tf.stack(batch_o_embeddings), tf.stack(batch_d_embeddings)
