import tensorflow as tf
import json
import os
import numpy as np
from config import MODEL, PATH, DATA  # Import hyperparameter settings and cache path
from model.dataloader import load_grid_features, load_grid_cells, get_sensor_coordinates, get_polygon
from shapely.geometry import shape

class PixelEmbedding(tf.keras.layers.Layer):
    def __init__(self):
        super(PixelEmbedding, self).__init__()
        
        self.grid_cells = load_grid_cells()
        self.grid_features = load_grid_features()
        self.H, self.W, self.C = self.grid_features.shape  # Grid dimensions
        
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
            list of tf.Tensor: O pixel embeddings ([B], N, D)
            list of tf.Tensor: D pixel embeddings ([B], N, D)
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
                print(f"Computing cache for sensor {sensor_id}.")
                o_polygon = get_polygon(sensor_id, 'o')
                d_polygon = get_polygon(sensor_id, 'd')
                
                o_indices, d_indices, union_indices = self.compute_pixel_indices(o_polygon, d_polygon)
                raw_features = tf.gather(tf.reshape(self.grid_features, [-1, self.C]), union_indices).numpy()  # (N, C)
                
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
        
        return batch_o_embeddings, batch_d_embeddings

    def compute_pixel_indices(self, O_polygon, D_polygon):
        """
        Compute the pixel indices that fall within the O and D polygons, and their union.
        
        Args:
            grid_cells (gpd.GeoDataFrame): Grid cells containing pixel locations.
            O_polygon (shapely.geometry.Polygon): Polygon defining the origin (O) region.
            D_polygon (shapely.geometry.Polygon): Polygon defining the destination (D) region.
        
        Returns:
            tuple: (o_indices, d_indices, union_indices)
        """

        # Identify grid cells intersecting with O and D polygons
        o_mask = self.grid_cells.intersects(O_polygon)
        d_mask = self.grid_cells.intersects(D_polygon)
        
        o_indices = np.where(o_mask)[0].tolist()  # 1D indices
        d_indices = np.where(d_mask)[0].tolist()  # 1D indices

        # Compute union of O and D indices
        union_indices = list(set(o_indices) | set(d_indices))
        
        return o_indices, d_indices, union_indices
        
    def compute_raw_isochrone_encoding(self, sensor_id, indices, region_type):
        """
        Compute raw isochrone encoding for each pixel index based on the shortest travel time isochrone.
        
        Args:
            sensor_id (str): Unique identifier of the sensor.
            indices (list): List of grid indices representing selected pixels.
            region_type (str): 'o' for origin, 'd' for destination.
        
        Returns:
            np.ndarray: Isochrone positional encoding (N, D).
        """
        # Define mapping for isochrone intervals (300s = 1, 600s = 2, ..., 3600s = 12)
        isochrone_levels_map = {t: i + 1 for i, t in enumerate(DATA['isochrone_intervals'])}
        
        # Load grid cells
        selected_grid_cells = self.grid_cells.iloc[indices]
        remaining_mask = np.ones(len(indices), dtype=bool)  # Mask for unassigned indices
        
        # Initialize isochrone labels with the highest value
        isochrone_labels = np.full(len(indices), 12, dtype=np.int32)
        
        # Iterate through all isochrone levels (except the largest), starting from the smallest
        for t in DATA['isochrone_intervals'][:-1]:
            isochrone_file = os.path.join(PATH["isochrones"], f"{sensor_id}", f"{sensor_id}_{t}_{'destination' if region_type == 'o' else 'start'}.json")
            with open(isochrone_file, 'r') as f:
                data = json.load(f)
                geojson_polygon = {
                    "type": "Polygon",
                    "coordinates": data['features'][0]['geometry']['coordinates']
                }
                isochrone_polygon = shape(geojson_polygon)
                
            # Find grid cells intersecting with this isochrone
            mask = selected_grid_cells[remaining_mask].intersects(isochrone_polygon)
            full_mask = np.zeros(len(indices), dtype=bool)
            full_mask[remaining_mask] = mask
            
            # Assign the isochrone level
            isochrone_labels[full_mask] = isochrone_levels_map[t]
            
            # Update remaining mask to exclude already assigned indices
            remaining_mask[full_mask] = False
        
        # Convert isochrone labels to final encoding (Transformer-style)
        div_term = np.exp(np.arange(0, MODEL['embedding_dim'], 2, dtype=np.float32) * -np.log(10000.0) / MODEL['embedding_dim'])
        sin_encoding = np.sin(isochrone_labels[:, None] * div_term)
        cos_encoding = np.cos(isochrone_labels[:, None] * div_term)

        encoding = np.empty((isochrone_labels.shape[0], MODEL['embedding_dim']), dtype=np.float32)
        encoding[:, 0::2] = sin_encoding  # Even indices: sin
        encoding[:, 1::2] = cos_encoding  # Odd indices: cos
        
        return encoding

    def compute_raw_absolute_position_encoding(self, sensor_coords, union_indices):
        """
        Compute absolute 2D positional encoding based on sensor coordinates in EPSG:27700.
        
        Args:
            sensor_coords (tuple): (x, y) coordinates of the sensor in EPSG:27700.
            union_indices (list): List of grid indices representing selected pixels.

        Returns:
            np.ndarray: Absolute positional encoding before MLP transformation (N, 3).
        """
        # Load grid cells to get pixel coordinates
        grid_cells = load_grid_cells()
        selected_grid_cells = grid_cells.iloc[union_indices]
        pixel_coords = np.array([(cell.centroid.x, cell.centroid.y) for cell in selected_grid_cells.geometry])
        
        sensor_x, sensor_y = sensor_coords

        # Compute polar coordinates relative to the sensor
        rho = np.sqrt((pixel_coords[:, 0] - sensor_x) ** 2 + (pixel_coords[:, 1] - sensor_y) ** 2)
        theta = np.arctan2(pixel_coords[:, 1] - sensor_y, pixel_coords[:, 0] - sensor_x)

        # Normalize distance with a scale factor
        scale_factor = np.max(rho) # relative to the sensor's case, not global scale_factor
        normalized_rho = rho / scale_factor

        encoding = np.stack([
            normalized_rho,   # Direct normalized distance
            np.sin(theta),    # Angular sin encoding
            np.cos(theta)     # Angular cos encoding
        ], axis=-1)

        return encoding
