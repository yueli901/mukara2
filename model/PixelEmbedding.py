import tensorflow as tf
import json
import os
import numpy as np
from config import MODEL, PATH, DATA  # Import hyperparameter settings and cache path
from model.dataloader import load_grid_features, load_grid_cells, get_polygon
from shapely.geometry import shape

class PixelEmbedding(tf.keras.layers.Layer):
    def __init__(self):
        super(PixelEmbedding, self).__init__()
        
        self.grid_cells = load_grid_cells()
        self.grid_features = load_grid_features()
        self.H, self.W, self.C = self.grid_features.shape  # Grid dimensions
        
        # MLP for feature embedding
        self.feature_mlp = tf.keras.layers.Dense(MODEL['embedding_dim'], activation='relu')
        
    def call(self, sensor_id):
        """
        Compute pixel embeddings for one sensor from pixel features.
        Compute pixel isochrone classes.
        Load cache for O/D pixel indices and pixel isochrone classes if already computed.
        
        Args:
            sensor_id (str): Sensor identifier.
        
        Returns:
            tuple:
                tf.Tensor: O pixel embeddings (N, D)
                tf.Tensor: D pixel embeddings (N, D)
        """
        sensor_cache_dir = os.path.join(PATH['cache'] + '_' + str(MODEL['embedding_dim']), sensor_id)
        
        if not os.path.exists(sensor_cache_dir):
            os.makedirs(sensor_cache_dir)
        
        indices_path = os.path.join(sensor_cache_dir, "indices.json")
        o_isochrone_cache = os.path.join(sensor_cache_dir, "isochrone_o.npy")
        d_isochrone_cache = os.path.join(sensor_cache_dir, "isochrone_d.npy")
        
        if os.path.exists(indices_path) and os.path.exists(o_isochrone_cache) and os.path.exists(d_isochrone_cache):
            with open(indices_path, 'r') as f:
                cache_data = json.load(f)
            o_indices = cache_data['o_indices']
            d_indices = cache_data['d_indices']
            union_indices = list(set(o_indices) | set(d_indices))
            
            raw_features = tf.gather(tf.reshape(self.grid_features, [-1, self.C]), union_indices).numpy()  # (N', C)
            o_isochrone_classes = np.load(o_isochrone_cache)
            d_isochrone_classes = np.load(d_isochrone_cache)
        else:
            print(f"Computing cache for sensor {sensor_id}.")
            o_polygon = get_polygon(sensor_id, 'o')
            d_polygon = get_polygon(sensor_id, 'd')
            
            o_indices, d_indices, union_indices = self.compute_pixel_indices(o_polygon, d_polygon)
            raw_features = tf.gather(tf.reshape(self.grid_features, [-1, self.C]), union_indices).numpy()  # (N', C)
            
            o_isochrone_classes = self.compute_pixel_isochrone_classes(sensor_id, o_indices, 'o') # (N, )
            d_isochrone_classes = self.compute_pixel_isochrone_classes(sensor_id, d_indices, 'd') # (N, )

            np.save(o_isochrone_cache, o_isochrone_classes)
            np.save(d_isochrone_cache, d_isochrone_classes)
            
            cache_data = {'o_indices': o_indices, 'd_indices': d_indices}
            with open(indices_path, 'w') as f:
                json.dump(cache_data, f)
        
        pixel_embeddings = self.feature_mlp(tf.convert_to_tensor(raw_features, dtype=tf.float32))
        
        o_positions = [union_indices.index(i) for i in o_indices]
        d_positions = [union_indices.index(i) for i in d_indices]
        
        o_pixel_embeddings = tf.gather(pixel_embeddings, o_positions)
        d_pixel_embeddings = tf.gather(pixel_embeddings, d_positions)

        return o_pixel_embeddings, d_pixel_embeddings, o_indices, d_indices, o_isochrone_classes, d_isochrone_classes

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
        
    def compute_pixel_isochrone_classes(self, sensor_id, indices, region_type):
        """
        Returns the isochrone class for each pixel indices based on the shortest travel time isochrone.
        
        Args:
            sensor_id (str): Unique identifier of the sensor.
            indices (list): List of grid indices representing selected pixels.
            region_type (str): 'o' for origin, 'd' for destination.
        
        Returns:
            np.ndarray: Isochrone class (N, ), with values 1, 2, ..., 12 indicating isochrone classes.
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
        
        return isochrone_labels