import tensorflow as tf
import numpy as np
from config import MODEL
from model.dataloader import get_pixel_coordinates

class FixedPatchClustering(tf.keras.layers.Layer):
    def __init__(self, name, grid_cells):
        super(FixedPatchClustering, self).__init__(name=name)

        self.grid_cells = grid_cells  # Store grid cells for position lookup
        self.patch_size = MODEL['patch_size']  # Fixed patch size (in pixels)

        # MLP to project sum-pooled cluster features into final embeddings
        self.cluster_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(MODEL['embedding_dim']/2, activation='relu', name=name+'_mlp_3'),
            tf.keras.layers.Dense(MODEL['embedding_dim'], name=name+'_output')
        ], name=name+'_cluster_mlp')

    def call(self, pixel_features, pixel_indices, pixel_isochrone_classes):
        """
        Assigns pixels to fixed-size patches and computes cluster embeddings.

        Args:
            pixel_features (tf.Tensor): (N, C) Raw feature vectors for pixels.
            pixel_indices (list): Indices of pixels in the grid.
            pixel_isochrone_classes (tf.Tensor): (N,) Isochrone class (1-12) for each pixel.

        Returns:
            tuple:
                - cluster_embeddings (tf.Tensor): (K, D) Transformed cluster embeddings.
                - cluster_centroids (tf.Tensor): (K, 2) Cluster centroid coordinates.
                - cluster_isochrone_encodings (tf.Tensor): (K, 12) Soft-clustered isochrone encodings.
        """

        # Retrieve pixel coordinates from grid indices (N, 2)
        pixel_positions = get_pixel_coordinates(self.grid_cells, pixel_indices)  # (N, 2)

        # Determine the bounding box of the entire area
        min_x, min_y = tf.reduce_min(pixel_positions, axis=0)  # Find the minimum x and y
        max_x, max_y = tf.reduce_max(pixel_positions, axis=0)  # Find the maximum x and y

        # Compute the number of patches in x and y directions
        num_patches_x = tf.cast(tf.math.ceil((max_x - min_x) / self.patch_size), tf.int32)
        # num_patches_y = tf.cast(tf.math.ceil((max_y - min_y) / self.patch_size), tf.int32)

        # Assign each pixel to a patch index based on its position
        patch_x = tf.cast((pixel_positions[:, 0] - min_x) / self.patch_size, tf.int32)  # (N,)
        patch_y = tf.cast((pixel_positions[:, 1] - min_y) / self.patch_size, tf.int32)  # (N,)
        patch_indices = patch_y * num_patches_x + patch_x  # Unique patch ID

        # Get unique patch indices
        unique_patches, _ = tf.unique(patch_indices)  # Unique patch IDs
        # num_clusters = tf.shape(unique_patches)[0]  # K (variable number of clusters)

        # Initialize lists for storing computed values
        centroids = []
        cluster_features = []
        cluster_isochrone_encodings = []

        # Compute centroids, sum-pooled features, and isochrone encoding in a single loop
        for patch in unique_patches:
            mask = tf.equal(patch_indices, patch)

            # Select data for this patch
            selected_positions = tf.boolean_mask(pixel_positions, mask)  # (N', 2)
            selected_features = tf.boolean_mask(pixel_features, mask)  # (N', C)
            selected_classes = tf.boolean_mask(pixel_isochrone_classes, mask)  # (N',)

            # Compute cluster centroid (mean position)
            centroid = tf.reduce_mean(selected_positions, axis=0)  # (2,)
            centroids.append(centroid)

            # Compute sum-pooled feature vectors
            summed_features = tf.reduce_sum(selected_features, axis=0)  # (C,)
            cluster_features.append(summed_features)

            # Compute isochrone encoding (soft-clustered distribution)
            one_hot_classes = tf.one_hot(selected_classes - 1, depth=12)  # (N', 12)
            isochrone_encoding = tf.reduce_mean(one_hot_classes, axis=0)  # (12,)
            cluster_isochrone_encodings.append(isochrone_encoding)

        # Convert lists to tensors
        cluster_centroids = tf.stack(centroids, axis=0)  # (K, 2)
        cluster_features = tf.stack(cluster_features, axis=0)  # (K, C)
        cluster_isochrone_encodings = tf.stack(cluster_isochrone_encodings, axis=0)  # (K, 12)

        # Transform sum-pooled cluster features into final cluster embeddings using MLP
        cluster_embeddings = self.cluster_mlp(cluster_features)  # (K, D)

        return cluster_embeddings, cluster_centroids, cluster_isochrone_encodings