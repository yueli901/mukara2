import tensorflow as tf
from config import MODEL

class CustomAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, name, grid_cells):
        super(CustomAttentionLayer, self).__init__(name=name)
        
        self.grid_cells = grid_cells  # Store grid cells for spatial reference

        # Layer Normalization
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=name+'_norm1')
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=name+'_norm2')

        # Feed Forward Network (FFN)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(MODEL['embedding_dim'] * 4, activation='relu', name=name+'_ffn_1'),
            tf.keras.layers.Dense(MODEL['embedding_dim'], name=name+'_ffn_2')
        ], name=name+'_ffn')

        # MLPs for learnable G and H
        self.G_MLP = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', name=name+'_g_mlp_1'),
            tf.keras.layers.Dense(1, activation='relu', name=name+'_g_mlp_2')
        ], name=name+'_g_mlp')

        self.H_MLP = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', name=name+'_h_mlp_1'),
            tf.keras.layers.Dense(1, name=name+'_h_mlp_2')
        ], name=name+'_h_mlp')

    def call(self, sensor_coords, query, key, q_cluster_centroids, k_cluster_centroids, q_cluster_isochrone_encodings, k_cluster_isochrone_encodings):
        """
        Custom Self-Attention and Cross-Attention Mechanism with Geometric Encoding.

        Args:
            sensor_coords (tf.Tensor): (2,) sensor x, y coordinates.
            query, key (tf.Tensor): (K, D) and (K', D) cluster embeddings.
            q_cluster_centroids, k_cluster_centroids (tf.Tensor): (K, 2) and (K', 2) cluster centroids.
            q_cluster_isochrone_encodings, k_cluster_isochrone_encodings (tf.Tensor): (K, 12) and (K', 12) isochrone encodings.

        Returns:
            tf.Tensor: (K, D) Updated cluster embeddings after attention.
        """

        # Get the number of clusters in q (K) and k (K')
        K = tf.shape(q_cluster_centroids)[0]  # Number of clusters in q
        K_prime = tf.shape(k_cluster_centroids)[0]  # Number of clusters in k

        # Compute pairwise distance vectors and their norms
        r_kk_prime = q_cluster_centroids[:, None, :] - k_cluster_centroids[None, :, :]  # (K, K', 2)
        r_ks = q_cluster_centroids - sensor_coords  # (K, 2)
        r_k_prime_s = k_cluster_centroids - sensor_coords  # (K', 2)

        norm_r_kk_prime = tf.norm(r_kk_prime, axis=-1)  # (K, K')
        norm_r_ks = tf.norm(r_ks, axis=-1)  # (K,)
        norm_r_k_prime_s = tf.norm(r_k_prime_s, axis=-1)  # (K',)

        # Compute attention bias
        R_kk_prime = self.attention_bias(q_cluster_isochrone_encodings, k_cluster_isochrone_encodings, norm_r_kk_prime, norm_r_ks, norm_r_k_prime_s, K, K_prime)

        # Compute scaled dot-product attention
        attention_logits = self.compute_scaled_attention(query, key)  # (K, K')
        attention_logits = attention_logits + R_kk_prime

        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(attention_logits, axis=-1)  # (K, K')

        # Compute new attention output
        attention_output = tf.matmul(attention_weights, key)  # (K, D)

        # Residual connection and LayerNorm
        query = self.norm1(query + attention_output)

        # Feed Forward Network
        ffn_output = self.ffn(query)

        # Second Residual connection and LayerNorm
        query = self.norm2(query + ffn_output)

        return query

    def compute_scaled_attention(self, query, key):
        """
        Compute scaled dot-product attention logits.

        Args:
            query (tf.Tensor): (K, D) Cluster embeddings.
            key (tf.Tensor): (K', D) Cluster embeddings.

        Returns:
            tf.Tensor: (K, K') Attention logits.
        """
        d_k = tf.cast(tf.shape(query)[-1], tf.float32)  # Scaling factor
        return tf.matmul(query, key, transpose_b=True) / tf.sqrt(d_k)  # (K, K')

    def attention_bias(self, d_k, d_k_prime, norm_r_kk_prime, norm_r_ks, norm_r_k_prime_s, K, K_prime):
        """
        Compute the geometric-aware scaling factor for attention using MLP.

        Args:
            d_k, d_k_prime (tf.Tensor): (K, 12) and (K', 12) Soft-clustered isochrone class encodings.
            norm_r_kk_prime, norm_r_ks, norm_r_k_prime_s (tf.Tensor): (K, K'), (K,), and (K',) norms of spatial distances.
            K (int): Number of clusters in q.
            K_prime (int): Number of clusters in k.

        Returns:
            tf.Tensor: (K, K') Geometric attention scaling factor.
        """
        # Explicitly broadcast d_k and d_k_prime to match (K, K', 12)
        d_k_broadcast = tf.broadcast_to(d_k[:, None, :], [K, K_prime, 12])  # (K, K', 12)
        d_k_prime_broadcast = tf.broadcast_to(d_k_prime[None, :, :], [K, K_prime, 12])  # (K, K', 12)

        # Compute G using learnable MLP
        G_input = tf.concat([d_k_broadcast, d_k_prime_broadcast], axis=-1)  # (K, K', 24)
        G = self.G_MLP(G_input)  # (K, K', 1)

        # Expand and broadcast norm_r_ks
        norm_r_ks_broadcast = tf.broadcast_to(norm_r_ks[:, None], [K, K_prime])  # (K, K')
        norm_r_ks_broadcast = tf.expand_dims(norm_r_ks_broadcast, axis=-1)  # (K, K', 1)

        # Expand and broadcast norm_r_k_prime_s
        norm_r_k_prime_s_broadcast = tf.broadcast_to(norm_r_k_prime_s[None, :], [K, K_prime])  # (K, K')
        norm_r_k_prime_s_broadcast = tf.expand_dims(norm_r_k_prime_s_broadcast, axis=-1)  # (K, K', 1)

        # Now they both have shape (K, K', 1) and can be concatenated
        H_input = tf.concat([norm_r_kk_prime[..., None], norm_r_ks_broadcast, norm_r_k_prime_s_broadcast], axis=-1)  # (K, K', 3)
        H = self.H_MLP(H_input)  # (K, K', 1)

        # Compute final geometric attention scaling factor
        return tf.squeeze(G * H, axis=-1)  # (K, K')