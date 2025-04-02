import os
import json
from tqdm import tqdm
import tensorflow as tf
from config import PATH, MODEL, DATA
from model.dataloader import load_graph, load_mesh_features

class Mukara(tf.keras.Model):
    def __init__(self):
        super(Mukara, self).__init__()

        # Load node/edge data and features 
        print("Loading nodes...")
        with open(PATH['nodes'], 'r') as f:
            self.nodes = json.load(f)

        print("Loading edges...")
        with open(PATH['edges'], 'r') as f:
            self.edges = json.load(f)

        self.mesh_features = load_mesh_features()
        with open(PATH['mesh2nodes'], 'r') as f:
            self.mesh2nodes = json.load(f)

        # Node embedder
        self.node_embedder = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', name="node_embed_dense1"),
            tf.keras.layers.Dense(MODEL['embedding_dim'], activation=None, name="node_embed_dense2")
        ], name="node_embedder")

        # Edge embedder
        self.edge_embedder = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', name="edge_embed_dense1"),
            tf.keras.layers.Dense(MODEL['embedding_dim'], activation=None, name="edge_embed_dense2")
        ], name="edge_embedder")

        # Node-to-edge updater
        self.node_to_edge = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', name="node2edge_dense1"),
            tf.keras.layers.Dense(MODEL['embedding_dim'], activation=None, name="node2edge_dense2")
        ], name="node_to_edge")

        # Edge-to-node attention heads
        self.attn_heads = MODEL['attention_heads']
        self.attn_query = [
            tf.keras.layers.Dense(MODEL['embedding_dim'], use_bias=False, name=f"attn_q_head{i}")
            for i in range(self.attn_heads)
        ]
        self.attn_key = [
            tf.keras.layers.Dense(MODEL['embedding_dim'], use_bias=False, name=f"attn_k_head{i}")
            for i in range(self.attn_heads)
        ]

        # Edge-to-node updater (post-attention)
        self.edge_to_node = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', name="edge2node_dense1"),
            tf.keras.layers.Dense(MODEL['embedding_dim'], activation=None, name="edge2node_dense2")
        ], name="edge_to_node")

        # layer norm for edge and node update
        self.edge_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="edge_norm")
        self.node_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="node_norm")

        # Output MLP (after final edge update)
        self.output_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', name="output_dense1"),
            tf.keras.layers.Dense(1, activation=None, name="output_dense2")
        ], name="output_mlp")
    
    def call(self, edge_id):
        """Forward pass of the model."""
        if isinstance(edge_id, tf.Tensor):
            edge_id = edge_id.numpy().item()
        edge_id = str(edge_id)

        subgraph = self.get_subgraph(edge_id)

        if len(subgraph) != MODEL['graph_size'] + 1:
            print(f"⚠️ Ignoring empty subgraph for edge {edge_id}")
            return None

        edge_embeddings = {}
        node_embeddings = {}

        for degree in sorted(subgraph.keys(), reverse=True):
            for update_data in subgraph[degree]["node2edge"]:
                eid = update_data["edge"]
                nid = update_data["node"]

                if nid not in node_embeddings:
                    node_embeddings[nid] = self.node_embedder(self.get_node_features(nid, degree))

                if eid not in edge_embeddings:
                    edge_embeddings[eid] = self.edge_embedder(self.get_edge_features(eid, degree))

                updated = self.node_to_edge(tf.concat([node_embeddings[nid], edge_embeddings[eid]], axis=-1))
                edge_embeddings[eid] = self.edge_norm(updated + edge_embeddings[eid])

            for update_data in subgraph[degree]["edge2node"]:
                nid = update_data["node"]
                eids = update_data["edge"]

                edge_embs = [edge_embeddings[eid] for eid in eids]

                if nid not in node_embeddings:
                    node_embeddings[nid] = self.node_embedder(self.get_node_features(nid, degree))
                node_embed = node_embeddings[nid]

                attn_outputs = []
                for i in range(self.attn_heads):
                    queries = self.attn_query[i](node_embed)
                    keys = tf.stack([self.attn_key[i](emb) for emb in edge_embs], axis=0)

                    attn_scores = tf.nn.softmax(tf.matmul(keys, queries, transpose_b=True) / tf.math.sqrt(tf.cast(MODEL['embedding_dim'], tf.float32)), axis=0)
                    weighted_sum = tf.reduce_sum(attn_scores * tf.stack(edge_embs, axis=0), axis=0)
                    attn_outputs.append(weighted_sum)

                attn_out = tf.concat(attn_outputs, axis=-1)
                updated = self.edge_to_node(tf.concat([attn_out, node_embed], axis=-1))
                node_embeddings[nid] = self.node_norm(updated + node_embed)

        aggregated_node_embedding = tf.reduce_sum(
            tf.stack([node_embeddings[n["node"]] for n in subgraph["0"]["edge2node"]]), axis=0
        )
        edge_embeddings[edge_id] = self.edge_embedder(self.get_edge_features(edge_id, degree))
        edge_embeddings[edge_id] = self.output_mlp(tf.concat([aggregated_node_embedding, edge_embeddings[edge_id]], axis=-1))
        prediction = tf.squeeze(edge_embeddings[edge_id])
        return prediction
        
    def get_edge_features(self, edge_id, degree):
        # Convert to tensor and reshape to (1, feature_dim) before passing to the model, add degree of current edge relative to the center edge
        edge_feature_tensor = tf.convert_to_tensor(self.edges[edge_id]["feature"] + [float(degree) / MODEL['graph_size']], dtype=tf.float32)
        return tf.expand_dims(edge_feature_tensor, axis=0)
    
    def get_node_features(self, node_id, degree):
        # Retrieve original node features, street count
        street_count = self.nodes.get(str(node_id), [])["street_count"]

        # Retrieve the list of pixel indices for the given node
        pixel_indices = self.mesh2nodes.get(str(node_id), [])

        # Extract pixel features from self.mesh_features using the indices
        pixel_features = [self.mesh_features[i // self.mesh_features.shape[1], i % self.mesh_features.shape[1], :]
                        for i in pixel_indices]

        # Aggregate pixel features using sum
        aggregated_pixel_feature = tf.reduce_sum(tf.stack(pixel_features), axis=0) if pixel_features else tf.zeros_like(self.mesh_features[0, 0, :])

        # Finalize node features
        degree_feature = tf.convert_to_tensor([float(degree) / MODEL['graph_size']], dtype=tf.float32)
        street_count = tf.convert_to_tensor([street_count / 3], dtype=tf.float32)
        aggregated_pixel_feature = tf.concat([aggregated_pixel_feature, degree_feature, street_count], axis=0)

        return tf.expand_dims(aggregated_pixel_feature, axis=0)
        
    def get_subgraph(self, edge_id):
        """Compute the subgraph structure dynamically if not cached."""
        if os.path.exists(f"{PATH['cache']}/{edge_id}.json"):
            with open(f"{PATH['cache']}/{edge_id}.json", 'r') as f:
                return json.load(f)
        
        print(f"Caching subgraph for edge {edge_id}.")
        center_edge = edge_id
        subgraph = {}
        visited_edges = set()
        visited_nodes = set()
        queue = [(center_edge, 0)]  # (edge_id, degree)
        
        while queue:
            current_edge, degree = queue.pop(0)

            if degree > MODEL['graph_size']:
                break
            
            if current_edge in visited_edges:
                continue
            
            visited_edges.add(current_edge)
            if degree not in subgraph:
                subgraph[degree] = {"node2edge": [], "edge2node": []}
            
            # Retrieve edge properties
            edge_data = self.edges[str(current_edge)]
            u, v = edge_data['u'], edge_data['v']
            
            for neighbor in [u, v]:
                outgoing_edges = []
                for next_v, next_key in self.graph[neighbor].items():
                    for edge_key, edge_data in next_key.items():
                        next_edge = edge_data['edge_id']
                        edge_data = self.edges[str(next_edge)]
                        next_node = edge_data['v'] if edge_data['u'] == neighbor else edge_data['u']
                        if next_node not in visited_nodes and next_edge not in visited_edges:
                            queue.append((next_edge, degree + 1))
                            outgoing_edges.append(str(next_edge))
                            visited_nodes.add(next_node)
                            subgraph[degree]["node2edge"].append({"node": next_node, "edge": str(next_edge)})
            
                if outgoing_edges:
                    subgraph[degree]["edge2node"].append({"node": neighbor, "edge": outgoing_edges})
        
        # Save to cache
        with open(f"{PATH['cache']}/{edge_id}.json", 'w') as f:
            json.dump(subgraph, f, indent=4)
        
        return subgraph

    def load_all_subgraphs(self):
        """Load all subgraph JSONs from disk into RAM"""
        cache_dir = PATH["cache"]
        self.subgraph_cache = {}

        filenames = [f for f in os.listdir(cache_dir) if f.endswith(".json")]

        for filename in tqdm(filenames, desc="Loading subgraphs", unit="file"):
            edge_id = filename.replace(".json", "")
            with open(os.path.join(cache_dir, filename), 'r') as f:
                self.subgraph_cache[edge_id] = json.load(f)