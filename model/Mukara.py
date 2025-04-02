import os
import json
import networkx as nx
import geopandas as gpd
import tensorflow as tf
from config import PATH, MODEL, DATA
from model.dataloader import load_graph, load_mesh_features

class Mukara(tf.keras.Model):
    def __init__(self):
        super(Mukara, self).__init__()
        
        # Load all nodes and edges, and build graph
        print("Loading nodes...")
        with open(PATH['nodes'], 'r') as f:
            self.nodes = json.load(f)  # Dictionary for fast lookup

        print("Loading edges...")
        with open(PATH['edges'], 'r') as f:
            self.edges = json.load(f)  # Dictionary for fast lookup

        # print("Loading graph...")
        # self.graph = load_graph()  # MultiDiGraph built with original OSM IDs
        # print("Graph loaded successfully.")

        self.mesh_features = load_mesh_features()
        with open(PATH['mesh2nodes'], 'r') as f:
            self.mesh2nodes = json.load(f)  # mesh nodes pairing result

        # GNN layers
        self.node_embedder = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(MODEL['embedding_dim'], activation=None)
        ])
        self.edge_embedder = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(MODEL['embedding_dim'], activation=None)
        ])
        self.node_to_edge = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(MODEL['embedding_dim'], activation=None)
        ])
        self.edge_to_node = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(MODEL['embedding_dim'], activation=None)
        ])

        # output layers
        self.output_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, activation=None)
        ])
    
    def call(self, edge_id):
        """Forward pass of the model."""
        subgraph = self.get_subgraph(edge_id)  # Structured JSON subgraph
        if len(subgraph) != MODEL['graph_size']+1:
            print(f"⚠️ Ignoring empty subgraph for edge {edge_id}")
            return None

        # Tensor pools to store embeddings for backpropagation
        edge_embeddings = {}
        node_embeddings = {}

        for degree in sorted(subgraph.keys(), reverse=True):  # Start from outermost
            for update_data in subgraph[degree]["node2edge"]:
                eid = update_data["edge"]
                nid = update_data["node"]
                
                # Retrieve or initialize node embedding
                if nid not in node_embeddings:
                    node_embeddings[nid] = self.node_embedder(self.get_node_features(nid, degree))
                    
                # Retrieve or initialize edge embedding
                if eid not in edge_embeddings:
                    edge_embeddings[eid] = self.edge_embedder(self.get_edge_features(eid, degree))

                # Update edge embedding using node embedding
                edge_embeddings[eid] = self.node_to_edge(tf.concat([node_embeddings[nid], edge_embeddings[eid]], axis=-1))
                
            for update_data in subgraph[degree]["edge2node"]:
                nid = update_data["node"]
                eids = update_data["edge"]
                
                # Retrieve all edge embeddings
                edge_tensors = [edge_embeddings[eid] for eid in eids]
                
                # Aggregate edge embeddings
                aggregated_edges = tf.reduce_sum(tf.stack(edge_tensors), axis=0)

                # Retrieve or initialize node embedding
                if nid not in node_embeddings:
                    node_embeddings[nid] = self.node_embedder(self.get_node_features(nid, degree))

                # Update node embedding using aggregated edges
                node_embeddings[nid] = self.edge_to_node(tf.concat([aggregated_edges, node_embeddings[nid]], axis=-1))

        # Final prediction based on the last updated two nodes of the edge and the edge emebddings
        aggregated_node_embedding = tf.reduce_sum(
            tf.stack([node_embeddings[n["node"]] for n in subgraph["0"]["edge2node"]]), axis=0
        )
        edge_embeddings[str(edge_id)] = self.edge_embedder(self.get_edge_features(str(edge_id), degree))
        edge_embeddings[str(edge_id)] = self.output_mlp(tf.concat([aggregated_node_embedding, edge_embeddings[str(edge_id)]], axis=-1))
        prediction = tf.squeeze((edge_embeddings[str(edge_id)]))
        return prediction
    
    def get_edge_features(self, edge_id, degree):
        # Convert to tensor and reshape to (1, feature_dim) before passing to the model, add degree of current edge relative to the center edge
        edge_feature_tensor = tf.convert_to_tensor(self.edges[edge_id]["feature"] + [float(degree) / MODEL['graph_size']], dtype=tf.float32)
        return tf.expand_dims(edge_feature_tensor, axis=0)
    
    def get_node_features(self, node_id, degree):
        # Retrieve the list of pixel indices for the given node
        pixel_indices = self.mesh2nodes.get(str(node_id), [])

        # Extract pixel features from self.mesh_features using the indices
        pixel_features = [self.mesh_features[i // self.mesh_features.shape[1], i % self.mesh_features.shape[1], :]
                        for i in pixel_indices]

        # Aggregate pixel features using sum
        aggregated_pixel_feature = tf.reduce_sum(tf.stack(pixel_features), axis=0) if pixel_features else tf.zeros_like(self.mesh_features[0, 0, :])

        # Append degree feature to aggregated pixel features
        degree_feature = tf.convert_to_tensor([float(degree) / MODEL['graph_size']], dtype=tf.float32)
        aggregated_pixel_feature = tf.concat([aggregated_pixel_feature, degree_feature], axis=0)

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
