import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import json
import os
from config import MODEL, PATH
from model.dataloader import load_mesh_features

class Mukara(nn.Module):
    def __init__(self):
        super(Mukara, self).__init__()

        self.z_shift = 0
        self.embedding_dim = MODEL['embedding_dim']
        self.mesh_features = load_mesh_features()
        self.mesh_flat = self.mesh_features.reshape(-1, self.mesh_features.shape[-1])

        # Load once
        with open(PATH['edge_features'], 'r') as f:
            self.edge_feature_dict = json.load(f)
        with open(PATH['mesh2nodes'], 'r') as f:
            self.mesh2nodes = json.load(f)
        
        self.node_dim = self.mesh_flat.shape[1]
        self.edge_dim = len(next(iter(self.edge_feature_dict.values()))["feature"])
        self.feature_dim = max(self.node_dim, self.edge_dim)  # unified feature dim

        # === Model layers ===
        input_dim = self.feature_dim + 1  # +1 for degree

        self.node_embedder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.embedding_dim)
        )

        self.edge_embedder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.embedding_dim)
        )

        self.edge_to_node = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, self.embedding_dim)
        )

        self.node_to_edge = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, self.embedding_dim)
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(self.embedding_dim *2 , 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, edge_id):
        g, center_node_ids = self.build_dgl_subgraph_from_json(edge_id)
        if g is None:
            return None

        # Add normalized degree to feature
        deg_norm = g.ndata['degree'].float().unsqueeze(1) / MODEL['graph_size']
        full_input = torch.cat([g.ndata['feat'], deg_norm], dim=-1)

        node_mask = g.ndata['type'] == 0
        edge_mask = g.ndata['type'] == 1

        node_emb = self.node_embedder(full_input[node_mask])
        edge_emb = self.edge_embedder(full_input[edge_mask])

        g.ndata['h'] = torch.zeros(g.num_nodes(), self.embedding_dim, device=full_input.device)
        g.ndata['h'][node_mask] = node_emb
        g.ndata['h'][edge_mask] = edge_emb

        max_deg = g.ndata['degree'].max().item()
        for d in reversed(range(max_deg + 1)):
            # Mask for node → edge updates
            node2edge_mask = (g.ndata['degree'] == d) & (g.ndata['type'] == 1)  # update edges
            edge2node_mask = (g.ndata['degree'] == d) & (g.ndata['type'] == 0)  # update nodes

            # === Node → Edge ===
            g.update_all(
                message_func=fn.copy_u('h', 'm'),
                reduce_func=fn.sum('m', 'agg_msg')
            )

            if 'agg_msg' in g.ndata:
                agg_msg = g.ndata['agg_msg']
                updated = self.node_to_edge(torch.cat([agg_msg, g.ndata['h']], dim=-1))
                g.ndata['h'] = torch.where(node2edge_mask.unsqueeze(1), updated, g.ndata['h'])

            # === Edge → Node ===
            g.update_all(
                message_func=fn.copy_u('h', 'm'),
                reduce_func=fn.sum('m', 'agg_msg')
            )

            if 'agg_msg' in g.ndata:
                agg_msg = g.ndata['agg_msg']
                updated = self.edge_to_node(torch.cat([agg_msg, g.ndata['h']], dim=-1))
                g.ndata['h'] = torch.where(edge2node_mask.unsqueeze(1), updated, g.ndata['h'])

        # === Final prediction ===
        center_edge_idx = g.num_nodes() - 1
        center_edge_emb = g.ndata['h'][center_edge_idx]

        center_node_embs = g.ndata['h'][center_node_ids]
        node_sum = torch.sum(center_node_embs, dim=0)

        final_input = torch.cat([center_edge_emb, node_sum], dim=-1)
        z_pred = self.output_mlp(final_input).squeeze(-1)
        return F.softplus(z_pred - self.z_shift) + self.z_shift


    def build_dgl_subgraph_from_json(self, edge_id):
        path = os.path.join(PATH['cache'], f"{edge_id}.json")
        if not os.path.exists(path):
            print(f"⚠️ Missing subgraph cache for edge {edge_id}")
            return None

        with open(path, 'r') as f:
            subgraph = json.load(f)        

        id_map = {}
        current_idx = 0

        features = []
        degrees = []
        types = [] # 'node', 'edge', or 'center_edge'
        src_list = []
        dst_list = []
        center_node_keys = []

        for degree_str in sorted(subgraph.keys(), key=int, reverse=True):
            degree = int(degree_str)
            data = subgraph[degree_str]

            if degree == 0:
                for conn in data['edge2node']:
                    center_node_keys.append(f"n{conn['node']}")

            for conn in data['node2edge']:
                nid = conn['node']
                eid = conn['edge']

                # Node
                node_key = f"n{nid}"
                if node_key not in id_map:
                    id_map[node_key] = current_idx
                    pixel_ids = self.mesh2nodes.get(str(nid), [])
                    if pixel_ids:
                        feats = [torch.tensor(self.mesh_flat[i], dtype=torch.float32) for i in pixel_ids]
                        feat = torch.sum(torch.stack(feats), dim=0)
                    else:
                        feat = torch.zeros((self.node_dim,), dtype=torch.float32)
                    padded_feat = torch.nn.functional.pad(feat, (0, self.feature_dim - self.node_dim))
                    features.append(padded_feat)
                    degrees.append(degree)
                    types.append(0)
                    current_idx += 1

                # Edge
                edge_key = f"e{eid}"
                if edge_key not in id_map:
                    id_map[edge_key] = current_idx
                    base_feat = torch.tensor(self.edge_feature_dict[str(eid)]["feature"], dtype=torch.float32)
                    padded_feat = torch.nn.functional.pad(base_feat, (0, self.feature_dim - self.edge_dim))
                    features.append(padded_feat)
                    degrees.append(degree)
                    types.append(1)
                    current_idx += 1

                # node → edge
                src_list.append(id_map[node_key])
                dst_list.append(id_map[edge_key])

            for conn in data['edge2node']:
                nid = conn['node']
                eids = conn['edge']
                node_key = f"n{nid}"

                if node_key not in id_map:
                    id_map[node_key] = current_idx
                    pixel_ids = self.mesh2nodes.get(str(nid), [])
                    if pixel_ids:
                        feats = [torch.tensor(self.mesh_flat[i], dtype=torch.float32) for i in pixel_ids]
                        feat = torch.sum(torch.stack(feats), dim=0)
                    else:
                        feat = torch.zeros((self.node_dim,), dtype=torch.float32)
                    padded_feat = torch.nn.functional.pad(feat, (0, self.feature_dim - self.node_dim))
                    features.append(padded_feat)
                    degrees.append(degree)
                    types.append(0)
                    current_idx += 1

                for eid in eids:
                    edge_key = f"e{eid}"
                    if edge_key not in id_map:
                        id_map[edge_key] = current_idx
                        base_feat = torch.tensor(self.edge_feature_dict[str(eid)]["feature"], dtype=torch.float32)
                        padded_feat = torch.nn.functional.pad(base_feat, (0, self.feature_dim - self.edge_dim))
                        features.append(padded_feat)
                        degrees.append(degree)
                        types.append(1)
                        current_idx += 1

                    # edge → node
                    src_list.append(id_map[edge_key])
                    dst_list.append(id_map[node_key])

        # === Add the center edge as a DGL node ===
        center_edge_feat = torch.tensor(self.edge_feature_dict[str(edge_id)]["feature"], dtype=torch.float32)
        padded_feat = torch.nn.functional.pad(center_edge_feat, (0, self.feature_dim - self.edge_dim))
        features.append(padded_feat)
        degrees.append(0)       # Always degree 0
        types.append(1)         # Same type as regular edges
        id_map[f"e{edge_id}"] = current_idx
        current_idx += 1

        g = dgl.graph((src_list, dst_list), num_nodes=current_idx)
        g.ndata['feat'] = torch.stack(features)
        g.ndata['degree'] = torch.tensor(degrees, dtype=torch.int64)
        g.ndata['type'] = torch.tensor(types, dtype=torch.long)
        g.edata['h'] = torch.zeros(g.num_edges(), self.embedding_dim)

        center_node_ids = [id_map[k] for k in center_node_keys if k in id_map]

        return g, center_node_ids