import torch
import h5py
import os
import numpy as np
import random
from tqdm import tqdm
import pandas as pd
import geopandas as gpd
import json
import pyproj
import osmnx as ox
import dgl

from model.utils import Scaler
from config import PATH, DATA, TRAINING

np.random.seed(TRAINING['seed'])
torch.manual_seed(TRAINING['seed'])
random.seed(TRAINING['seed'])

def load_mesh_features():
    """
    Load the grid feature tensor with z-score normalization.
    Returns:
        torch tensor: shape=(height, width, channel) representing the z-score normalized pixel features for 2022.
    """
    with h5py.File(PATH["population_and_employment"], 'r') as f:
        pe = f['features'][..., [item for sublist in [DATA['population'], DATA['employment']] for item in sublist]]
    year, row, col, c = pe.shape
    pe = np.reshape(pe, (year * row * col, -1))
    pe = (pe - np.mean(pe, axis=0)) / (np.std(pe, axis=0) + 1e-8)
    pe = np.reshape(pe, (year, row, col, -1))
    pe = torch.tensor(pe, dtype=torch.float32)

    with h5py.File(PATH["landuse_and_poi"], "r") as h5f:
        lp = h5f["features"][..., [item for item in DATA['landuse_poi']]]
    lp = np.reshape(lp, (row * col, -1))
    lp = (lp - np.mean(lp, axis=0)) / (np.std(lp, axis=0) + 1e-8)
    lp = np.reshape(lp, (row, col, -1))
    lp = torch.tensor(lp, dtype=torch.float32)
    lp = lp.unsqueeze(0).expand(year, -1, -1, -1)  # broadcast
    grid_features = torch.cat([pe, lp], dim=-1)
    grid_features = grid_features[-1, ...]  # year 2022
    print(f"Grid features shape {grid_features.shape}")

    return grid_features

def load_mesh_cells():
    grid_cells = gpd.read_file(os.path.join(PATH['grid']))

    if grid_cells.crs is None:
        raise ValueError("Grid cells shapefile does not have a CRS defined.")

    if grid_cells.crs.to_epsg() != 27700:
        raise ValueError("The CRS of the shapefile is not EPSG:27700.")

    return grid_cells

def get_pixel_coordinates(grid_cells, indices):
    pixel_coords = np.array([(grid_cells.geometry.iloc[i].centroid.x, grid_cells.geometry.iloc[i].centroid.y) for i in indices])
    return torch.tensor(pixel_coords, dtype=torch.float32)  # (N, 2)

def load_gt():
    gt_df = pd.read_csv(PATH["ground_truth"])

    edge_ids = gt_df["edge_id"].values
    traffic_volume = gt_df["Traffic Volume"].values.astype(float)

    traffic_volume_tensor = torch.tensor(traffic_volume, dtype=torch.float32)
    scaler = Scaler(traffic_volume_tensor)
    traffic_volume_normalized = scaler.transform(traffic_volume_tensor)

    edge_to_gt = {str(eid): val for eid, val in zip(edge_ids, traffic_volume_normalized.cpu().numpy())}

    return edge_to_gt, scaler

def load_dgl_graph():
    nodes = gpd.read_file(PATH['graph_nodes'])
    edges = gpd.read_file(PATH['graph_edges'])

    edges.set_index(['u', 'v', 'key'], inplace=True)

    with open(PATH['edge_features'], 'r') as f:
        edge_feature_dict = json.load(f)

    mesh_feats = load_mesh_features()
    with open(PATH['mesh2edges'], 'r') as f:
        mesh_map = json.load(f)

    graph_nx = ox.graph_from_gdfs(nodes, edges)
    g = dgl.from_networkx(graph_nx, edge_attrs=['edge_id'])

    edge_ids = g.edata['edge_id'].numpy().tolist()
    edge_features = []

    for eid in tqdm(edge_ids, desc="Processing edge features"):
        eid_str = str(eid)
        base_feat = edge_feature_dict[eid_str]['feature']
        pixel_ids = mesh_map.get(eid_str, [])
        pixel_embs = [mesh_feats[i // mesh_feats.shape[1], i % mesh_feats.shape[1], :] for i in pixel_ids]
        pixel_sum = torch.sum(torch.stack(pixel_embs), dim=0) if pixel_embs else torch.zeros_like(mesh_feats[0, 0, :])
        final_feat = torch.tensor(np.concatenate([base_feat, pixel_sum.numpy()]), dtype=torch.float32)
        edge_features.append(final_feat)

    g.edata['feat'] = torch.stack(edge_features)

    return g, edge_ids
