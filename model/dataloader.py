import tensorflow as tf
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
tf.random.set_seed(TRAINING['seed'])
random.seed(TRAINING['seed'])


def load_mesh_features():
    """
    Load the grid feature tensor with z-score normalization.
    Returns:
        tf tensor: shape=(height, width, channel) representing the z-score normalized pixel features for 2022.
    """
    with h5py.File(PATH["population_and_employment"], 'r') as f:  
        pe = f['features'][..., [item for sublist in [DATA['population'], DATA['employment']] for item in sublist]]
    year, row, col, c = pe.shape
    pe = np.reshape(pe, (year * row * col, -1))
    pe = (pe - np.mean(pe, axis=0)) / (np.std(pe, axis=0) + 1e-8)	
    pe = np.reshape(pe, (year, row, col, -1))
    pe = tf.convert_to_tensor(pe, dtype=tf.float32)

    with h5py.File(PATH["landuse_and_poi"], "r") as h5f:
        lp = h5f["features"][..., [item for item in DATA['landuse_poi']]]
    lp = np.reshape(lp, (row * col, -1))
    lp = (lp - np.mean(lp, axis=0)) / (np.std(lp, axis=0) + 1e-8)	
    lp = np.reshape(lp, (row, col, -1))
    lp = tf.convert_to_tensor(lp, dtype=tf.float32)
    lp = tf.broadcast_to(lp, (year, row, col, lp.shape[-1]))
    grid_features = tf.concat([pe, lp], axis=-1)
    grid_features = grid_features[-1,...] # year 2022
    print(f"Grid features shape {grid_features.shape}")

    return grid_features


def load_mesh_cells():
    """
    Load the spatial grid cells, ensure they are in EPSG:27700, 
    and normalize their geometries.

    Returns:
        geopandas.GeoDataFrame: Grid cells with fully normalized geometries.
    """
    grid_cells = gpd.read_file(os.path.join(PATH['grid']))

    # Ensure CRS is correct
    if grid_cells.crs is None:
        raise ValueError("Grid cells shapefile does not have a CRS defined.")
    
    if grid_cells.crs.to_epsg() != 27700:
        raise ValueError("The CRS of the shapefile is not EPSG:27700.")

    return grid_cells


def get_pixel_coordinates(grid_cells, indices):
    """
    Fetches pixel coordinates given their indices in the grid.

    Returns:
        tf.Tensor: (N, 2) x, y coordinates of pixels.
    """
    pixel_coords = np.array([(grid_cells.geometry.iloc[i].centroid.x, grid_cells.geometry.iloc[i].centroid.y) for i in indices])
    return tf.convert_to_tensor(pixel_coords, dtype=tf.float32)  # (N, 2)


def load_gt():
    """
    Load the ground truth (traffic volume).
    Returns: 
        A tuple (id_to_gt, scaler), where:
        - id_to_gt is a dictionary {edge_id: normalized_traffic_volume}
        - scaler is the Scaler instance used for normalization.
    """
    gt_df = pd.read_csv(PATH["ground_truth"])

    edge_ids = gt_df["edge_id"].values
    traffic_volume = gt_df["Traffic Volume"].values.astype(float)

    traffic_volume_tensor = tf.convert_to_tensor(traffic_volume, dtype=tf.float32)

    scaler = Scaler(traffic_volume_tensor)
    traffic_volume_normalized = scaler.transform(traffic_volume_tensor)

    edge_to_gt = dict(zip(edge_ids, traffic_volume_normalized.numpy()))

    return edge_to_gt, scaler

def load_dgl_graph():
    """
    Builds a DGL graph from GeoDataFrames and attaches edge features computed from structural attributes and aggregated pixel-level data.
    Returns the graph and a list of external edge_ids (globally consistent in this project) in the sequence of DGL internal edge order.
    """
    # Load node and edge GeoDataFrames
    nodes = gpd.read_file(PATH['graph_nodes'])
    edges = gpd.read_file(PATH['graph_edges'])

    edges.set_index(['u', 'v', 'key'], inplace=True)

    # Load edge features from JSON
    with open(PATH['edge_features'], 'r') as f:
        edge_feature_dict = json.load(f)

    # Load mesh features and pixel-to-edge mapping
    mesh_feats = load_mesh_features()
    with open(PATH['mesh2edges'], 'r') as f:
        mesh_map = json.load(f)

    # Build full graph with edge_id as edge attribute
    graph_nx = ox.graph_from_gdfs(nodes, edges)
    g = dgl.from_networkx(graph_nx, edge_attrs=['edge_id'])

    # Retrieve edge_ids in DGL's internal order
    edge_ids = g.edata['edge_id'].numpy().tolist()
    edge_features = []

    for eid in tqdm(edge_ids, desc="Processing edge features"):
        eid_str = str(eid)
        base_feat = edge_feature_dict[eid_str]['feature']
        pixel_ids = mesh_map.get(eid_str, [])
        pixel_embs = [mesh_feats[i // mesh_feats.shape[1], i % mesh_feats.shape[1], :] for i in pixel_ids]
        pixel_sum = np.sum(pixel_embs, axis=0) if pixel_embs else np.zeros_like(mesh_feats[0, 0, :])
        final_feat = np.concatenate([base_feat, pixel_sum])
        edge_features.append(final_feat)

    edge_features = tf.convert_to_tensor(np.stack(edge_features), dtype=tf.float32)
    g.edata['feat'] = edge_features

    return g, edge_ids
