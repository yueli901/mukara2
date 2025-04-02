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

def load_graph():
    """
    Rebuild the graph from the geojson file of nodes and edges.
    """
    nodes = gpd.read_file(PATH['graph_nodes'])
    edges = gpd.read_file(PATH['graph_edges'])

    # If the index is a simple integer index, fix it:
    if not isinstance(edges.index, pd.MultiIndex):
        # Ensure columns 'u', 'v', and 'key' exist
        if {'u', 'v', 'key'}.issubset(edges.columns):
            edges.set_index(['u', 'v', 'key'], inplace=True)
    print(edges.index)
    
    graph = ox.graph_from_gdfs(nodes, edges)
    return graph