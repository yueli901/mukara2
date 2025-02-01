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
from functools import partial
from shapely.geometry import shape
from shapely.ops import transform

from model.utils import Scaler
from config import PATH, DATA, TRAINING

np.random.seed(TRAINING['seed'])
tf.random.set_seed(TRAINING['seed'])
random.seed(TRAINING['seed'])


def load_grid_features():
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

def load_grid_cells():
    """
    Load the spatial grid cells and ensure they are in EPSG:27700.

    Returns:
        geopandas.GeoDataFrame: Grid cells in EPSG:27700.
    """
    grid_cells = gpd.read_file(os.path.join(PATH['grid']))
    
    if grid_cells.crs is None:
        raise ValueError("Grid cells shapefile does not have a CRS defined.")
    
    if grid_cells.crs.to_epsg() != 27700:
        raise ValueError("The CRS of the shapefile is not EPSG:27700.")
    
    return grid_cells

def load_gt():
    """
    Load the ground truth (traffic volume).
    Returns: 
        a dictionary {sensor_id: traffic_volume,...}. z-score normalized.
    """
    with h5py.File(PATH["ground_truth"], 'r') as f:  
        data = f['summary'][:] # an array with shape (n, 3). axis 1 are ids, traffic volume, number of days available for mean calculation. 

    traffic_volume = data[:,1]
    ids = data[:,0].astype(int).astype(str)

    # check isochrone data availability
    available_ids_in_files = set()
    all_files = [filename for filename in os.listdir(PATH["isochrones"]) if '2024' in filename]
    for filename in all_files:
        parts = filename.split('_')
        available_ids_in_files.add(parts[1])

    valid_ids = set(ids).intersection(available_ids_in_files)

    # Filter the ids and traffic_volume arrays based on the valid IDs
    mask = np.isin(ids, list(valid_ids))
    available_ids = ids[mask]
    available_traffic_volume = traffic_volume[mask]

    # Print the number of available IDs
    print(f"Number of valid IDs: {len(available_ids)}")

    scaler = Scaler(available_traffic_volume)
    traffic_volume_normalized = scaler.transform(available_traffic_volume)

    id_to_gt = {sensor_id: traffic_volume_normalized[idx] for idx, sensor_id in enumerate(available_ids)}

    return id_to_gt, scaler


def get_sensor_coordinates(sensor_id):
    """
    Returns the sensor coordinates in EPSG:27700 given a sensor ID.

    Args:
        sensor_id (str): Unique sensor identifier.

    Returns:
        tuple: Transformed (x, y) coordinates in EPSG:27700.
    """
    # Load sensor data
    sensors = pd.read_csv(os.path.join(PATH['data'], 'Sites.csv'))
    sensor_data = sensors[sensors['Id'] == sensor_id]
    
    if sensor_data.empty:
        raise ValueError(f"Sensor ID {sensor_id} not found in dataset.")
    
    lon = sensor_data['Longitude'].values[0]
    lat = sensor_data['Latitude'].values[0]
    
    # Define transformation from WGS84 (EPSG:4326) to British National Grid (EPSG:27700)
    transformer = pyproj.Transformer.from_crs("epsg:4326", "epsg:27700", always_xy=True)
    x, y = transformer.transform(lon, lat)
    
    return (x, y)


def get_polygon(sensor_id, region_type):
    """
    Retrieve the largest isochrone polygon for a given sensor ID and region type (O or D),
    transform it to EPSG:27700, and return as a Shapely polygon.

    Args:
        sensor_id (str): Unique identifier of the sensor.
        region_type (str): 'o' for origin, 'd' for destination.

    Returns:
        shapely.geometry.Polygon: Transformed isochrone polygon in EPSG:27700.
    """
    if region_type == 'o':
        location_type = 'destination'
    elif region_type == 'd':
        location_type = 'start'
    else:
        raise ValueError("Invalid region_type. Use 'o' for origin or 'd' for destination.")
    
    # Construct file paths for isochrones at different time intervals
    relevant_files = [
        os.path.join(PATH["isochrones"], f'{sensor_id}_{i}_{location_type}.json')
        for i in DATA['isochrone_intervals']
    ]
    
    # Load the largest isochrone polygon (last file in the sorted list)
    largest_isochrone_file = relevant_files[-1]
    if not os.path.exists(largest_isochrone_file):
        raise FileNotFoundError(f"Isochrone file not found: {largest_isochrone_file}")
    
    with open(largest_isochrone_file, 'r') as f:
        data = json.load(f)
        geojson_polygon = {
            "type": "Polygon",
            "coordinates": data['features'][0]['geometry']['coordinates']
        }
    
    isochrone_polygon = shape(geojson_polygon)
    
    # Define transformation from WGS84 (EPSG:4326) to British National Grid (EPSG:27700)
    project = partial(
        pyproj.transform,
        pyproj.Proj(init='epsg:4326'),  # Source projection (WGS84)
        pyproj.Proj(init='epsg:27700')  # Target projection (British National Grid)
    )
    
    transformed_polygon = transform(project, isochrone_polygon)
    
    return transformed_polygon