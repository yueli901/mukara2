import tensorflow as tf
import h5py
import os
import numpy as np
import random
from tqdm import tqdm
import pandas as pd
import geopandas as gpd
import json
from shapely.geometry import shape
from pyproj import Transformer
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds
from shapely.ops import transform

from model.utils import Scaler
from config import PATH, DATA, TRAINING

np.random.seed(TRAINING['seed'])
tf.random.set_seed(TRAINING['seed'])
random.seed(TRAINING['seed'])



def get_static_features():
    """
    population_and_employment shape (year=8, height, width, c=2)
    landuse_and_poi shape (height, width, 12)
    z-score normalize
    return grid static feature (year=8, height, width, c=14)
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
    static_features = tf.concat([pe, lp], axis=-1)
    static_features = static_features[-1,...] # year 2022
    print(f"Grid features shape {static_features.shape}")

    return static_features


def get_gt():
    """
    Dataset shape (n, 3). 
    3 for (ids, traffic volume, number of days available for mean calculation). 
    Output z-score normalize
    """
    with h5py.File(PATH["ground_truth"], 'r') as f:  
        data = f['summary'][:]

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

def get_sensor_location(sensor_id):
    sensors = pd.read_csv(os.path.join(PATH['data'], 'Sites.csv'))
    lon = sensors['Longitude'][sensors['Id']==sensor_id]
    lat = sensors['Latitude'][sensors['Id']==sensor_id]
    return (lon, lat)

def process_sensors(id_to_gt, grid_features):
    # get grid origin
    grid_cells = gpd.read_file(PATH['grid'])
    assert grid_cells.crs.to_epsg() == 27700, "The CRS of the shapefile is not EPSG:27700."
    grid_bounds = grid_cells.total_bounds  # [minx, miny, maxx, maxy]
    grid_origin = (grid_bounds[0], grid_bounds[1]) # Lower-left corner (minx, miny) in EPSG:27700

    # max isochrone interval for size calculation
    ids = list(id_to_gt.keys())
    roi_edge_length = compute_roi_edge_length(ids, size=DATA['isochrone_intervals'][-1]) if len(DATA['isochrone_intervals'])>0 else 100
    # loop for each sensor
    os.makedirs(PATH['roi'], exist_ok=True)
    finished = set([f.split('_')[0] for f in os.listdir(PATH['roi'])])
    ids = list(set(ids) - finished)
    for sensor_id in tqdm(ids, desc="Processing sensor ROIs: "):
        process_sensor(sensor_id, roi_edge_length, grid_features, grid_origin)

def compute_roi_edge_length(ids, size):
    """
    Go through all isochrones with the size, and determine the largest window size in pixels.
    """
    cache = os.path.join(PATH["isochrones"], "largest_sides_cache.json")
    record = {}
    if os.path.exists(cache):
        with open(cache, "r") as json_file:
            record = json.load(json_file)

    if str(size) in record: 
        edge_length = record[str(size)]
        print(f"Use calculated edge_length {edge_length} for size {size}.")
    else:
        grid_resolution = DATA['grid_resolution']
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)

        json_files = os.listdir(PATH["isochrones"])
        filtered_files = [file for file in json_files if f'_{size}_' in file and file.split('_')[1] in ids]
        largest_side = 0
        for file in tqdm(filtered_files, desc="Computing largest bbox: ", total=len(filtered_files)):
            with open(os.path.join(PATH["isochrones"], file), 'r') as f:
                data = json.load(f)
                bbox = data.get('bbox')
                minx, miny = transformer.transform(bbox[0], bbox[1])
                maxx, maxy = transformer.transform(bbox[2], bbox[3])
                side = max(maxx - minx, maxy - miny)
                largest_side = max(largest_side, side)
        edge_length = int(largest_side // grid_resolution) + 1  # Ensure coverage
        record[str(size)] = edge_length
        print(f"Edge_length {edge_length} for size {size} was determined and cached.")
        with open(os.path.join(PATH["isochrones"], "largest_sides_cache.json"), "w") as json_file:
            json.dump(record, json_file, indent=4)

    return edge_length

# Function to process each sensor
def process_sensor(sensor_id, roi_edge_length, grid_features, grid_origin):
    """
    Produce the origin and destination windows (edge, edge, c) for the given sensor.
    Cache output tensor to roi_cache.
    """
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)

    for location_type in ["start", "destination"]:
        od = 'origin' if location_type == 'destination' else 'destination' # sensor location as destination means isochrone area is origin
        file_name = os.path.join(PATH['roi'],f"{sensor_id}_{od}.npy")
        if os.path.exists(file_name):
            return
    
        # Find all JSON files for the current sensor and location type
        relevant_files = []
        for i in DATA['isochrone_intervals']:
            relevant_files.append(os.path.join(PATH["isochrones"], f'2024_{sensor_id}_{i}_{location_type}.json'))
        
        # Load the largest isochrone to determine the bbox position
        if relevant_files:            
            with open(relevant_files[-1], 'r') as f:
                data = json.load(f)
                bbox = data['bbox']
                minx, miny = transformer.transform(bbox[0], bbox[1])
                maxx, maxy = transformer.transform(bbox[2], bbox[3])
                bbox_center = ((minx + maxx) / 2, (miny + maxy) / 2)  # in EPSG:27700
        else:
            sensor_coordinates = get_sensor_location(sensor_id)
            bbox_center = transformer.transform(*sensor_coordinates)

        grid_window, window_bounds = extract_grid_window(grid_features, bbox_center, roi_edge_length, grid_origin)
        
        # Process each isochrone and create channels
        isochrone_channels = []
        for file in relevant_files:
            with open(file, 'r') as f:
                data = json.load(f)
                geojson_polygon = {
                    "type": "Polygon",
                    "coordinates": data['features'][0]['geometry']['coordinates']
                }
                isochrone_polygon = shape(geojson_polygon)

                # Rasterize the isochrone polygon onto the extracted grid window
                isochrone_channel = create_isochrone_channels(grid_window.shape, window_bounds, isochrone_polygon)
                isochrone_channels.append(isochrone_channel)
        
        # Stack the final tensor
        tensor = np.dstack([grid_window] + isochrone_channels)
        np.save(file_name, tensor)


# Function to extract the grid window
def extract_grid_window(grid, bbox_center, edge_length, grid_origin):
    """
    Crop a fixed-size window from the entire grid_features based on pixel indices.
    If the window goes out of bounds, pad the grid with zeros.
    """
    grid_resolution = DATA['grid_resolution']
    center_x, center_y = bbox_center

    # Convert bbox center from EPSG:27700 to grid pixel indices
    center_col = int((center_x - grid_origin[0]) // grid_resolution)
    center_row = int((center_y - grid_origin[1]) // grid_resolution)

    # Compute the half window size
    half_window = edge_length // 2

    # Adjust start and end indices to ensure correct range
    if edge_length % 2 == 0:  # Even edge_length
        start_col = center_col - half_window
        end_col = center_col + half_window
        start_row = center_row - half_window
        end_row = center_row + half_window
    else:  # Odd edge_length
        start_col = center_col - half_window
        end_col = center_col + half_window + 1
        start_row = center_row - half_window
        end_row = center_row + half_window + 1

    # Handle out-of-bounds indices by padding the grid
    pad_top = max(0, -start_row)
    pad_left = max(0, -start_col)
    pad_bottom = max(0, end_row - grid.shape[0])
    pad_right = max(0, end_col - grid.shape[1])

    # Pad the grid with zeros if necessary
    if pad_top > 0 or pad_left > 0 or pad_bottom > 0 or pad_right > 0:
        # Determine padding for each dimension
        if grid.ndim == 2:  # 2D grid (height, width)
            pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))
        elif grid.ndim == 3:  # 3D grid (height, width, channels)
            pad_width = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
        else:
            raise ValueError(f"Unsupported grid dimensions: {grid.ndim}")
        
        grid = np.pad(grid, pad_width, mode='constant', constant_values=0)

    # Recompute indices after padding
    start_row += pad_top
    start_col += pad_left
    end_row += pad_top
    end_col += pad_left

    # Extract the fixed-size window
    window = grid[start_row:end_row, start_col:end_col]

    # Ensure the window size is exactly edge_length x edge_length
    if window.shape[:2] != (edge_length, edge_length):
        raise ValueError(f"Extracted window size {window.shape[:2]} does not match expected size ({edge_length}, {edge_length}).")

    # Calculate window bounds in spatial coordinates
    window_bounds = (
        grid_origin[0] + start_col * grid_resolution,
        grid_origin[1] + start_row * grid_resolution,
        grid_origin[0] + end_col * grid_resolution,
        grid_origin[1] + end_row * grid_resolution
    )

    return window, window_bounds

# Function to create an isochrone channel
def create_isochrone_channels(grid_window_shape, window_bounds, isochrone_polygon):
    """
    Rasterize an isochrone polygon onto the extracted grid window.

    Parameters:
    - grid_window_shape: tuple (height, width) of the fixed grid window.
    - window_bounds: tuple (minx, miny, maxx, maxy) defining the spatial extent of the grid window.
    - isochrone_polygon: Shapely Polygon object representing the isochrone in its original CRS.
    - transformer: PyProj Transformer to convert isochrone_polygon into EPSG:27700.

    Returns:
    - A binary mask of the same shape as the grid window, where pixels inside the polygon are 1, and outside are 0.
    """
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)

    # Transform the polygon to EPSG:27700
    isochrone_polygon_27700 = transform(transformer.transform, isochrone_polygon)

    # Create a transform for the grid window
    trans = from_bounds(*window_bounds, grid_window_shape[1], grid_window_shape[0])

    # Rasterize the transformed polygon
    mask = geometry_mask([isochrone_polygon_27700], out_shape=grid_window_shape[:2], transform=trans, invert=True)
    return mask.astype(int)

