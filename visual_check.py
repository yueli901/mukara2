import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import tensorflow as tf
import matplotlib.pyplot as plt
import contextily as ctx
from tqdm import tqdm
from shapely.geometry import shape
from config import PATH, DATA, TRAINING
from model.Mukara import Mukara
from model.dataloader import get_sensor_coordinates, get_polygon, load_gt
import model.utils as utils

# Load sensor metadata
sensor_metadata = pd.read_csv(os.path.join(PATH['data'], 'Sites.csv'), dtype={'Id': str})

# Load GT and scaler
id_to_gt, scaler = load_gt()

# Load Mukara model
model = Mukara()

# Call the model with a dummy input to initialize weights
model("2")

# Load pre-trained weights
weights_path = os.path.join(PATH["param"], "2.3-1epoch.h5")
model.load_weights(weights_path)
print(f"Model weights loaded from {weights_path}")

# Create output directory
output_dir = os.path.join(PATH["evaluate"], "visual_check")
os.makedirs(output_dir, exist_ok=True)

# Function to compute GEH
def compute_geh(gt, pred):
    return np.sqrt(2 * (gt - pred) ** 2 / (gt + pred + 1e-6))

# Open CSV file to write predictions incrementally
csv_path = os.path.join(PATH["evaluate"], "predictions.csv")
with open(csv_path, "w") as f:
    f.write("sensor_id,predicted_value\n")

# Process sensors one at a time
for sensor_id in tqdm(id_to_gt.keys(), desc="Processing sensors"):
    try:
        # Get sensor metadata
        sensor_info = sensor_metadata[sensor_metadata["Id"] == sensor_id]
        sensor_name = sensor_info["Name"].values[0] if not sensor_info.empty else "Unknown"

        # Get sensor coordinates
        sensor_coords = get_sensor_coordinates(sensor_id).numpy()

        # Predict for this sensor
        pred_norm = model(sensor_id).numpy().flatten()[0]
        pred_original = scaler.inverse_transform(np.array([pred_norm]))[0]

        # Load ground truth
        gt_original = scaler.inverse_transform(np.array([id_to_gt[sensor_id]]))[0]
        geh = compute_geh(gt_original, pred_original)

        # Append prediction to CSV
        with open(csv_path, "a") as f:
            f.write(f"{sensor_id},{pred_original:.2f}\n")

        # Load the largest O and D isochrones
        o_polygon = get_polygon(sensor_id, 'o')
        d_polygon = get_polygon(sensor_id, 'd')

        # Compute bounding box for visualization
        minx, miny, maxx, maxy = o_polygon.union(d_polygon).bounds

        # Convert isochrones to GeoDataFrame for plotting
        gdf = gpd.GeoDataFrame(geometry=[o_polygon, d_polygon], crs="EPSG:27700")

        # Plot
        fig, ax = plt.subplots(figsize=(8, 8))
        gdf.plot(ax=ax, edgecolor=['blue', 'green'], facecolor='none', linewidth=2, linestyle='-')

        # Plot sensor location
        ax.scatter(sensor_coords[0], sensor_coords[1], color='red', s=100, label="Sensor")

        # Add basemap
        ctx.add_basemap(ax, crs=gdf.crs, source=ctx.providers.CartoDB.Positron)

        # Formatting
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_title(f"Sensor {sensor_id}, {sensor_name}\nGT: {gt_original:.2f}, Pred: {pred_original:.2f}, GEH: {geh:.2f}", fontsize=12)
        ax.legend()

        # Save figure
        save_path = os.path.join(output_dir, f"{sensor_id}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"Error processing sensor {sensor_id}: {e}")

print(f"All predictions and visualizations saved. CSV stored at {csv_path}")