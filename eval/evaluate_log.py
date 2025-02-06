import os
import re
import pandas as pd
import numpy as np

def extract_metrics(log_path):
    """
    Reads the log file and extracts training and validation metrics.
    """
    # Initialize empty lists to store data
    epochs = []
    sensors = []
    train_total_loss = []
    train_geh = []
    train_compactness = []
    train_separation = []
    train_balance = []
    
    valid_total_loss = []
    valid_geh = []
    valid_compactness = []
    valid_separation = []
    valid_balance = []

    # Regular expressions to capture relevant data from logs
    epoch_sensor_pattern = re.compile(r'Epoch (\d+), Sensor (\d+)')
    train_pattern = re.compile(
        r'Train Loss: Total Loss: ([\d.]+), Traffic \(MGEH\): ([\d.]+), '
        r'Compactness: ([\d.]+), Separation: ([\d.]+), Balance: ([\d.]+)'
    )
    valid_pattern = re.compile(
        r'Valid Loss: Total Loss: ([\d.]+), Traffic \(MGEH\): ([\d.]+), '
        r'Compactness: ([\d.]+), Separation: ([\d.]+), Balance: ([\d.]+)'
    )

    # Read the log file
    with open(log_path, 'r') as f:
        for line in f:
            epoch_sensor_match = epoch_sensor_pattern.search(line)
            train_match = train_pattern.search(line)
            valid_match = valid_pattern.search(line)

            if train_match:
                epoch = int(epoch_sensor_match.group(1))
                sensor = int(epoch_sensor_match.group(2))
                epochs.append(epoch)
                sensors.append(sensor)

                train_total_loss.append(float(train_match.group(1)))
                train_geh.append(float(train_match.group(2)))
                train_compactness.append(float(train_match.group(3)))
                train_separation.append(float(train_match.group(4)))
                train_balance.append(float(train_match.group(5)))

            if valid_match:
                valid_total_loss.append(float(valid_match.group(1)))
                valid_geh.append(float(valid_match.group(2)))
                valid_compactness.append(float(valid_match.group(3)))
                valid_separation.append(float(valid_match.group(4)))
                valid_balance.append(float(valid_match.group(5)))

    # Create a DataFrame
    df = pd.DataFrame({
        'epoch': epochs,
        'sensor': sensors,
        'train_total_loss': train_total_loss,
        'train_geh': train_geh,
        'train_compactness': train_compactness,
        'train_separation': train_separation,
        'train_balance': train_balance,
        'valid_total_loss': valid_total_loss,
        'valid_geh': valid_geh,
        'valid_compactness': valid_compactness,
        'valid_separation': valid_separation,
        'valid_balance': valid_balance
    })


    return df