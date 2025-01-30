import tensorflow as tf
from config import TRAINING
import random

tf.random.set_seed(TRAINING['seed'])
random.seed(TRAINING['seed'])

class Scaler:
	def __init__(self, data):
		# Ignore NaNs in mean and std calculations
		valid_data = tf.where(tf.math.is_nan(data), tf.zeros_like(data), data)
		count = tf.reduce_sum(tf.cast(~tf.math.is_nan(data), tf.float32))

		self.mean = tf.reduce_sum(valid_data) / count
		self.std = tf.sqrt(tf.reduce_sum(tf.square(valid_data - self.mean)) / count)

		tf.print("Mean:", self.mean, "Standard Deviation:", self.std)

	def transform(self, data):
		# Standardize data while ignoring NaNs
		return tf.where(tf.math.is_nan(data), data, (data - self.mean) / (self.std + 1e-8))
    
	def inverse_transform(self, data):
		# Reverse standardization while ignoring NaNs
		return tf.where(tf.math.is_nan(data), data, data * self.std + self.mean)


def MSE_z(label_z, pred_z, scaler):
	return tf.reduce_mean(tf.square(label_z - pred_z))

def MAE(label_z, pred_z, scaler):
	pred = scaler.inverse_transform(pred_z)
	label = scaler.inverse_transform(label_z)
	return tf.reduce_mean(tf.abs(label - pred))

def MSE(label_z, pred_z, scaler):
	pred = scaler.inverse_transform(pred_z)
	label = scaler.inverse_transform(label_z)
	return tf.reduce_mean(tf.square(label - pred))

def MGEH(label_z, pred_z, scaler):
	pred = scaler.inverse_transform(pred_z)
	label = scaler.inverse_transform(label_z)
	gehs = tf.sqrt(2 * tf.square(pred - label) / (tf.abs(pred) + label + 1e-8)) # pred for small volume sensors can be negative
	return tf.reduce_mean(gehs)


def train_test_sampler(sensor_ids, true_probability):

    # Shuffle the sensor IDs to ensure randomness
    random.shuffle(sensor_ids)
    
    # Split the sensor IDs based on the true_probability
    train_ids = [sensor_id for sensor_id in sensor_ids if random.random() < true_probability]
    test_ids = [sensor_id for sensor_id in sensor_ids if sensor_id not in train_ids]
    
    return train_ids, test_ids
