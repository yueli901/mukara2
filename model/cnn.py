import tensorflow as tf
from tensorflow.keras import layers
from config import MODEL, TRAINING, PATH
import numpy as np
import os

np.random.seed(TRAINING['seed'])
tf.random.set_seed(TRAINING['seed'])

# import datetime
# current_time = datetime.datetime.now()
# formatted_time = current_time.strftime('%Y%m%d-%H%M%S')

class MLP(tf.keras.Sequential):
    """ Multilayer perceptron. """
    def __init__(self, hiddens, act_type, out_act, weight_initializer=None, **kwargs):
        """
        hiddens: list
            The list of hidden units of each dense layer.
        act_type: str
            The activation function after each dense layer.
        out_act: bool
            Whether to apply activation function after the last dense layer.
        """
        super(MLP, self).__init__(**kwargs)
        for i, h in enumerate(hiddens):
            activation = None if i == len(hiddens) - 1 and not out_act else act_type
            self.add(tf.keras.layers.Dense(
                h, activation=activation, kernel_initializer=weight_initializer, 
                name=f'dense_{i}'  # Ensure unique name for each layer
            ))


class CNN(tf.keras.Model):
    def __init__(self, 
                 depth: int, 
                 channels: list, 
                 kernel_size: int, 
                 strides: int, 
                 pool_size: int,
                 pool_strides: int,
                 output_dense: int,
                 weight_initializer=None, 
                 name_prefix = str,
                 **kwargs):
        """
        For generating node embeddings from ROI.
        """
        super(CNN, self).__init__(**kwargs)
        
        # Create layers dynamically based on the depth and channels
        self.conv_layers = []
        for i in range(depth):
            self.conv_layers.append(
                layers.Conv2D(
                    filters=channels[i],
                    kernel_size=kernel_size,
                    strides=strides,
                    padding="same",
                    activation="relu",
                    kernel_initializer=weight_initializer,
                    name=name_prefix + f"_conv_{i}"
                )
            )
            self.conv_layers.append(
                layers.MaxPooling2D(
                    pool_size=pool_size,
                    strides=pool_strides,
                    padding="same",
                    name=name_prefix + f"_maxpool_{i}"
                )
            )
        
        self.flatten = layers.Flatten(name="flatten")
        
        # Add a dense layer to convert to 1D node embeddings
        self.dense_output = layers.Dense(
            units=output_dense,
            activation="relu",
            kernel_initializer=weight_initializer,
            name=name_prefix + "_dense_output"
        )
    
    def call(self, inputs):
        """
        Forward pass of the CNN head.
        """
        x = inputs
        for layer in self.conv_layers:
            x = layer(x)
        x = self.flatten(x)
        x = self.dense_output(x)
        return x


class Model(tf.keras.Model):
    """
    Input: grid feature (row=653, col=574, c)
    Output: average daily traffic volume (batch_size,)
    """

    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)

        # CNN for origin grid feature extraction
        self.cnn_o = CNN(
            depth=MODEL['depth_cnn'],
            channels=MODEL['channels'],
            kernel_size=MODEL['kernel_size'],
            strides=MODEL['strides'],
            pool_size=MODEL['pool_size'],
            pool_strides=MODEL['pool_strides'],
            output_dense=MODEL['output_dense'],
            weight_initializer='he_normal',
            name_prefix = 'orig'
        )

        # CNN for destination grid feature extraction
        self.cnn_d = CNN(
            depth=MODEL['depth_cnn'],
            channels=MODEL['channels'],
            kernel_size=MODEL['kernel_size'],
            strides=MODEL['strides'],
            pool_size=MODEL['pool_size'],
            pool_strides=MODEL['pool_strides'],
            output_dense=MODEL['output_dense'],
            weight_initializer='he_normal',
            name_prefix = 'dest'
        )

        # MLP for roi_o/d embedding concat and final output
        self.mlp = MLP(MODEL['hiddens'] + [1], act_type=MODEL['activation'], out_act=False)

    def call(self, ids):
        roi_origin_list = []
        roi_destination_list = []
        for sensor_id in ids:
            origin_path = os.path.join(PATH['roi'], f"{sensor_id}_origin.npy")
            roi_origin_list.append(np.load(origin_path))
            destination_path = os.path.join(PATH['roi'], f"{sensor_id}_destination.npy")
            roi_destination_list.append(np.load(destination_path))

        roi_origin = tf.stack(roi_origin_list, axis=0)  # Shape: (batch, row, col, c)
        roi_destination = tf.stack(roi_destination_list, axis=0)  # Shape: (batch, row, col, c)

        cnn_o_output = self.cnn_o(roi_origin)  # Shape: (batch, d)
        cnn_d_output = self.cnn_d(roi_destination)  # Shape: (batch, d)

        concatenated_output = tf.concat([cnn_o_output, cnn_d_output], axis=-1)  # Shape: (batch, 2d)

        traffic_volume = self.mlp(concatenated_output)  # Final scalar output

        return traffic_volume