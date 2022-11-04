import numpy as np
import tensorflow as tf

class Network:
    def __init__(self, input_size, layer_sizes=[128, 64, 1], lr=1e-3, rescaling_factor=1./1, model=False):
        if model!=False:
            self.input_size = input_size
            self.layer_sizes = layer_sizes
            self.lr = lr
            self.rescaling_factor = rescaling_factor
            self.model=model
        else:
            self.input_size = input_size
            self.layer_sizes = layer_sizes
            self.lr = lr
            self.rescaling_factor = rescaling_factor

            self.initializer = tf.keras.initializers.LecunNormal(seed=1)
            self.model = tf.keras.Sequential()
            self.model.add(tf.keras.layers.Rescaling(self.rescaling_factor))
            self.model.add(tf.keras.layers.Input(shape=(self.input_size,)))
            self.model.add(tf.keras.layers.Normalization(axis=-1))
            self.model.add(tf.keras.layers.Dense(
                self.layer_sizes[0], 
                activation='swish', 
                kernel_initializer=self.initializer, 
                name='1'))
            self.model.add(tf.keras.layers.Dense(
                self.layer_sizes[1], 
                activation='swish', 
                kernel_initializer=self.initializer,
                name='2'))
            self.model.add(tf.keras.layers.Dense(
                self.layer_sizes[2], 
                activation='sigmoid', 
                kernel_initializer=self.initializer,
                name='3'))
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=tf.keras.metrics.Accuracy(),
                )

    def set_weight_and_bias(self, arrays):
        self.model.set_weights(arrays)
        
    def predict(self, input):
        return self.model.predict(input)

    def get_layer_sizes(self):
        return self.layer_sizes

    def get_weight_and_bias(self):
        return self.model.get_weights()

    def get_model(self):
        return self.model

                