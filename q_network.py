import tensorflow as tf
import numpy as np 

class QNetwork:

    def __init__(self,input_shape, hidden_units, output_size, learning_rate=0.01):
        self.input_shape = input_shape
        hidden_units_1, hidden_units_2, hidden_units_3 = hidden_units
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=hidden_units_1, input_dim=input_shape, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=hidden_units_2, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=hidden_units_3, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=output_size, activation=tf.keras.activations.linear)
        ])

        self.model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate), loss='mse',metrics=['accuracy'])
        
        
    def predict(self, state, batch_size=1):
        return self.model.predict(state, batch_size)
    
    def train(self, states, action_values, batch_size):
        self.model.fit(states, action_values, batch_size=batch_size, verbose=0, epochs=1)