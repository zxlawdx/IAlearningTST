# IAlearningTST
#I'm Learning about machine learning and neural networks (Estou aprendendo sobre Aprendizagem de Maquina e redes neurais)
# that's representation it's form how i'll can implement -->

# Importing the library

import tensorflow as tf
import numpy as np


#Ex Data

celsius np.array([-40, 10, 6, 8, 15, 22, 38), dtype-float)
fahrenheit np.array([-40, 14, 32, 46, 59, 72, 100], dtype-float)

#implement layers

capa tf.keras.layers.Dense (units-1, input_shape=[1])

#creating model

modelo tf.keras. Sequential ([capa])

#Compilando o modelo

modelo.compile(
optimizer-tf.keras.optimizers.Adam (0.1),
loss-mean_squared_error)

#initializing training

print("Começando treinamento...")
print("Treinando o modelo")
historial modelo.fit(celsius, fahrenheit, epochs-1000, verbose-False)
print("Modelo definido!")
