IAlearningTST
Overview / Visão Geral

I'm learning about machine learning and neural networks, specifically focusing on how to implement them using TensorFlow.
Estou aprendendo sobre Aprendizagem de Máquina e redes neurais, com foco em como implementá-las usando TensorFlow.
Project Description / Descrição do Projeto

This project demonstrates a simple neural network model to convert Celsius temperatures to Fahrenheit.
Este projeto demonstra um modelo simples de rede neural para converter temperaturas Celsius em Fahrenheit.
Libraries Used / Bibliotecas Utilizadas

python

import tensorflow as tf
import numpy as np

Example Data / Dados de Exemplo

The training data consists of Celsius temperatures and their corresponding Fahrenheit values:
Os dados de treinamento consistem em temperaturas Celsius e seus valores correspondentes em Fahrenheit:

python

celsius = np.array([-40, 10, 6, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

Model Architecture / Arquitetura do Modelo

We implement a single-layer neural network using the following layer:
Implementamos uma rede neural de camada única usando a seguinte camada:

python

layer = tf.keras.layers.Dense(units=1, input_shape=[1])

Creating the Model / Criando o Modelo

We create a sequential model with the defined layer:
Criamos um modelo sequencial com a camada definida:

python

model = tf.keras.Sequential([layer])

Compiling the Model / Compilando o Modelo

The model is compiled with the Adam optimizer and mean squared error as the loss function:
O modelo é compilado com o otimizador Adam e o erro quadrático médio como função de perda:

python

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

Training the Model / Treinando o Modelo

We begin the training process with the following code:
Iniciamos o processo de treinamento com o seguinte código:

python

print("Começando o treinamento...")
print("Treinando o modelo")
history = model.fit(celsius, fahrenheit, epochs=1000, verbose=False)
print("Modelo definido!")

Interactive Exploration / Exploração Interativa

To test the model, you can use the following code snippet to make predictions:
Para testar o modelo, você pode usar o seguinte trecho de código para fazer previsões:

python

# Testing the model with new Celsius values
new_celsius = np.array([0, 25, 100], dtype=float)
predictions = model.predict(new_celsius)
for c, f in zip(new_celsius, predictions):
    print(f"{c}°C = {f[0]:.2f}°F")

Conclusion / Conclusão

After training, the model should be able to predict Fahrenheit temperatures based on Celsius inputs. Feel free to read through my code and explore!
Após o treinamento, o modelo deve ser capaz de prever temperaturas em Fahrenheit com base em entradas em Celsius. Fique à vontade para ler meu código e explorar!
