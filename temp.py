import tensorflow as tf
import numpy as np
import pickle

# Dados de treinamento
celsius = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 330, 302, 320, 390, 470], dtype=float)
fahrenheit = np.array([32, 68, 104, 140, 176, 212, 248, 284, 320, 356, 392, 428, 464, 500, 536, 572, 626, 575.6, 608, 734, 878], dtype=float)

# Definindo o modelo
capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])

# Compilando o modelo
modelo.compile(optimizer=tf.keras.optimizers.Adam(2.12), loss='mean_squared_error')

# Treinando o modelo
print("Iniciando treinamento...")
historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)
print("Modelo treinado")

# Fazer previsões com o modelo treinado
predicao = modelo.predict(np.array([100.0]))
print("Previsões feitas:", predicao)

# Verificar e imprimir as variáveis internas do modelo
print("Variáveis internas do modelo (pesos):")
print(capa.get_weights())

# Salvando o modelo completo (arquitetura + pesos)
modelo.save('c:/Faculdade/redeneural/IAlearningTST/learned/modelo_temp/modelo.h5')
print("Modelo salvo no formato .h5")

# Salvando apenas os pesos do modelo
modelo.save_weights('c:/Faculdade/redeneural/IAlearningTST/learned/modelo_temp/pesos_modelo.weights.h5')
print("Pesos salvos no formato .h5")

# Salvando o histórico de treinamento em um arquivo .pkl
caminho_pkl = r'c:/Faculdade/redeneural/IAlearningTST/learned/modelo_temp/historial.pkl'
try:
    with open(caminho_pkl, 'wb') as file:
        pickle.dump(historial.history, file)
    print("Histórico de treinamento salvo no arquivo .pkl")
except Exception as e:
    print(f"Erro ao salvar o arquivo .pkl: {e}")
