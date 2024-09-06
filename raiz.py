import os
import numpy as np
import pickle
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

# Gerar dados de treinamento
def generate_data(n_samples=5000):
    X = np.random.rand(n_samples) * 100  # Aumentar número de amostras
    y = np.sqrt(X)  # Raiz quadrada dos números
    return X, y

# Preparar dados com normalização
def prepare_data():
    X, y = generate_data()
    
    # Normalizar os dados
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    X_scaled = scaler_X.fit_transform(X.reshape(-1, 1))
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
    
    split_index = int(len(X) * 0.8)
    X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
    y_train, y_test = y_scaled[:split_index], y_scaled[split_index:]
    
    return X_train, X_test, y_train, y_test, scaler_X, scaler_y

# Definir e treinar o modelo
def create_and_train_model(X_train, y_train):
    modelo = Sequential([
        Dense(256, input_dim=1, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    modelo.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    # Treinamento do modelo
    history = modelo.fit(X_train, y_train, epochs=1000, batch_size=32, validation_split=0.2)
    
    # Criar diretório se não existir
    save_dir = r"C:\\Faculdade\\testeredes\\learned\\modelo_raiz"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Salvar o modelo completo
    modelo.save(os.path.join(save_dir, 'modelo.h5'))
    
    # Salvar o histórico de treinamento
    with open(os.path.join(save_dir, 'historial.pkl'), 'wb') as file:
        pickle.dump(history.history, file)
    
    # Salvar os pesos do modelo
    modelo.save_weights(os.path.join(save_dir, 'pesos_modelo.weights.h5'))
    
    return modelo

# Carregar o modelo
def load_trained_model():
    # Carregar o modelo completo
    save_dir = r"C:\\Faculdade\\testeredes\\learned\\modelo_raiz"
    modelo = load_model(os.path.join(save_dir, 'modelo.h5'))
    
    return modelo

# Função para prever a raiz quadrada usando o modelo
def raiz(entrada, modelo, scaler_X, scaler_y):
    entrada_array = np.array([entrada], dtype=float).reshape(-1, 1)
    
    # Normalizar a entrada
    entrada_normalizada = scaler_X.transform(entrada_array)
    
    resultado_normalizado = modelo.predict(entrada_normalizada)[0][0]
    
    # Desnormalizar o resultado
    resultado = scaler_y.inverse_transform(np.array([[resultado_normalizado]]))[0][0]
    return resultado

# Código principal para treinamento e uso
def main():
    # Preparar dados
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = prepare_data()
    
    # Criar e treinar o modelo
    modelo = create_and_train_model(X_train, y_train)
    
    # Avaliar o modelo (opcional)
    loss = modelo.evaluate(X_test, y_test)
    print(f'Perda no conjunto de teste: {loss}')
    
    # Carregar o modelo
    modelo = load_trained_model()
    
    # Entrada do usuário
    entrada = float(input("Digite um valor: "))
    
    # Usar a função raiz
    resultado = raiz(entrada, modelo, scaler_X, scaler_y)
    
    print(f"A raiz quadrada aproximada de {entrada} é {resultado:.4f}")

# Executar o código principal
if __name__ == "__main__":
    main()
