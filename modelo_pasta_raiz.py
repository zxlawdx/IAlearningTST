from pathlib import Path
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from pprint import pprint

# Caminho da pasta
pasta_modelos1 = Path(r'C:/Faculdade/testeredes/learned/modelo_raiz/')

# Arquivos esperados
arquivos_modelos1 = {
    'historial.pkl': pasta_modelos1 / 'historial_raiz.pkl',
    'modelo.h5': pasta_modelos1 / 'modelo_raiz.h5',
    'pesos_modelo_raiz.weights.h5': pasta_modelos1 / 'pesos_modelo_raiz.weights.h5'
}

def carregar_dados_pasta(pasta, arquivos):
    """
    Função para carregar e exibir dados dos arquivos em uma pasta especificada.
    
    Args:
        pasta (Path): Caminho da pasta onde os arquivos estão localizados.
        arquivos (dict): Dicionário onde as chaves são os nomes dos arquivos e os valores são os caminhos completos.
    """
    print(f"\nVerificando arquivos na pasta: {pasta}")
    
    for nome_arquivo, caminho in arquivos.items():
        print(f"Verificando caminho {nome_arquivo}: {caminho}")
        if caminho.is_file():
            try:
                if nome_arquivo.endswith('.pkl'):
                    with open(caminho, 'rb') as file:
                        dados = pickle.load(file)
                    print(f"Conteúdo do arquivo {nome_arquivo}:")
                    pprint(dados)
                elif nome_arquivo.endswith('.h5'):
                    if 'modelo' in nome_arquivo:
                        modelo = load_model(caminho)
                        print(f"\nResumo do modelo {nome_arquivo}:")
                        modelo.summary()
                    else:
                        # Defina a arquitetura do modelo de acordo com a que foi usada para treinar e salvar os pesos
                        modelo_pesos = tf.keras.Sequential([
                            tf.keras.layers.Dense(256, input_dim=1, activation='relu'),
                            tf.keras.layers.Dense(256, activation='relu'),
                            tf.keras.layers.Dense(128, activation='relu'),
                            tf.keras.layers.Dense(64, activation='relu'),
                            tf.keras.layers.Dense(32, activation='relu'),
                            tf.keras.layers.Dense(1)
                        ])
                        modelo_pesos.compile(optimizer='adam', loss='mean_squared_error')
                        modelo_pesos.load_weights(caminho)
                        print(f"\nPesos carregados com sucesso do arquivo {nome_arquivo}")
                        print("Pesos do modelo:")
                        print(modelo_pesos.get_weights())
            except Exception as e:
                print(f"Erro ao carregar o arquivo {nome_arquivo}: {e}")
        else:
            print(f"Arquivo {nome_arquivo} não encontrado.")

# Carregar dados da primeira pasta
carregar_dados_pasta(pasta_modelos1, arquivos_modelos1)
