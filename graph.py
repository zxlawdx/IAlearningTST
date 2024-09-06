import matplotlib.pyplot as plt
import os
import pickle

# Diretório para armazenar o gráfico
GRAPH_DIR = 'c:/Faculdade/redeneural/IAlearningTST/learned/graphs'

def save_training_graph(historial_path):
    # Carregar o histórico
    with open(historial_path, 'rb') as file:
        historial = pickle.load(file)
    
    # Verifica e cria o diretório para armazenar o gráfico
    if not os.path.exists(GRAPH_DIR):
        os.makedirs(GRAPH_DIR)
    
    # Plota o gráfico
    plt.figure()
    plt.plot(historial.history['loss'])
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.title('Gráfico de Perda durante o Treinamento')
    
    # Salva o gráfico como uma imagem PNG
    graph_path = os.path.join(GRAPH_DIR, 'training_graph.png')
    plt.savefig(graph_path)
    plt.close()
    print(f'Gráfico salvo em {graph_path}')

if __name__ == "__main__":
    historial_path = 'c:/Faculdade/redeneural/IAlearningTST/learned/historial.pkl'
    save_training_graph(historial_path)
