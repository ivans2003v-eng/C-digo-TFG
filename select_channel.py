import numpy as np
from sklearn.neighbors import KernelDensity
from pylab import *
import pandas as pd
import os 

def calculate_entropy(window_matrix, spectras):
    """
    Calcula la entropía espectral y temporal de cada ventana de señal.

    Parámetros
     - window_matrix : np.ndarray. Matriz de ventanas de señal, cada fila es una ventana.
     - spectras : list de np.ndarray. Lista de densidades espectrales normalizadas correspondientes a cada ventana.

    Devuelve
     - s_etps : list de float. Entropía espectral de cada ventana.
     - t_etps : list de float. Entropía temporal de cada ventana, estimada mediante Kernel Density Estimation (KDE).
    """
    s_etps = []
    t_etps = []
    for i, window in enumerate(window_matrix):
        # Spectral entropy
        spectrum = spectras[i]
        s_etps.append(-sum(spectrum*log2(spectrum)))
        
        # Temporal entropy
        x_grid = np.linspace(min(window), max(window), 100)
        kde_skl = KernelDensity(bandwidth=8)
        kde_skl.fit(window.reshape(-1, 1))
        log_pdf = kde_skl.score_samples(x_grid.reshape(-1, 1))
        kde = np.exp(log_pdf)
        nkde = kde/np.sum(kde)
        t_etps.append(-sum(nkde*log2(nkde+1e-10)))

    return s_etps, t_etps


def save_channel_rankings(patient, channel_scores, output_dir='.'):
    """
    Guarda el ranking de canales de un paciente según diferentes métricas.

    Parámetros
     - patient : str. Nombre o identificador del paciente.
     - channel_scores : pd.DataFrame. DataFrame donde las filas son canales y las columnas son métricas de evaluación.
     - output_dir : str, opcional. Directorio donde se guardará el archivo CSV (por defecto '.').
    """

    all_metrics = channel_scores.columns
    all_channels = channel_scores.index.tolist()
    
    ranked_data = {}
    
    for metric in all_metrics:
        # Ordenar canales de mayor a menor para cada métrica
        ordered = channel_scores[metric].sort_values(ascending=False).index.tolist()
        ranked_data[metric] = ordered
    
    # Convertimos el diccionario a DataFrame
    ranked_df = pd.DataFrame(ranked_data).T  # Transponer para que las métricas sean filas
    
    # Guardar el DataFrame en un solo archivo .h5
    filename = f"{output_dir}/{patient}_ranking_channels.csv"
    ranked_df.to_csv(filename, index=True)
    print(f"Ranking guardado en: {filename}")


def get_best_channels(channelsdf, nchannels=2):
    """
    Selecciona los mejores canales a partir de un DataFrame en el que cada fila contiene un ranking de canales 
    (ordenados de mejor a peor). Cada fila corresponde a una métrica de evaluación de rendimiento.

    Parámetros:
    - channelsdf: DataFrame donde cada fila es una lista ordenada de canales según su rendimiento.
    - nchannels: número de mejores canales a devolver.

    Devuelve:
    - Lista con los 'nchannels' mejores canales, según la puntuación agregada.
    """

    # Inicializamos el diccionario de puntuaciones
    channel_score = {}

    # Recorremos cada fila (una métrica) y asignamos puntajes
    for row in channelsdf.values:
        for position, channel in enumerate(row):
            channel_score[channel] = channel_score.get(channel, 0) + position

    # Ordenamos los canales por su puntuación acumulada (menor es mejor)
    best_channels = sorted(channel_score, key=channel_score.get)[:nchannels]

    return best_channels
