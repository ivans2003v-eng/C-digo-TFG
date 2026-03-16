import pandas as pd 
from sklearn.preprocessing import StandardScaler
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import LSTM, Dense, Dropout
#from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import os

def split_train_test(df, target, training_files, test_file):
    """
    Esta función divide un DataFrame que contiene características extraídas de señales EEG en conjuntos de entrenamiento 
    y prueba, filtrando por el nombre del archivo de origen. 
    
    Parámetros de entrada:
      - df (pd.DataFrame): DataFrame que contiene todas las ventanas de EEG con sus características
      - target (str): nombre de la columna que representa la etiqueta (seizure True o False)
      - training_files (list[str]): lista de nombres de archivo que se utilizarán como conjunto de entrenamiento
      - test_file (str): nombre de archivo que se utilizará como conjunto de prueba

    Salida:
      - x_train (pd.DataFrame): características del conjunto de entrenamiento
      - y_train (pd.Series): etiquetas correspondientes al conjunto de entrenamiento (target)
      - x_test (pd.DataFrame): características del conjunto de prueba.
      - y_test (pd.Series): etiquetas correspondientes al conjunto de prueba.
    """

    df_train = eval(get_expression(training_files, 'register'))     
    df_train = df_train.drop(['register'], axis=1)                  
    x_train, y_train = df_train[df_train.columns.difference([target])], df_train[target]   
    
    df_test = eval(get_expression(test_file, 'register'))        
    df_test = df_test.drop(['register'], axis=1)                    
    x_test, y_test = df_test[df_test.columns.difference([target])], df_test[target]   
    
    return x_train, y_train, x_test, y_test


def get_expression(my_list, feature):
    """
    Esta función filtra un DataFrame, seleccionando las filas donde la columna feature tenga un 
    valor contenido en la lista my_list.

    Parámetros de entrada:
      - my_list (list): una lista de valores que se desean buscar en la columna especificada del DataFrame.
      - feature (str): el nombre de la columna del DataFrame sobre la que se aplica el filtro.

    Salida:
      - expression (str): una cadena de texto que representa una condición booleana
    """
    expression = 'df['
    for item in my_list:
        expression +=f'(df[\'{feature}\'] == \'{item}\')|'
    return expression[:-1] + ']'


def non_seizure_files_division(seizure_files, bbdd_path):
    """
    Identifica los ficheros que no contienen crisis epilépticas y los divide en 2 subconjuntos: uno de entrenamiento
    de hasta 4 archivos y otro subconjunto con el resto de archivos para prueba. 

    Parámetros de entrada:
      - seizure_files (list[str]): lista de nombres de archivos .edf que contienen crisis epilépticas.
      - bbdd_path (str): ruta al directorio donde se encuentran los archivos .edf (la base de datos de señales EEG).
    
    Salida: 
      - patient_nseizure_train_files (list[str]): archivos sin crisis seleccionados para entrenamiento.
      - patient_non_seizure_test_files (list[str]): archivos sin crisis restantes usados para prueba.
    """

    files = [
        f for f in os.listdir(bbdd_path)
        if os.path.isfile(os.path.join(bbdd_path, f)) and f.endswith('.edf')]
    
    # Extraer los nombres base de los archivos con crisis (sin extensión)
    seizure_basenames = [os.path.splitext(f)[0] for f in seizure_files]

    # Filtrar los archivos sin crisis: su nombre base no debe estar en seizure_basenames
    non_seizure_files = [f for f in files if os.path.splitext(f)[0] not in seizure_basenames]

    patient_nseizure_train_files = non_seizure_files[-min(4, len(seizure_files)):]   # esto es distinto a DataSetCreation, lo divide en train y test
    patient_non_seizure_test_files = non_seizure_files[min(4, len(seizure_files)):]
        
    return patient_nseizure_train_files, patient_non_seizure_test_files


def get_false_detection_dftest(df, target, files):
    """
    Prepara los datos de prueba a partir de un DataFrame filtrado según registros específicos.

    Parámetros de entrada
     - df : pd.DataFrame. DataFrame completo con todas las características y etiquetas.
     - target : str. Nombre de la columna objetivo/etiqueta.
     - files : list o str. Información usada para filtrar los registros relevantes mediante `get_expression`.

    Salida
     - x_test : pd.DataFrame. DataFrame de características para el conjunto de prueba.
     - y_test : pd.Series. Etiquetas correspondientes para el conjunto de prueba.
    """

    df_test = eval(get_expression(files, 'register'))  # filtra
    df_test = df_test.drop(['register'], axis=1)       # una vez seleccionadas las filas, se elimina la columna 'register'
    x_test, y_test = df_test[df_test.columns.difference([target])], df_test[target]
    
    return x_test, y_test


def create_sequences(df, label_column='seizure', sequence_length=5):
    """
    Crea secuencias de características y etiquetas a partir de un DataFrame para entrenamiento de modelos secuenciales.

    Parámetros de entrada:
     - df : pd.DataFrame. DataFrame con las características y la columna de etiquetas.
     - label_column : str, opcional. Nombre de la columna que contiene las etiquetas (por defecto 'seizure').
     - sequence_length : int, opcional. Longitud de las secuencias a generar (por defecto 5).

    Salida:
     - X : np.ndarray. Array de secuencias de características con forma (n_secuencias, sequence_length, n_características).
     - y : np.ndarray. Array de etiquetas correspondientes al último paso de cada secuencia.
    """

    X, y = [], []
    feature_columns = df.columns.drop(label_column)
    
    for i in range(len(df) - sequence_length + 1):
        seq = df.iloc[i:i+sequence_length][feature_columns].values
        label = df.iloc[i + sequence_length - 1][label_column]
        X.append(seq)
        y.append(label)
        
    return np.array(X), np.array(y)


def evaluate_seizure_detection(y_true, y_pred, timeW):   
    """
    Evalúa la detección de convulsiones reales y calcula el retraso en su detección.

    Parámetros de entrada
     - y_true : pd.Series. Etiquetas verdaderas (0 = sin convulsión, 1 = convulsión).
     - y_pred : pd.Series o np.ndarray. Predicciones del modelo (True/False o 1/0).
     - timeW : float. Duración de cada ventana de predicción en segundos.

    Salida
     - detected : list de tuples. Cada elemento corresponde a una convulsión y contiene:
         - bool: True si la convulsión fue detectada, False en caso contrario.
         - float: retraso en segundos hasta la primera detección (0 si no se detectó).
    """

    y_true_arr = (y_true.to_numpy() - 0.5)               # Convierte los 0 en -0.5 y los 1 en 0.5 de y_true
    state_changes = np.where(np.diff(np.signbit(y_true_arr)))[0] + 1   # identifica los puntos donde cambian los valores de con a sin convolución y viceversa
    seizures = [state_changes[n:n+2] for n in range(0, len(state_changes), 2)]    
    detected = []
    for seizure in seizures:
        seizure_pred = list(y_pred[seizure[0]:seizure[1]])
        try:
            delay = seizure_pred.index(True)*timeW
            detected.append((True, delay))
        except ValueError:
            detected.append([False, 0])

    return detected


def evaluate_false_detections(y_true, y_pred, timeW):
    """
    Calcula el número de falsas detecciones de convulsiones en un conjunto de predicciones.

    Parámetros de entrada
     - y_true : pd.Series o np.ndarray. Etiquetas verdaderas (0 = sin convulsión, 1 = convulsión).
     - y_pred : pd.Series o np.ndarray. Predicciones del modelo (0 = sin convulsión, 1 = convulsión).
     - timeW : float. Duración de cada ventana de predicción en segundos.

    Salida
     - false_detections : list de tuples. Cada elemento contiene:
         - total_test_time : float. Tiempo total del test en segundos.
         - num_false_alarms : int. Número de falsas detecciones (transiciones de 0 a 1 fuera de convulsión).
    """

    total_test_time = timeW*len(y_pred)
    new_y_pred = [y_pred[i] for i in range(len(y_pred)) if y_pred[i] != y_pred[i-1]]
    false_detections = [(total_test_time, sum(new_y_pred))]

    return false_detections
    
    