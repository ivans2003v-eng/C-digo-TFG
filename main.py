# IMPORTS
import numpy as np
from scipy import signal
import pandas as pd
import glob, os
import json
import seaborn as sns
from sklearn.utils import class_weight
import time
from sklearn.metrics import recall_score
from scipy.stats import *
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from feature_extraction import *
from load_data import EDFReader, MATReader
from detection_models import get_expression, non_seizure_files_division, create_sequences, split_train_test, get_false_detection_dftest, merge_channels
from select_channel import get_best_channels, calculate_entropy, save_channel_rankings
from evaluation import *
import random
import tensorflow as tf

# FIJAR SEMILLAS PARA REPRODUCIBILIDAD
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# CARGAR LA CONFIGURACIÓN DE LA BASE DE DATOS
with open('config.json', 'r', encoding='utf-8') as archivo:
    database = json.load(archivo)

# CARGAR LA CONFIGURACIÓN DE LA EJECUCIÓN
with open('config_execution.json', 'r', encoding='utf-8') as archivo:
    config_execution = json.load(archivo)

basedir = os.getcwd()
os.chdir(database['bbdd_directory'])  

patients = config_execution['selected_patients']
timeW = config_execution['window_duration']
decimationCoeff = 1                        


respuesta = input("¿Quiere leer el dataset en bruto y extraer las características de cada ventana de la señal? (s/n): ").strip().lower()
if respuesta in ["si", "sí", "s", "yes"]:

    # COMIENZO DE LECTURA DE DATOS Y CREACIÓN DEL DATASET
    for patient in patients:
        print('---------------------------------------------- Patient: ' + patient + ' ----------------------------------------------------')
        fdir = database['bbdd_directory'] + '\\' + patient       
        os.chdir(fdir)                   

        if database['format'] == "EDF":
            print("Leyendo el dataset con formato EDF...")
            annotation = glob.glob('*txt')     
            edf_reader = EDFReader(annotation[0], database)                        
            registers = edf_reader.get_registers()
            channel_index = edf_reader.get_channel_index()

        if database['format'] == "MAT":
            print("Leyendo el dataset con formato MAT...")
            annotation = patient + "_info.mat"    
            mat_reader = MATReader(annotation, database, patient)
            registers = mat_reader.get_registers()

        dataframe = pd.DataFrame()
        for key, value in registers.items():   # Itera sobre cada registro del paciente. 
                                               # Value es cada instancia de la clase Register y key es el nombre del file
            
            files_with_seizures = sorted(registers.keys())     # las key son los nombres de los archivos, es decir que saca los nombre de archivos que contienen seizure

            print("Leyendo las señales...")
            if key in files_with_seizures: 
                if database['format'] == "MAT":
                    signals, originalfs, time = mat_reader.read_data(key)   # signals es una matriz en la que cada fila es un canal y las columnas son las muestras
                                                                            # originalsfs = frecuencia de muestreo
                                                                            # time = tiempo en segundos de la señal
                    channel_index = mat_reader.get_channel_index()
                    
                if database['format'] == "EDF":
                    signals, originalfs, time = read_data(key, channel_index.values())  # signals es una matriz en la que cada fila es un canal y las columnas son las muestras
                                                                                        # originalsfs = frecuencia de muestreo
                                                                                        # time = tiempo en segundos de la señal
                
                print("-Readed " + key)

                print("-Procesando la señal...")
                # Decimation
                signals = signal.decimate(signals, decimationCoeff)  # reduce la tasa de muestreo según el coeficiente decimationCoeff
                fs = int(originalfs//decimationCoeff)                # calcula la nueva tasa de muestreo después de la decimación
                
                # Filtering 
                if config_execution["filters"]["Filtro paso alto"]:
                    signals = highpass(signals, 0.5, fs)                 # pasa las señales por todos estos filtros
                if config_execution["filters"]["Filtro paso bajo"]:
                    signals = lowpass(signals, 127, fs)
                if config_execution["filters"]["Filtro notch"]:
                    signals = notchfilter(signals, 60, fs)
                    signals = notchfilter(signals, 120, fs)

                #signals = (signals - np.mean(signals, axis=1, keepdims=True)) / (np.std(signals, axis=1, keepdims=True) + 1e-6)
                media = np.mean(signals, axis=1, keepdims=True)
                desviacion = np.std(signals, axis=1, keepdims=True) + 1e-6
                
                parts_non_seizure = []
                parts_with_seizure = []

                # Seizure vector creation
                samples = signals.shape[1] 
                seizure = zeros(samples)                     # inicializa un vector de ceros para marcar las muestras que contienen convulsiones
                for n in range (len(value.seizures)):        # value es de la clase Register y seizure es un atributo de la clase Register
                    start = value.seizures[n][0]*fs
                    end = value.seizures[n][1]*fs
                    seizure[start:end] = np.ones(end-start)  # llena el vector seizure con 1 en las regiones que corresponden a convulsiones
                
                #print(signals.shape)
                signals = np.vstack([signals, seizure])  # La última fila de la matriz será la etiqueta de seizure o no 
                print(signals.shape)
                overlap = 0.5
                auxdf = pd.DataFrame() 

                print("-Segmentando la señal en ventanas...")
                for n in range (len(value.seizures)):        # value es de la clase Register y seizure es un atributo de la clase Register
                    start = value.seizures[n][0]*fs
                    end = value.seizures[n][1]*fs

                    # ==== SEGMENTO ANTES DE LA PRIMERA CRISIS ====
                    if (n == 0) and (start != 1): 
                        signals_trunc, time, nw, N = trunc(signals[:, :start], timeW, fs)  # segmenta las señales en ventanas de duración timeW

                        # ==== Cálculo de correlaciones ====
                        #matrix_correlations_all_windows = np.zeros((len(channel_index), nw))
                        #if nw != 1:
                        #    for i, j in enumerate(range(0, signals[:-1, start:end].shape[1], N)):
                        #        matrix_correlations = np.corrcoef(signals[:-1, j:j+N])
                        #        matrix_abs_correlations = np.abs(matrix_correlations)
                        #        matrix_sum_correlations = matrix_abs_correlations.sum(axis=1)
                        #        matrix_correlations_all_windows[:, i] = matrix_sum_correlations

                        for channel, s in enumerate(signals_trunc[:-1]):
                            newSignal = np.reshape(s, [nw, N])
                            spectras = calculate_spectras(newSignal, fs)
                            varianza = np.var(newSignal[2:], axis=1)
                            s_entropy, t_entropy = calculate_entropy(newSignal[2:], spectras)

                            newSignal_norm = (newSignal - media[channel, 0]) / desviacion[channel, 0]

                            newdf = channel_processing(newSignal_norm, fs)
                            newdf['channel'] = pd.Series( [channel_index[channel]]*(nw-2), index = newdf.index) # se agrega una columna con el nombre del canal correspondiente
                            newdf['register'] = pd.Series( [key]*(nw-2))                      # se agrega una columna que contiene el nombre del registro
                            newdf['seizure'] = 0
                            newdf['var'] =  varianza
                            newdf['s_entropy'], newdf['t_entropy'] = s_entropy, t_entropy
                            #newdf['sum_corr'] = matrix_correlations_all_windows[channel,2:]
                            auxdf = pd.concat([auxdf, newdf], ignore_index=True)              # esto hace que se vayan acumulando los resultados de todos los canales en un df    
                        print("--Añadido inicio")

                    # ==== PERIODO DURANTE LA CRISIS ====
                    seizure_signal = signals[:, start:end][-1]  # Última fila
                    seizureW, nw = segment_with_seizure(seizure_signal, N, overlap)
                    seizureW = (np.sum(seizureW, axis=1) > N // 2)

                    # ==== Cálculo de correlaciones ====
                    #matrix_correlations_all_windows = np.zeros((len(channel_index), nw))
                    #if nw != 1:
                    #    for i, j in enumerate(range(0, signals[:-1, start:end].shape[1], N)):
                    #        matrix_correlations = np.corrcoef(signals[:-1, j:j+N])
                    #        matrix_abs_correlations = np.abs(matrix_correlations)
                    #        matrix_sum_correlations = matrix_abs_correlations.sum(axis=1)
                    #        matrix_correlations_all_windows[:, i] = matrix_sum_correlations

                    for channel, s in enumerate(signals[:, start:end][:-1]):
                        newSignal, nw = segment_with_seizure(s, N, overlap)
                        spectras = calculate_spectras(newSignal, fs)
                        varianza = np.var(newSignal[2:], axis=1)
                        s_entropy, t_entropy = calculate_entropy(newSignal[2:], spectras)
                        
                        newSignal_norm = (newSignal - media[channel, 0]) / desviacion[channel, 0]

                        newdf = channel_processing(newSignal_norm, fs)
                        newdf['channel'] = pd.Series([channel_index[channel]] * len(newdf), index=newdf.index)
                        newdf['register'] = pd.Series([key] * len(newdf), index=newdf.index)
                        newdf['seizure'] = pd.Series( seizureW[2:], index = newdf.index)
                        newdf['var'] =  varianza
                        newdf['s_entropy'], newdf['t_entropy'] = s_entropy, t_entropy
                        #newdf['sum_corr'] = matrix_correlations_all_windows[channel,2:]
                        auxdf = pd.concat([auxdf, newdf], ignore_index=True)  
                    print("--Añadido periodo con crisis")

                    # ==== PERIODO ENTRE CRISIS ====
                    if n > 0: 
                        signals_trunc, time, nw, N = trunc(signals[:, (value.seizures[n-1][1]*fs):start], timeW, fs)  # segmenta las señales en ventanas de duración timeW

                        # ==== Cálculo de correlaciones ====
                        #matrix_correlations_all_windows = np.zeros((len(channel_index), nw))
                        #if nw != 1:
                        #    for i, j in enumerate(range(0, signals[:-1, start:end].shape[1], N)):
                        #        matrix_correlations = np.corrcoef(signals[:-1, j:j+N])
                        #        matrix_abs_correlations = np.abs(matrix_correlations)
                        #        matrix_sum_correlations = matrix_abs_correlations.sum(axis=1)
                        #        matrix_correlations_all_windows[:, i] = matrix_sum_correlations


                        for channel, s in enumerate(signals_trunc[:-1]):
                            newSignal = np.reshape(s, [nw, N])
                            spectras = calculate_spectras(newSignal, fs)
                            varianza = np.var(newSignal[2:], axis=1)
                            s_entropy, t_entropy = calculate_entropy(newSignal[2:], spectras)

                            newSignal_norm = (newSignal - media[channel, 0]) / desviacion[channel, 0]

                            newdf = channel_processing(newSignal_norm, fs)
                            newdf['channel'] = pd.Series( [channel_index[channel]]*(nw-2), index = newdf.index) # se agrega una columna con el nombre del canal correspondiente
                            newdf['register'] = pd.Series( [key]*(nw-2))                      # se agrega una columna que contiene el nombre del registro
                            newdf['seizure'] = 0
                            newdf['var'] =  varianza
                            newdf['s_entropy'], newdf['t_entropy'] = s_entropy, t_entropy
                            #newdf['sum_corr'] = matrix_correlations_all_windows[channel,2:]
                            auxdf = pd.concat([auxdf, newdf], ignore_index=True)              # esto hace que se vayan acumulando los resultados de todos los canales en un df    
                        print("--Añadido entre primera y segunda")
                    
                    # ==== PERIODO DESPUÉS DE LA ÚLTIMA CRISIS ====
                    if (n == (len(value.seizures)-1)) and (n != len(signals)): 
                        signals_trunc, time, nw, N = trunc(signals[:, end:], timeW, fs)  # segmenta las señales en ventanas de duración timeW

                        # ==== Cálculo de correlaciones ====
                        #matrix_correlations_all_windows = np.zeros((len(channel_index), nw))
                        #if nw != 1:
                        #    for i, j in enumerate(range(0, signals[:-1, start:end].shape[1], N)):
                        #        matrix_correlations = np.corrcoef(signals[:-1, j:j+N])
                        #        matrix_abs_correlations = np.abs(matrix_correlations)
                        #        matrix_sum_correlations = matrix_abs_correlations.sum(axis=1)
                        #        matrix_correlations_all_windows[:, i] = matrix_sum_correlations

                        for channel, s in enumerate(signals_trunc[:-1]):
                            newSignal = np.reshape(s, [nw, N])
                            spectras = calculate_spectras(newSignal, fs)
                            varianza = np.var(newSignal[2:], axis=1)
                            s_entropy, t_entropy = calculate_entropy(newSignal[2:], spectras)

                            newSignal_norm = (newSignal - media[channel, 0]) / desviacion[channel, 0]

                            newdf = channel_processing(newSignal_norm, fs)
                            newdf['channel'] = pd.Series( [channel_index[channel]]*(nw-2), index = newdf.index) # se agrega una columna con el nombre del canal correspondiente
                            newdf['register'] = pd.Series( [key]*(nw-2))                      # se agrega una columna que contiene el nombre del registro
                            newdf['seizure'] = 0
                            newdf['var'] =  varianza
                            newdf['s_entropy'], newdf['t_entropy'] = s_entropy, t_entropy
                            #newdf['sum_corr'] = matrix_correlations_all_windows[channel,2:]
                            auxdf = pd.concat([auxdf, newdf], ignore_index=True)              # esto hace que se vayan acumulando los resultados de todos los canales en un df    
                        print("--Añadido fin")

                # Add to the patient dataframe
                #print(auxdf)
                dataframe = pd.concat([dataframe, auxdf], ignore_index=True)          # añade los resultados del archico al df general del paciente 
                #print(dataframe.shape)
                print("-Rows created for " + key)

        print("Creando dataset con características de cada ventana...")
        # Save the datase and the csv with the list of significant channels
        dataframe.to_hdf(database['result_directory'] + '\\' + patient + 'features' + '.h5', key = 'fullpatient', mode = 'w', format = 'table')  # Guarda el df como un archivo HDF5

else:
    print("Saltando la creación del dataset con las características de cada señal...")

    
####################################################################################################################


respuesta = input("¿Quiere calcular el orden de canales más relevantes? (s/n): ").strip().lower()
if respuesta in ["si", "sí", "s", "yes"]:

    for patient in patients: 
        # === CREACIÓN DE ORDEN DE CANALES ===
        df = pd.read_hdf(database['result_directory'] + '\\' + patient + 'features' + '.h5', key = 'fullpatient')

        multindex_dataframe = df.groupby(by=['channel', "seizure"])[df.select_dtypes(include=[np.number]).columns].mean()
        channel_scores = pd.DataFrame()

        unstacked = multindex_dataframe.unstack(level='seizure')

        channel_scores['var'] = np.abs(unstacked['var'][1])
        channel_scores['var_diff'] = np.abs(unstacked['var'][1] - unstacked['var'][0])

        channel_scores['s_entropy'] = np.abs(unstacked['s_entropy'][1])
        channel_scores['s_entropy_diff'] = np.abs(unstacked['s_entropy'][1] - unstacked['s_entropy'][0])

        channel_scores['t_entropy'] = np.abs(unstacked['t_entropy'][1])
        channel_scores['t_entropy_diff'] = np.abs(unstacked['t_entropy'][1] - unstacked['t_entropy'][0])

        channel_scores['varXentropy'] = channel_scores['var'] * channel_scores['s_entropy']
        channel_scores['varXentropy_diff'] = channel_scores['var_diff'] * channel_scores['s_entropy_diff']

        #channel_scores['sum_corr'] = np.abs(unstacked['sum_corr'][1])
        #channel_scores['sum_corr_diff'] = np.abs(unstacked['sum_corr'][1] - unstacked['sum_corr'][0])

        print(channel_scores.sort_values('var', ascending=False))

        output_dir = database['result_directory']
        save_channel_rankings(patient, channel_scores, output_dir=output_dir)

else:
    print("Saltando el calculo del orden de canales...")

############################################################################################################


respuesta = input("¿Quiere entrenar el modelo? (s/n): ").strip().lower()
respuesta_eval = input("¿Quiere evaluar el modelo? (s/n): ").strip().lower()
print("Espere un momento...\n")

# DIVIDIR EN TRAIN Y TEST 
nchannels = 3
all_patient_results = {}  # Diccionario para acumular resultados de todos los pacientes

for i, patient in enumerate(patients):

    # Calculate most significant channels for patient 
    channelsdf = pd.read_csv(database['result_directory'] + '\\' + patient +  '_ranking_channels.csv', index_col=0)
    best_channels = get_best_channels(channelsdf, nchannels) 
    #print(best_channels)

    # Read the patient features dataframe    
    df = pd.read_hdf(database['result_directory'] + '\\' + patient + 'features' + '.h5', key = 'fullpatient')
    df = df.drop(['var', 's_entropy', 't_entropy'], axis=1)

    # Identificar los archivos con colvulsión
    df2 = df[['seizure', 'register']]
    df2 = df2[df2['seizure'] == 1]
    files_with_seizures = df2['register'].unique()

    # Extract only the best channels
    df = eval(get_expression(best_channels, 'channel')) 
    if nchannels > 1:
        df = merge_channels(df, best_channels)
    else:
        df = df.drop(['channel'], axis=1)

    patient_path = config_execution['database_directory'] + '\\' + patient

    # Get the training files of the patient
    patient_nseizure_train_files, patient_nseizure_test_files = non_seizure_files_division(files_with_seizures, patient_path)
    
    # Generate all the train and test dfs and store them in a list
    detected = []
    mydf_dict = {} 

    sequence_length = 15
    results = []

    for seizure_file in files_with_seizures:
        #print(f"Evaluando con test en archivo: {seizure_file}")
    
        # Archivos para entrenamiento y prueba
        training_seizure_files = list(files_with_seizures)
        training_seizure_files.remove(seizure_file)
        training_files = training_seizure_files + patient_nseizure_train_files
        seizure_test_file = [seizure_file]

        # Separar por train y test
        df_train = eval(get_expression(training_files, 'register')).drop(['register'], axis=1)
        df_test = eval(get_expression(seizure_test_file, 'register')).drop(['register'], axis=1)

        # Crear secuencias para LSTM
        X_train, y_train = create_sequences(pd.concat([df_train.drop('seizure', axis=1), df_train['seizure']], axis=1),
                                            label_column='seizure',
                                            sequence_length=sequence_length)

        X_test, y_test = create_sequences(pd.concat([df_test.drop('seizure', axis=1), df_test['seizure']], axis=1),
                                        label_column='seizure',
                                        sequence_length=sequence_length)
        """
        if len(X_test) == 0 or len(X_train) == 0:
            print(f"Skipping {seizure_file} due to insufficient data.")
            continue
        """

        if respuesta in ["si", "sí", "s", "yes"]:

            # Construir modelo LSTM
            model = Sequential()
            model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
            model.add(Dropout(0.3))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            # Calcular pesos de clase
            weights = class_weight.compute_class_weight( 
                class_weight='balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            class_weights = dict(enumerate(weights)) 

            # Entrenar
            model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=35,
                    batch_size=32,
                    class_weight=class_weights,
                    verbose=1)

            # Guardar modelo entrenado 
            directorio = config_execution['database_directory'] 
            ruta_modelo = os.path.join(directorio, "modelo.keras")
            model.save(ruta_modelo)

        else: 
            print("Saltando el entrenamiento del modelo...")
            model = load_model("modelo.keras")

        # Evaluación del modelo
        if respuesta_eval in ["si", "sí", "s", "yes"]:

            # Evaluar
            y_prob = model.predict(X_test).flatten()

            # Suavizado por media móvil sobre las probabilidades
            # Reduce falsas alarmas aisladas sin destruir detecciones reales
            smooth_window = 5
            y_prob_smooth = np.convolve(y_prob, np.ones(smooth_window)/smooth_window, mode='same')
            y_pred = (y_prob_smooth > 0.5).astype(int)

            metrics = get_lstm_metrics(y_test, y_pred, timeW=timeW)
            sensitivity = recall_score(y_test, y_pred, zero_division=0)

            results.append({'test_file': seizure_file, 
                            'accuracy': metrics['accuracy'], 
                            'f1_score': metrics['f1_score'], 
                            'detection_rate': metrics['detection_rate'], 
                            'avg_detection_delay': metrics['avg_detection_delay'],
                            'false_alarms_per_hour': metrics['false_alarms_per_hour'],
                            'sensitivity':sensitivity})

            print(f" -> Accuracy: {metrics['accuracy']:.3f} | F1-score: {metrics['f1_score']:.3f}")

        else: 
            print("Saltando paso de evaluación del modelo...")

    # GUARDAR RESULTADOS POR PACIENTE (después de todos los seizure_files)
    if results:
        # Directorio para guardar resultados
        results_output_path = os.path.join(database['result_directory'] + '\\' +'summary_results')
        os.makedirs(results_output_path, exist_ok=True)

        # Mostrar resultados del paciente
        results_df = pd.DataFrame(results)
        print(f"\n=== Cross-Validation Results para {patient} ===")
        print(results_df)

        print("\nPromedio:")
        print(results_df[['accuracy', 'f1_score','detection_rate', 'avg_detection_delay', 'false_alarms_per_hour']].mean())

        # Crear el diccionario con los resultados del paciente
        results_json = {
            'patient': patient,
            'database': database['name'],  # Asegúrate de que este campo exista en tu config.json
            'results': results
        }

        # Nombre del archivo JSON: paciente_basededatos_results.json
        filename = f"{patient}_{database['name']}_results.json"
        file_path = os.path.join(results_output_path, filename) 

        # Guardar el archivo
        with open(file_path, 'w') as f:
            json.dump(results_json, f, indent=4)

        print(f"\nResultados guardados en: {file_path}")

        # Acumular resultados para el resumen final
        all_patient_results[patient] = results_df

# === RESUMEN FINAL DE TODOS LOS PACIENTES ===
if all_patient_results:
    # Media global (valores de la primera gráfica)
    all_dfs = pd.concat(all_patient_results.values(), ignore_index=True)
    print("\n" + "=" * 80)
    print("MÉTRICAS PROMEDIO GLOBALES")
    print("=" * 80)
    for metric in ['accuracy', 'f1_score', 'sensitivity', 'detection_rate', 'avg_detection_delay', 'false_alarms_per_hour']:
        print(f"  {metric:.<30} {all_dfs[metric].mean():.4f}")

    print("\n" + "=" * 80)
    print("RESUMEN FINAL DE RESULTADOS POR PACIENTE")
    print("=" * 80)
    for patient, res_df in all_patient_results.items():
        print(f"\n{patient}:")
        avg = res_df[['accuracy', 'f1_score', 'sensitivity', 'detection_rate', 'avg_detection_delay', 'false_alarms_per_hour']].mean()
        for metric_name, value in avg.items():
            print(f"  {metric_name:.<30} {value:.4f}")
    print("\n" + "=" * 80)

#########################################################################################################

respuesta = input("\n¿Quiere ver gráficas de la evaluación del modelo? (s/n): ").strip().lower()
if respuesta in ["si", "sí", "s", "yes"]:

    # VER GRÁFICA COMPARATIVA ENTRE DIFERENTES BASES DE DATOS
    # Ruta donde están guardados los archivos JSON
    results_folder = os.path.join(database['result_directory'] + '\\' +'summary_results')
    print(results_folder)

    # Diccionario para acumular métricas por base de datos
    summary = {}

    # Leer todos los archivos JSON
    for filename in os.listdir(results_folder):
        if filename.endswith("_results.json"):
            filepath = os.path.join(results_folder, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                db_name = data['database']

                results = data['results']
                if results:
                    # Inicializar si no existe
                    if db_name not in summary:
                        summary[db_name] = {'accuracy': [], 'f1_score': [], 'sensitivity': []}

                    # Añadir métricas de cada resultado
                    for metric in summary[db_name].keys():
                        metric_values = [r[metric] for r in results if metric in r]
                        summary[db_name][metric].extend(metric_values)

    # Calcular la media de todas las métricas por base de datos
    for db_name in summary:
        for metric in summary[db_name]:
            values = summary[db_name][metric]
            summary[db_name][metric] = np.mean(values) if values else 0

    # Verificar y graficar
    if not summary:
        print("No se encontraron datos para graficar.")
    else:
        metrics = ['accuracy', 'f1_score', 'sensitivity']
        dbs = list(summary.keys())
        x = np.arange(len(metrics))
        width = 0.15

        fig, ax = plt.subplots(figsize=(10, 6))

        for i, db in enumerate(dbs):
            values = [summary[db][metric] for metric in metrics]
            print(values)
            ax.bar(x + i * width, values, width, label=db)

        ax.set_ylabel('Valor')
        ax.set_title('Métricas promedio por base de datos')
        ax.set_xticks(x + width * (len(dbs) - 1) / 2)
        ax.set_xticklabels(metrics, rotation=30)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()

    ###########################################################################

    # GRÁFICO DE CONJUNTO DE MÉTRICAS POR PACIENTE
    metrics = ['accuracy', 'f1_score', 'sensitivity']
    data_by_patient = {}

    for filename in os.listdir(results_folder):
        if filename.endswith('_results.json'):
            filepath = os.path.join(results_folder, filename)
            with open(filepath, 'r') as f:
                content = json.load(f)
                patient = content['patient']
                results = content['results']
                if results:
                    # Calcular promedio de cada métrica
                    avg_metrics = {}
                    for metric in metrics:
                        values = [r[metric] for r in results if metric in r]
                        avg_metrics[metric] = sum(values) / len(values) if values else 0
                    data_by_patient[patient] = avg_metrics

    colors = [#"#6583B3",  # Azul medianoche
            "#DF7377",  # Rojo apagado
            "#9A8EC0",  # Lavanda/gris violeta
            "#E2D194"]  # Amarillo dorado suave

    # Verifica si hay datos
    if not data_by_patient:
        print("No se encontraron datos de pacientes.")
    else:
        patients = list(data_by_patient.keys())
        x = np.arange(len(patients))  # posiciones en eje X

        bar_width = 0.2
        offset = np.linspace(-bar_width * (len(metrics) - 1) / 2, bar_width * (len(metrics) - 1) / 2, len(metrics))

        plt.figure(figsize=(14, 6))

        for i, metric in enumerate(metrics):
            values = [data_by_patient[p][metric] for p in patients]
            plt.bar(x + offset[i], values, width=bar_width, label=metric.replace("_", " ").title(), color=colors[i])

        plt.xticks(x, patients, rotation=45, ha='right')
        plt.ylabel("Valor")
        plt.title("Métricas por paciente")
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
