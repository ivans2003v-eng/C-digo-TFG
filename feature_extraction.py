import pyedflib
import numpy as np
from pylab import *
from scipy import signal
from scipy.signal import welch, filtfilt, lfilter
import pandas as pd
from scipy.stats import *
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import json
# import skimage

# CARGAR LA CONFIGURACIÓN DE LA EJECUCIÓN
with open('config_execution.json', 'r', encoding='utf-8') as archivo:
    config_execution = json.load(archivo)

def read_data(filename, channels=[]):
    """
    Lee señales EEG desde un archivo EDF y devuelve los datos en formato NumPy.

    Parámetros
     - filename : str. Ruta del archivo EDF.
     - channels : list, opcional. Canales a leer. Si no se indican, se leen todos.

    Devuelve
     - data : np.ndarray. Señales en forma (n_canales, n_muestras).
     - fs : float. Frecuencia de muestreo (Hz).
     - time : np.ndarray. Vector de tiempo en segundos.
    """
    f = pyedflib.EdfReader(filename)    # Lee los datos 
    
    # if no channels are passed to the function
    if len(channels) == 0:
        channels = f.getSignalLabels()
    
    channel_names = f.getSignalLabels()
    fs = f.getSampleFrequencies()           
    
    data = np.zeros((len(channels), f.getNSamples()[0]))    # Crea una matriz de ceros donde el nº filas = nº canales y el nº columnas = nº muestras del canal
                                                            # Si channels=['C3', 'C4'] y cada canal tiene 4096 muestras, la matriz data tendrá forma (2, 4096)
    for i, channel in enumerate(channels):                  # Si channels=['C3', 'C4'], en la primera iteración i=0, channel='C3'. 
        channel_index = next((idx for idx, name in enumerate(channel_names) if channel in name), None)  # Se busca la posición de 'C3' en channel_names y se extraen los datos de 'C3' y se guardan en data[0, :]
        
        if channel_index is not None:
            data[i, :] = f.readSignal(channel_index)
        else:
            print(f"Canal '{channel}' no encontrado en channel_names.")

    time = np.linspace(0, data.shape[1]/fs[0], data.shape[1])    # La duración es el nº muestras entre la frecuencia de muestreo, lo que da la duración total de la señal en segundos
    f._close()
    return data, fs[0], time

def trunc(data, timeW, fs):         
    """
    Trunca las señales para que encajen exactamente en ventanas de duración fija.
    
    Parámetros
     - data : np.ndarray. Señales en forma (n_canales, n_muestras).
     - timeW : float. Duración de la ventana en segundos.
     - fs : float. Frecuencia de muestreo (Hz).

    Devuelve
     - data : np.ndarray. Señales truncadas en forma (n_canales, n_muestras_truncadas).
     - time : np.ndarray. Vector de tiempo de la señal truncada en segundos.
     - nw : int. Número total de ventanas completas.
     - N : int. Número de muestras por ventana.
    """
    samples = data.shape[1]  
    
    N = int(timeW*fs)     # Calcula el num de muestras que corresponden a una ventana de timeW segundos
    nw = int(samples//N)  # Calcula cuántas ventanas caben en la señal

    data = data[:, -nw*N:]   # Recorta la matriz de datos para que su num de columnas sea múltiplo de N
    time = np.linspace(0, data.shape[1]/fs, data.shape[1])   # Duración de la señal truncada en segundos
    return data, time, nw, N 


def segment_with_seizure(s, N, overlap):
    """
    Segmenta una señal en ventanas con solapamiento.

    Parámetros
     - s : np.ndarray. Señal unidimensional a segmentar.
     - N : int. Número de muestras por ventana.
     - overlap : float. Proporción de solapamiento entre ventanas (0–1).

    Devuelve
     - windows : np.ndarray. Ventanas de la señal en forma (n_ventanas, N).
     - nw : int. Número total de ventanas obtenidas.
    """
    step_size = int(N * (1 - overlap))
    n_samples = len(s)
    nw = (n_samples - N) // step_size + 1

    windows = np.zeros((nw, N))
    for i in range(nw):
        start = i * step_size
        end = start + N
        windows[i, :] = s[start:end]

    return windows, nw


# Un filtro highpass permite pasar las frecuencias por encima de una freq de corte y las inferiores las atenua
def highpass(s, f0, fs, order = 5):                                       
    """ 
    Aplica un filtro paso alto a una señal para eliminar frecuencias bajas.

    Parámetros
     - s : np.ndarray. Señales en forma (n_canales, n_muestras).
     - f0 : float. Frecuencia de corte (Hz).
     - fs : float. Frecuencia de muestreo (Hz).
     - order : int, opcional. Orden del filtro Butterworth (por defecto 5).

    Devuelve
     - filtered : np.ndarray. Señales filtradas con el mismo formato que la entrada.
    """
    b, a = signal.butter(order, f0 / (fs/2), btype='high', analog=False)    # signal.butter crea un filtro -> a y b son los coeficientes que hacen el filtro
    return filtfilt(b, a, s, axis = 1)             # aplica el filtro a la señal s con filtfilt (axis=1 dice que las señales están organizadas por columnas)


# Un filtro lowpass permite pasar las frecuencias por debajo de una freq de corte y las superiores las atenua 
def lowpass(s, f0, fs, order=5):
    """
    Aplica un filtro paso bajo a una señal para eliminar frecuencias altas.

    Parámetros
     - s : np.ndarray. Señales en forma (n_canales, n_muestras).
     - f0 : float. Frecuencia de corte (Hz).
     - fs : float. Frecuencia de muestreo (Hz).
     - order : int, opcional. Orden del filtro Butterworth (por defecto 5).

    Devuelve
     - filtered : np.ndarray. Señales filtradas con el mismo formato que la entrada.

    """
    b, a = signal.butter(order, f0 / (fs/2))
    return lfilter(b, a, s, axis = 1)           


# Un filtro rechazo de banda atenúa las freq dentro de un rango específico
def notchfilter(s, f0, fs, Q = 30):
    """
    Aplica un filtro notch a la señal para eliminar una frecuencia específica (rechazo de banda).
    
    Parámetros
     - s : np.ndarray. Señales en forma (n_canales, n_muestras) o unidimensional.
     - f0 : float. Frecuencia a eliminar (Hz), por ejemplo 50 o 60 Hz.
     - fs : float. Frecuencia de muestreo (Hz).
     - Q : float, opcional. Factor de calidad del filtro, controla el ancho de banda (por defecto 30).
    
    Devuelve
     - filtered : np.ndarray. Señales filtradas con la misma forma que la entrada.
    """
    b, a = signal.iirnotch(f0 / (fs/2), Q) 
    return lfilter(b, a, s)


def frec2sample_range(fi, fo, fs, N):    # convierte un rango de frecuencias (fi, fo) a un rango de indices de muestras (si, so)
    """
    Convierte un rango de frecuencias a un rango de índices de muestras.

    Parámetros
     - fi : float. Frecuencia inicial del rango (Hz).
     - fo : float. Frecuencia final del rango (Hz).
     - fs : float. Frecuencia de muestreo (Hz).
     - N : int. Número total de muestras de la señal.

    Devuelve
     - si : int. Índice de la muestra correspondiente a la frecuencia inicial.
     - so : int. Índice de la muestra correspondiente a la frecuencia final.
    """
    si = max(1,floor(fi*N/fs))           # calcula el índice de muestra correspondiente a la frecuencia fi
    so = ceil(fo*N/fs)                   # calcula el índice de muestra correspondiente a la frecuencia fo
    return int(si), int(so)


def calculate_spectras(window_matrix, sample_rate):
    """
    Calcula la densidad espectral de potencia normalizada (PSD) de cada ventana de señal.

    Parámetros
     - window_matrix : np.ndarray. Matriz de ventanas de señal, cada fila es una ventana.
     - sample_rate : float. Frecuencia de muestreo de la señal (Hz).

    Devuelve
     - spectras : list de np.ndarray. Lista de PSD normalizadas para cada ventana, 
       limitada al rango de 1 a 60 Hz.
    """
    spectras = []
    for window in window_matrix:
        # Spectral entropy
        freqs, psd = welch(window, sample_rate, nperseg=(sample_rate*2), window='hamm', scaling='density', axis=0)
        idx_min = np.argmax(np.round(freqs) > 1) - 1
        idx_max = np.argmax(np.round(freqs) > 60)
        npsd = psd[idx_min:idx_max]/sum(psd[idx_min:idx_max])
        spectras.append(npsd)
    return spectras


def band_energy(fft, fs):  # Calcula la energía de la señal en diferentes bandas de freq a partir de su Transformada de Fourier
    """
    Calcula la energía de una señal en diferentes bandas de frecuencia a partir de su FFT.

    Parámetros
     - fft : np.ndarray. Transformada de Fourier de la señal (magnitudes de frecuencia).
     - fs : float. Frecuencia de muestreo de la señal (Hz).

    Devuelve
     - et : float. Energía total de todas las frecuencias.
     - d : float. Energía en la banda Delta (0.5–4 Hz).
     - t : float. Energía en la banda Theta (4–7 Hz).
     - a : float. Energía en la banda Alpha (7–13 Hz).
     - b : float. Energía en la banda Beta (13–30 Hz).
     - g : float. Energía en la banda Gamma (30–50 Hz).
    """
    N = len(fft)
    
    et = sum(fft) # Energia total de todas las frecuencias 
    dsi, dso = frec2sample_range(0.5, 4, fs, N)  # Convierte el rango de frecuencias 0.5–4 Hz a índices con frec2sample_range
    d = sum(fft[dsi:dso])                        # suma las amplitudes en los índices correspondientes a la banda Delta 
    tsi, tso = frec2sample_range(4, 7, fs, N)
    t = sum(fft[tsi:tso])                        # suma las amplitudes en los índices correspondientes a la banda Theta 
    asi, aso = frec2sample_range(7, 13, fs, N)
    a = sum(fft[asi:aso])                        # suma las amplitudes en los índices correspondientes a la banda Alpha 
    bsi, bso = frec2sample_range(13, 30, fs, N)
    b = sum(fft[bsi:bso])                        # suma las amplitudes en los índices correspondientes a la banda Beta
    gsi, gso = frec2sample_range(30, 50, fs, N)
    g = sum(fft[gsi:gso])                        # suma las amplitudes en los índices correspondientes a la banda Gamma 
    return et, d, t, a, b, g                     # Devuelve las energías de cada banda

def psd(signal):
    """
    Calcula la densidad espectral de potencia (PSD) de una señal a partir de su FFT.

    Parámetros
     - signal : np.ndarray. Señal unidimensional de la cual se quiere obtener la PSD.

    Devuelve
     - Px : np.ndarray. Vector con la potencia de cada componente de frecuencia de la señal.
    """
    X = fft(signal)        # Calcula la FFT de la señal 
    Px = real(X*conj(X))   # Calcula el conjugado complejo de cada valor en X
    return Px              # Px es un vector con la potencia que tiene cada componente de freq de la señal  

def exponential_smooth(timeseries, alpha=0.3):
    """
    Aplica suavizado exponencial simple a una serie temporal.

    Parámetros
     - timeseries : np.ndarray o pd.Series. Serie temporal a suavizar.
     - alpha : float, opcional. Nivel de suavizado (0–1), por defecto 0.3.

    Devuelve
     - fittedvalues : np.ndarray o pd.Series. Serie temporal suavizada.
    """
    model = SimpleExpSmoothing(timeseries)    
    fit = model.fit(smoothing_level=alpha)
    return fit.fittedvalues            # devuelve la serie temporal con las fluctuaciones suavizadas 

def spectral_centroid(nf, ps):         # calcula el centroide espectral: qué freq contiene la mayor parte de la energía de la señal
    return sum(nf * ps)

def variational_coeff(nf, ps, sc):     # Calcula el coeficiente de variación espectral: dispersión relativa de una distribución
    return sum( (((nf - sc))**2) * ps) / sum(ps)

def spectral_skew(nf, ps, sc, vc):     # Asimetría espectral: indica si la potencia se concentra más hacia freq bajas o altas 
    return sum( ((nf - sc)/vc)**3 * ps) / sum(ps)
   
def bandpower(freqs, psd, band, output = False):       
    """
    Calcula la potencia espectral dentro de un rango de frecuencias específico.

    Parámetros
     - freqs : np.ndarray. Vector de frecuencias correspondientes a cada componente de la PSD.
     - psd : np.ndarray. Potencia espectral de la señal.
     - band : list o tuple. Banda de frecuencias [f_min, f_max] (Hz) sobre la que se calcula la potencia.
     - output : bool, opcional. Parámetro no utilizado actualmente (por defecto False).

    Devuelve
     - psd_band : float. Suma de la potencia espectral dentro del rango de frecuencias especificado.
    """
    band = np.asarray(band)
    low, high = band                   # frecuencias mínima y máxima de la banda
    
    # Find closest indices of band in frequency vector
    idx_min = np.argmax(np.round(freqs) > low) -1      #Encuentra los índices más cercanos de las freq dentro de la banda
    idx_max = np.argmax(np.round(freqs) > high)
    # select frequencies of interest
    psd = sum(psd[idx_min:idx_max])    # Suma las potencias espectrales en ese rango de frecuencias 

    return psd

def power_measures(data, sample_rate, output=False):   
    """
    Calcula la potencia de una señal en diferentes bandas de frecuencia y la PSD normalizada.

    Parámetros
     - data : np.ndarray. Señal unidimensional a analizar.
     - sample_rate : float. Frecuencia de muestreo de la señal (Hz).
     - output : bool, opcional. Parámetro no utilizado actualmente (por defecto False).

    Devuelve
     - total_power : float. Potencia total en la banda 0.5–127 Hz.
     - band_powers : np.ndarray. Potencias en bandas específicas: Delta, Theta, Alpha, Beta, Gamma1–Gamma5.
     - npsd : np.ndarray. PSD normalizada en el rango 1–60 Hz.
    """

    bandpasses = [[[0.5,127],'power'],
                  [[0.5,4],'power_delta'],
                  [[4,8],'power_theta'],
                  [[8,13],'power_alpha'],
                  [[13,30],'power_beta'],
                  [[30,50],'power_gamma1'],
                  [[50,70],'power_gamma2'],
                  [[70,90],'power_gamma3'],
                  [[90,110],'power_gamma4'],
                  [[110,127],'power_gamma5']]
    
    # Compute the periodogram (Welch)
    freqs, psd = welch(data, sample_rate, nperseg=(sample_rate), window='hamm', scaling='density', axis=0)  # se calcula la densidad espectral de potencia (psd). 
                                                                                                            # Devuelve un vector de freq y uno con la psd correspondiente a cada freq
    idx_min = np.argmax(np.round(freqs) > 1) - 1   # Encuentra el índice en freqs que corresponde a la 1º freq mayor que 1 Hz
    idx_max = np.argmax(np.round(freqs) > 60)      # Encuentra el índice que corresponde a la 1º freq mayor que 60Hz
    npsd = psd[idx_min:idx_max]/sum(psd[idx_min:idx_max])   #Calcula la PSD normalizada en el rango de 1Hz - 60Hz
    
    bandpass_data = np.zeros(len(bandpasses))               # Crea un arreglo de ceros del mismo tamaño que el num de bandas definidas en bandpasses
    for i, [bandpass, freq_name] in enumerate(bandpasses):
        bandpass_data[i] = bandpower(freqs, psd, bandpass)  # Rellena el arreglo con la potencia en cada banda 
    return bandpass_data[0], bandpass_data[1:], npsd  # Devuelve la potencia total en la banda de 0.5-127Hz, potencias de las bandas específicas y la PSD normalizada entre 1-60Hz

def channel_processing(channel_matrix, fs):    
    """
    Extrae características estadísticas y de potencia de múltiples canales de señal EEG.

    Parámetros
     - channel_matrix : np.ndarray. Matriz de señales, cada fila es un canal/señal.
     - fs : float. Frecuencia de muestreo de las señales (Hz).

    Devuelve
     - df : pd.DataFrame. DataFrame donde cada fila corresponde a un canal/señal y cada columna a una característica extraída.
    """
    
    ninstances = channel_matrix.shape[0]    # num de canales/señales (filas de channel_matrix)
    power_bands = zeros([ninstances, 9])    # matriz para almacenar la potencia en 9 bandas de freq para cada señal 
    total_energy = zeros(ninstances)        # vector de la energía total de cada señal 

    variancev = zeros(ninstances)           # estos son vectores que almacenarán la varianza...
    stdv = zeros(ninstances)                # ...desviación estándar...
    zcrossingsv = zeros(ninstances)         # ...cruces por cero...
    p2pv = zeros(ninstances)                # ...rango pico a pico (diferencia entre valor máximo y mínimo)...
    shannon_entropy = zeros(ninstances)     # ...y entropía de Shannon

    features = [
        'variance', 'std', 'zero_crossings', 'peak2peak', 'shannon_entropy',
        'variance-1', 'std-1', 'zero_crossings-1', 'peak2peak-1', 'shannon_entropy-1',
        'variance-2', 'std-2', 'zero_crossings-2', 'peak2peak-2', 'shannon_entropy-2',
        
        'total_energy', 'delta', 'theta', 'alpha', 'beta',
        'total_energy-1', 'delta-1', 'theta-1', 'alpha-1', 'beta-1',
        'total_energy-2', 'delta-2', 'theta-2', 'alpha-2', 'beta-2',
        
        'delta_theta', 'theta_alpha', 'alpha_beta', 'gammas',
        'delta_theta-1', 'theta_alpha-1', 'alpha_beta-1', 'gammas-1',
        'delta_theta-2', 'theta_alpha-2', 'alpha_beta-2', 'gammas-2'
    ]

    for index, row in enumerate(channel_matrix):                                    # Recorre cada señal de channel_matrix
        total_energy[index], power_bands[index, :], npsd = power_measures(row, fs)  # Calcula la energía total, potencias en 9 bandas y la psd de la señal
        shannon_entropy[index] = -sum(npsd*log2(npsd))                              # calcula la entropía de Shannon
        
        try:                            # extrae características básicas de la señal
            variancev[index] = var(row)
            stdv[index] = std(row)
            zcrossingsv[index] = len(np.where(np.diff(np.sign(row)))[0])
            p2pv[index] = max(row)-min(row)

        except ZeroDivisionError:       # en caso de error (las señales constantes causan divisiones por cero) se asignan valores predefinidos
            variancev[index] = 0.001
            stdv[index] = 0.001
            zcrossingsv[index] = 0.001
            p2pv[index] = 0.001
            
    # Calculate relative energy of each band
    relative_power_bands = np.divide(power_bands, total_energy[:, None])    # potencias de cada banda entre la energía total
    #Calculate energy in theta and alpha, and the rest of the bands
    delta_theta = 10*log10(power_bands[:, 0] + power_bands[:, 1])
    theta_alpha = 10*log10(power_bands[:, 1] + power_bands[:, 2])
    alpha_beta = 10*log10(power_bands[:, 2] + power_bands[:, 3])
    gammas = 10*log10(power_bands[:, 4] + power_bands[:, 5] + power_bands[:, 6] + power_bands[:, 7] + power_bands[:, 8])
    
    power_bands_db = 10*log10(power_bands)     # convierte las potencias y energía total a decibelios

    data = [  (variancev[2:]), (stdv[2:]), (zcrossingsv[2:]), (p2pv[2:]), (shannon_entropy[2:]),
                (variancev[1:-1]), (stdv[1:-1]), (zcrossingsv[1:-1]), (p2pv[1:-1]), (shannon_entropy[1:-1]),
                (variancev[0:-2]), (stdv[0:-2]), (zcrossingsv[0:-2]), (p2pv[0:-2]), (shannon_entropy[0:-2]),
                
                (total_energy[2:]), (power_bands_db[:, 0][2:]),   (power_bands_db[:, 1][2:]), (power_bands_db[:, 2][2:]), (power_bands_db[:, 3][2:]),
                (total_energy[1:-1]), (power_bands_db[:, 0][1:-1]),   (power_bands_db[:, 1][1:-1]), (power_bands_db[:, 2][1:-1]), (power_bands_db[:, 3][1:-1]),
                (total_energy[0:-2]), (power_bands_db[:, 0][0:-2]),   (power_bands_db[:, 1][0:-2]), (power_bands_db[:, 2][0:-2]), (power_bands_db[:, 3][0:-2]),
            
                delta_theta[2:], theta_alpha[2:], alpha_beta[2:], gammas[2:],
                delta_theta[1:-1], theta_alpha[1:-1], alpha_beta[1:-1], gammas[1:-1],
                delta_theta[0:-2], theta_alpha[0:-2], alpha_beta[0:-2], gammas[0:-2]
                
    ]   # Crea una matriz donde cada fila contiene las características de una instancia y hasta 2 pasos antes

            # variancev = [10, 20, 30, 40, 50]
            # variancev[2:]   → [30, 40, 50]
            # variancev[1:-1] → [20, 30, 40]
            # variancev[0:-2] → [10, 20, 30]
    
    data = np.array(data).transpose()     # transpone el array de datos para que cada fila corresponda a una señal y cada columna a una característica
    print(data.shape)
    df = pd.DataFrame(data, columns = features)   # crea un dataframe con las características organizadas en columnas 
    
    return df