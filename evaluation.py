import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def get_lstm_metrics(y_test, y_pred, timeW):
    """
    Calcula métricas de evaluación para un modelo LSTM de detección de crisis.

    Parámetros
     - y_test : np.ndarray o pd.Series. Etiquetas verdaderas de la señal (0 = no crisis, 1 = crisis).
     - y_pred : np.ndarray o pd.Series. Predicciones del modelo (0 = no crisis, 1 = crisis).
     - timeW : float. Duración de cada ventana de predicción en segundos.

    Devuelve
     - metrics : dict. Diccionario con las métricas calculadas:
         - 'detection_rate' : tasa de crisis detectadas correctamente.
         - 'avg_detection_delay' : retraso promedio en la detección de crisis (segundos).
         - 'false_alarms_per_hour' : número de falsas alarmas por hora.
         - 'accuracy' : precisión global de la predicción.
         - 'f1_score' : F1-score de la predicción.
    """
    
    # 1. Detección por crisis
    detections = []
    delays = []
    idx = 0
    while idx < len(y_test):
        if y_test[idx] == 1:
            start = idx
            while idx < len(y_test) and y_test[idx] == 1:
                idx += 1
            end = idx

            # ¿Se detectó alguna vez durante la crisis?
            if y_pred[start:end].sum() > 0:
                detections.append(1)
                # Calcular el retraso (primer positivo menos inicio)
                delay = np.argmax(y_pred[start:end]) * timeW
                delays.append(delay)
            else:
                detections.append(0)
        else:
            idx += 1

    detection_rate = np.mean(detections)
    avg_delay = np.mean(delays) if delays else np.nan

    # 2. Falsas alarmas (positivos fuera de crisis)
    false_alarms = ((y_pred == 1) & (y_test == 0)).sum()

    # Duración total del test (en horas)
    total_duration_h = (len(y_test) * timeW) / 3600
    false_alarms_per_hour = false_alarms / total_duration_h

    # 3. Precisión
    acc = accuracy_score(y_test, y_pred)

    # 4. f1 score
    f1 = f1_score(y_test, y_pred)

    return {
        'detection_rate': detection_rate,
        'avg_detection_delay': avg_delay,
        'false_alarms_per_hour': false_alarms_per_hour,
        'accuracy': acc,
        'f1_score': f1
    }
