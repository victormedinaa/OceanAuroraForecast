import numpy as np

class Metricas:
    def __init__(self, latitudes):
        """ Inicializa la clase con latitudes y calcula las ponderaciones normalizadas. """
        self.latitudes = latitudes
        # Convertir latitudes a radianes
        lat_rad = np.deg2rad(self.latitudes)
        # Calcular los pesos y normalizar
        weights = np.cos(lat_rad)
        normalization_factor = np.mean(weights)
        self.weights = (weights / normalization_factor).reshape(1, -1, 1)
        
    
    def rmse(self, pred, target):
        """ Calcula el RMSE ponderado entre pred y target. """
        # Expandir las ponderaciones
        weights_full = np.tile(self.weights, (pred.shape[0], 1, pred.shape[2]))
        
        
        # Calcular los errores cuadrados
        errors_squared = (pred - target) ** 2
        # Aplanar los arrays
        errors_squared_flat = errors_squared.reshape(-1)
        weights_flat = weights_full.reshape(-1)
        # Crear una máscara para eliminar valores NaN
        mask = ~np.isnan(errors_squared_flat) & ~np.isnan(weights_flat)
        # Aplicar la máscara
        errors_squared_flat = errors_squared_flat[mask]
        weights_flat = weights_flat[mask]
        
        
        # Calcular la media ponderada de los errores cuadrados
        mse_weighted = np.average(errors_squared_flat, weights=weights_flat)
        rmse_weighted = np.sqrt(mse_weighted)
        return rmse_weighted
    
    def bias(self, pred, target):
        """ Calcula el sesgo ponderado entre pred y target. """
        # Expandir las ponderaciones
        weights_full = np.tile(self.weights, (pred.shape[0], 1, pred.shape[2]))
        
        # Calcular las diferencias
        differences = pred - target
        # Aplanar los arrays
        differences_flat = differences.reshape(-1)
        weights_flat = weights_full.reshape(-1)
        # Crear una máscara para eliminar valores NaN
        mask = ~np.isnan(differences_flat) & ~np.isnan(weights_flat)
        # Aplicar la máscara
        differences_flat = differences_flat[mask]
        weights_flat = weights_flat[mask]
        
        
        # Calcular el Bias ponderado
        bias_weighted = np.average(differences_flat, weights=weights_flat)
        return bias_weighted
    
    def acc(self, pred, target):
        """ Calcula el Coeficiente de Correlación de Anomalías (ACC) ponderado entre pred y target. """
        # Expandir las ponderaciones
        weights_full = np.tile(self.weights, (pred.shape[0], 1, pred.shape[2]))
        
        # Aplanar los arrays
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        weights_flat = weights_full.reshape(-1)
        # Crear una máscara para eliminar valores NaN
        mask = ~np.isnan(pred_flat) & ~np.isnan(target_flat) & ~np.isnan(weights_flat)
        # Aplicar la máscara
        pred_flat = pred_flat[mask]
        target_flat = target_flat[mask]
        weights_flat = weights_flat[mask]
        
        
        # Calcular las medias ponderadas
        pred_mean = np.average(pred_flat, weights=weights_flat)
        target_mean = np.average(target_flat, weights=weights_flat)
        
        
        # Calcular las anomalías
        pred_anomaly = pred_flat - pred_mean
        target_anomaly = target_flat - target_mean
        
        # Calcular covarianza y varianzas ponderadas
        covariance = np.average(pred_anomaly * target_anomaly, weights=weights_flat)
        pred_variance = np.average(pred_anomaly ** 2, weights=weights_flat)
        target_variance = np.average(target_anomaly ** 2, weights=weights_flat)
        
        
        # Calcular el ACC
        denominator = np.sqrt(pred_variance * target_variance)
        if denominator != 0:
            acc_weighted = covariance / denominator
        else:
            acc_weighted = np.nan
            print("El denominador es cero, ACC no está definido.")
        return acc_weighted
