import torch
import numpy as np
from aurora import rollout
from .metrics import Metricas
from src.utils.visualization import plot_target_prediction

def evaluate_model(model, model_paths, test_generator, device, latitudes, batch_size):
    metricas = Metricas(latitudes)
    for model_idx, model_path in enumerate(model_paths):
        print(f"Cargando modelo {model_idx + 1} desde {model_path}...")
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(f"Modelo {model_idx + 1} cargado.")

        predictions = []
        targets = []
        timestamps = []

        print(f"Realizando inferencia con el modelo {model_idx + 1}...")
        for batch, batch_target in test_generator:
            if getattr(batch.metadata, 'is_padding', False):
                continue

            target = batch_target.surf_vars['thetao'].to(device)
            targets.append(target.cpu().numpy())
            timestamps.extend(batch.metadata.time)

            with torch.no_grad():
                outputs = [out.to(device) for out in rollout(model, batch, steps=1)]
                model_output = outputs[0]
                prediction = model_output.surf_vars['thetao'].cpu().numpy()
                predictions.append(prediction)

        print(f"Inferencia completada para el modelo {model_idx + 1}.")

        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        rmse_value = metricas.rmse(predictions, targets)
        bias_value = metricas.bias(predictions, targets)
        acc_value = metricas.acc(predictions, targets)

        print(f"Resultados para el modelo {model_idx + 1}:")
        print(f"RMSE: {rmse_value}")
        print(f"Bias: {bias_value}")
        print(f"ACC: {acc_value}")

        idx = 0
        if idx < len(predictions):
            target_sample = targets[idx]
            prediction_sample = predictions[idx]
            timestamp_sample = timestamps[idx * batch_size:(idx + 1) * batch_size]

            for i in range(target_sample.shape[0]):
                print(f"Visualizando muestra {idx * batch_size + i} del modelo {model_idx + 1}...")
                plot_target_prediction(
                    target_sample[i],
                    prediction_sample[i],
                    timestamp_sample[i],
                    idx=idx * batch_size + i
                )
        else:
            print("Index out of range para visualización.")
