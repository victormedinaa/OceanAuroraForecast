import torch
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from aurora import rollout
from src.training.metrics import Metricas
import numpy as np

def configure_optimizer(model, lr):
    """
    Configura y devuelve el optimizador.

    Args:
        model: Modelo a optimizar.
        lr (float): Tasa de aprendizaje.

    Returns:
        torch.optim.Optimizer: Optimizador configurado.
    """
    return AdamW(model.parameters(), lr=lr)

def train(
    model,
    train_generator,
    val_generator,
    num_epochs,
    criterion,
    optimizer,
    device,
    latitudes
):
    """
    Entrena el modelo y calcula el RMSE en la validación.
    """
    train_losses = []
    val_losses = []
    val_rmses = []
    scaler = GradScaler()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        total_batches = 0

        for batch, batch_target in tqdm(train_generator, desc=f"Epoch {epoch + 1}/{num_epochs} - Train"):
            optimizer.zero_grad()

            if getattr(batch.metadata, 'is_padding', False):
                continue

            target = batch_target.surf_vars['thetao'].to(device)

            with autocast(device_type=device.type):
                outputs = [out.to(device) for out in rollout(model, batch, steps=1)]
                model_output = outputs[0]
                output_tensor = model_output.surf_vars['thetao']

                loss_matrix = criterion(output_tensor, target)
                loss_per_sample = loss_matrix.mean(dim=[2, 3])
                loss_per_sample = loss_per_sample.squeeze(1)
                loss = loss_per_sample.mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            
            

            train_loss += loss.item()
            total_batches += 1

       
        average_train_loss = train_loss / total_batches if total_batches > 0 else 0
        train_losses.append(average_train_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_losses[-1]:.6f}")

        # Validación
        model.eval()
        val_loss = 0.0
        total_val_batches = 0
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for batch, batch_target in tqdm(val_generator, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"):
                if getattr(batch.metadata, 'is_padding', False):
                    continue

                target = batch_target.surf_vars['thetao'].to(device)
                outputs = [out.to(device) for out in rollout(model, batch, steps=1)]
                model_output = outputs[0]
                output_tensor = model_output.surf_vars['thetao']

                loss_matrix = criterion(output_tensor, target)
                loss_per_sample = loss_matrix.mean(dim=[2, 3])
                loss_per_sample = loss_per_sample.squeeze(1)
                loss = loss_per_sample.mean()
                
                

                val_loss += loss.item()
                total_val_batches += 1

                
                val_predictions.append(output_tensor.detach().cpu().numpy())
                val_targets.append(target.detach().cpu().numpy())

            

        average_val_loss = val_loss / total_val_batches if total_val_batches > 0 else 0
        val_losses.append(average_val_loss)

        if val_predictions and val_targets:
            val_predictions = np.concatenate(val_predictions, axis=0)
            val_targets = np.concatenate(val_targets, axis=0)
            val_predictions = val_predictions.squeeze(axis=1)
            val_targets = val_targets.squeeze(axis=1)
            val_rmse = Metricas.rmse(val_predictions, val_targets, latitudes)
            val_rmses.append(val_rmse)
            print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_losses[-1]:.6f}, Validation RMSE: {val_rmse:.6f}")
        else:
            val_rmses.append(0)
            print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_losses[-1]:.6f}, No se pudo calcular el RMSE.")

    return train_losses, val_losses, val_rmses
