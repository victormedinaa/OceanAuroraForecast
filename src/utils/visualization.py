import matplotlib.pyplot as plt
import numpy as np

def plot_losses(train_losses, val_losses):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Validation Loss')
    plt.legend()
    plt.show()

def plot_target_prediction(target, prediction, timestamp, idx):
    

    target_img = target.squeeze()
    prediction_img = prediction.squeeze()
    difference_img = target_img - prediction_img

    plt.figure(figsize=(18, 6))

    vmin = min(target_img.min(), prediction_img.min())
    vmax = max(target_img.max(), prediction_img.max())

    plt.subplot(1, 3, 1)
    plt.imshow(target_img, cmap='viridis', vmin=vmin, vmax=vmax)
    plt.title(f'Target - Sample {idx}\nTimestamp: {timestamp}')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(prediction_img, cmap='viridis', vmin=vmin, vmax=vmax)
    plt.title(f'Prediction - Sample {idx}\nTimestamp: {timestamp}')
    plt.colorbar()

    diff_vmax = np.max(np.abs(difference_img))
    plt.subplot(1, 3, 3)
    plt.imshow(difference_img, cmap='seismic', vmin=-diff_vmax, vmax=diff_vmax)
    plt.title(f'Difference (Target - Prediction)\nSample {idx}')
    plt.colorbar()

    plt.tight_layout()
    plt.show()
