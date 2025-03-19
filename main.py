# Importación de librerías necesarias
import torch
import numpy as np
from src.data.data_loader import load_dataset, load_lsm
from src.data.data_preprocessing import DataPreprocessor
from src.training.train import train, configure_optimizer
from src.training.inference import evaluate_model
from src.utils.visualization import plot_losses
from src.data.batch_generator import BatchGenerator
from aurora import Aurora  
import torch.optim as optim

# Configuración de dispositivo
# Se utiliza GPU si está disponible, de lo contrario, CPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo configurado: {device}")

# ===================================
# Configuración de variables y rutas
# ===================================
variables = ['thetao']
depth_slice = slice(0, 10)

# Rutas de los archivos del dataset y máscara LSM

dataset_path = "D://ruta_al_archivo//cmems_mod_glo_phy_my_0.083deg_P1D-m_6years_thetao_v3.nc"
lsm_path = "D://ruta_al_archivo//datos_mascara.nc"

# ===================================
# Carga y preprocesamiento de datos
# ===================================

print("Cargando dataset...")
dataset = load_dataset(dataset_path, variables, depth_slice)

print("Cargando y aplicando máscara LSM...")
dataset['lsm'] = load_lsm(lsm_path, dataset)


# Configuración de niveles oceánicos y preprocesador
ocean_levels = dataset['depth'].values
preprocessor = DataPreprocessor(variables=variables, depth_levels=ocean_levels)

print("Iniciando preprocesamiento completo...")
# División del dataset en entrenamiento, validación y prueba
train_dataset, val_dataset, test_dataset = preprocessor.preprocess_all(dataset)

print("Preprocessing and normalization complete.")


# ===================================
# Configuración del modelo Aurora
# ===================================
print("Configurando el modelo Aurora...")
surf_vars = ('thetao',) # Variable oceanica superficial
static_vars = ('lsm',) # Variables estática
atmos_vars = ('thetao',) # Variable oceanica con niveles de profundidad

# Inicialización del modelo Aurora
model = Aurora(
        surf_vars=surf_vars,
        static_vars=static_vars,
        atmos_vars=atmos_vars,
        use_lora=False,
        autocast=True
    ).to(device)

# Carga de pesos preentrenados y ajustes del modelo
model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt", strict=False)
model.train()
model.configure_activation_checkpointing()
print("Modelo Aurora cargado y ajustado exitosamente.")





    

# ===================================
# Configuración del entrenamiento
# ===================================
print("Configurando el entrenamiento...")
batch_size = 3 # 8 # Tamaño de batch
sample_size = 3  # Tamaño de muestras (días)
num_epochs = 30 #15  # Número de épocas
latitudes = dataset['latitude'].values # Latitudes del dataset
criterion = torch.nn.L1Loss(reduction='none')  # Función de pérdida
optimizer = configure_optimizer(model, lr=1e-5) # Configuración del optimizador
print("Entrenamiento configurado.")

# ===================================
# Configuración para experimentos
# ===================================

# Experimento 2: Congelar la red excepto el decodificador
# Descomentar las siguientes líneas para habilitar este experimento

#for name, param in model.named_parameters():
   # param.requires_grad = False

#for name, param in model.decoder.named_parameters():
   # param.requires_grad = True

# Configuración del optimizador y función de pérdida
#optimizer_exp2 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
#criterion = torch.nn.L1Loss(reduction='none')

#############################################################################################################

# Experimento 3: Descongelar toda la red
# Descomentar las siguientes líneas para habilitar este experimento
# model.load_state_dict(torch.load('C://Users//Victor//deep_ocean//TFG_victor//saved_models//modelo_aurora_experimento_2.pth'))
# for param in model.parameters():
#     param.requires_grad = True  # Descongelar todos los parámetros


#optimizer = optim.AdamW(model.parameters(), lr=1e-5)
#criterion = torch.nn.L1Loss(reduction='none') 
#############################################################################################################


# ===================================
# Generadores de batches
# ===================================
print("Creando generadores de batches...")
train_generator = BatchGenerator(train_dataset, sample_size, batch_size, shuffle=True, padding=True)
val_generator = BatchGenerator(val_dataset, sample_size, batch_size, shuffle=False, padding=False)
print("Generadores de batches creados.")


# ===================================
# Entrenamiento del modelo
# ===================================
print("Iniciando el entrenamiento...")
train_losses, val_losses, val_rmses = train(
    model=model,
    train_generator=train_generator,
    val_generator=val_generator,
    num_epochs=num_epochs,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    latitudes=latitudes
)
print("Entrenamiento completado.")


# Guardar el modelo entrenado
model_save_path = "C://ruta_al_archivo//modelo_entrenado.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Modelo guardado en {model_save_path}")

# ===================================
# Visualización de resultados
# ===================================
print("Graficando las pérdidas de entrenamiento y validación...")
#plot_losses(train_losses, val_losses)

# ===================================
# Evaluación del modelo
# ===================================
print("Iniciando la evaluación en el conjunto de prueba...")
test_generator = BatchGenerator(test_dataset, sample_size=3, batch_size=3, shuffle=False, padding=False)


# Rutas de modelos para evaluación
model_paths = [
    'C://Users//ruta_al_archivo//aurora_exp1_bs3.pth',
    'C://Users//ruta_al_archivo//aurora_exp2_bs3.pth',
    'C://Users//ruta_al_archivo//aurora_exp3_bs3.pth',
    'C://Users//ruta_al_archivo//aurora_exp1_bs8.pth',
    'C://Users//ruta_al_archivo//aurora_exp2_bs8.pth',
    'C://Users//ruta_al_archivo//aurora_exp2_bs8.pth',
    ]




# Evaluación del modelo con los generadores y rutas especificadas
print("Iniciando la evaluación...")
evaluate_model(model, model_paths, test_generator, device, latitudes, batch_size)
print("Evaluación completada.")


# ===================================
# Configuración para predicciones a largo plazo
# ===================================

# El objetivo es realizar predicciones para un horizonte de 10 días (t+1, t+2, ..., t+10) en lugar de un único paso (t+2).
# Para ello, es necesario ajustar la configuración del dataset y los generadores para incluir estos pasos adicionales.

# 1. Ajustar el tamaño del sample_size:
# Actualmente, sample_size está configurado en 3 (2 días de entrada y 1 día como target).
# Para realizar predicciones de 10 días, sample_size debe incluir al menos 12 pasos temporales:
# 2 días de entrada (input) y 10 días de salida (target).
# Por lo tanto, se debe cambiar el sample_size a 12 al crear el test_generator:
# test_generator = BatchGenerator(test_dataset, sample_size=12, batch_size=batch_size, shuffle=False, padding=False)

# 2. Modificar la carga de datos para incluir los 10 pasos futuros:
# En las funciones de carga de datos como load_ocean_surface y load_ocean_atmos, ajustar el rango temporal.
# En lugar de isel(time=slice(2,None)), se debe usar isel(time=slice(2, 12)) para incluir 10 pasos futuros como target.

# 3. Configuración del batch_target:
# Actualmente, batch_target contiene un único paso futuro (t+2). Para predicciones a largo plazo, debe contener 10 pasos futuros.
# El target debe tener una forma (batch_size, 10, lat, lon) para representar las predicciones a lo largo de los 10 días.

# 4. Implementación del "rollout":
# Una vez configurados los datos y el generador, se utiliza una función de rollout para iterar las predicciones:
# - La función tomará los 2 días iniciales de entrada y generará una predicción para t+1.
# - Esta predicción se utilizará como entrada para generar t+2, y así sucesivamente hasta t+10.

# Nota:
# Estos ajustes aseguran que el dataset contiene la serie temporal completa y que el modelo es capaz de generar
# múltiples pasos de predicción a partir de un solo input inicial. Asegúrate de haber modificado el código de carga
# de datos y el generador de batches antes de ejecutar este tipo de predicciones.

