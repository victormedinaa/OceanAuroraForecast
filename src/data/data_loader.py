import xarray as xr
import numpy as np

def load_dataset(dataset_path: str, variables: list, depth_slice: slice) -> xr.Dataset:
    """
    Carga el dataset y selecciona las variables y profundidad especificadas.

    Args:
        dataset_path (str): Ruta al archivo del dataset.
        variables (list): Lista de variables a seleccionar.
        depth_slice (slice): Slice para seleccionar los niveles de profundidad.

    Returns:
        xr.Dataset: Dataset cargado y filtrado.
    """
    dataset = xr.open_dataset(dataset_path)
    dataset = dataset[variables]
    dataset = dataset.isel(depth=depth_slice)
    return dataset

def load_lsm(lsm_path: str, dataset: xr.Dataset) -> xr.DataArray:
    """
    Carga la máscara de superficie terrestre (lsm) y la interpola al grid del dataset.

    Args:
        lsm_path (str): Ruta al archivo de la máscara lsm.
        dataset (xr.Dataset): Dataset al cual se interpolará la lsm.

    Returns:
        xr.DataArray: Máscara lsm interpolada.
    """
    lsm = xr.open_dataset(lsm_path)
    lsm_copy = lsm.copy()
    lsm_copy = lsm_copy.assign_coords(longitude=(((lsm_copy.longitude + 180) % 360) - 180))
    lsm_interp = lsm_copy.interp(latitude=dataset.latitude, longitude=dataset.longitude, method="nearest")
    lsm_interp_clean = lsm_interp.fillna(0)
    return lsm_interp_clean['lsm']
