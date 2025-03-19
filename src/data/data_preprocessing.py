import xarray as xr
import numpy as np
from aurora import normalisation

class DataPreprocessor:
    def __init__(self, variables, depth_levels=None):
        self.variables = variables
        self.depth_levels = depth_levels

    def adjust_longitudes(self, dataset: xr.Dataset) -> xr.Dataset:
        dataset = dataset.assign_coords(longitude=((dataset.longitude + 360) % 360))
        return dataset

    def adjust_latitudes(self, dataset: xr.Dataset) -> xr.Dataset:
        latitude = dataset['latitude'].values
        if not np.all(np.diff(latitude) < 0):
            dataset = dataset.sortby('latitude', ascending=False)
        return dataset

    def fill_nan_with_mean(self, dataset: xr.Dataset) -> xr.Dataset:
        for var in self.variables:
            data_var = dataset[var]
            if data_var.isnull().any():
                dataset[var] = data_var.fillna(data_var.mean())
        return dataset

    def add_season_attribute(self, dataset: xr.Dataset) -> xr.Dataset:
        def get_season(month):
            if month in [12, 1, 2]:
                return 'winter'
            elif month in [3, 4, 5]:
                return 'spring'
            elif month in [6, 7, 8]:
                return 'summer'
            else:
                return 'fall'

        dataset['season'] = xr.apply_ufunc(
            np.vectorize(get_season),
            dataset['time'].dt.month,
            vectorize=True,
            output_dtypes=[str]
        )
        return dataset

    def normalize_dataset(self, dataset: xr.Dataset) -> None:
        if self.depth_levels is None:
            raise ValueError("Depth levels must be provided for normalization.")

        for level in self.depth_levels:
            level_str = f"{level}"
            var = "thetao"
            data = dataset[var].sel(depth=level).values
            mean = np.nanmean(data)
            std = np.nanstd(data)
            normalisation.locations[f"{var}_{level_str}"] = mean
            normalisation.scales[f"{var}_{level_str}"] = std

        for var in self.variables:
            if 'depth' in dataset[var].dims:
                data = dataset[var].isel(depth=0).values
            else:
                data = dataset[var].values
            mean = np.nanmean(data)
            std = np.nanstd(data)
            normalisation.locations[var] = mean
            normalisation.scales[var] = std

    def split_by_time(self, dataset, train_ratio=0.7, val_ratio=0.15):
        num_times = len(dataset['time'])
        train_index = int(train_ratio * num_times)
        val_index = int(val_ratio * num_times)
        train_dataset = dataset.isel(time=slice(0, train_index))
        val_dataset = dataset.isel(time=slice(train_index, train_index + val_index))
        test_dataset = dataset.isel(time=slice(train_index + val_index, num_times))
        return train_dataset, val_dataset, test_dataset

    def preprocess_all(self, dataset: xr.Dataset) -> tuple:
        print("Ajustando logitudes...")
        dataset = self.adjust_longitudes(dataset)

        print("Ajustando latitudes...")
        dataset = self.adjust_latitudes(dataset)

        print("REllenando NaN con la media...")
        dataset = self.fill_nan_with_mean(dataset)

        print("Añadiendo la estación a la instancia...")
        dataset = self.add_season_attribute(dataset)

        print("Dividiendo el conjunto de datos en subconjuntos de entreno,validación y prueba...")
        train_dataset, val_dataset, test_dataset = self.split_by_time(dataset)

        print("normalizando el conjunto de entreno...")
        self.normalize_dataset(train_dataset)

        return train_dataset, val_dataset, test_dataset


