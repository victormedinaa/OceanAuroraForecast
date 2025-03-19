import numpy as np
import torch
from src.models.aurora import Batch, Metadata
from datetime import datetime

class BatchGenerator:
    def __init__(self, dataset, sample_size, batch_size, shuffle=True, padding=True):
        self.dataset = dataset
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.padding = padding
        self.samples = self.generate_sliding_windows()
        if self.shuffle:
            self.samples = self.shuffle_samples()
        print(f"BatchGenerator inicializado con {len(self.samples)} muestras.")

    def generate_sliding_windows(self):
        window_size = self.sample_size
        windows = [slice(i, i + window_size) for i in range(0, len(self.dataset.time) - window_size + 1)]
        samples = [self.dataset.isel(time=w) for w in windows]
        return samples

    def shuffle_samples(self):
        samples_copy = self.samples.copy()
        np.random.shuffle(samples_copy)
        return samples_copy

    def load_ocean_batch(self, sample_sets):
        is_padding = any(sample.attrs.get('is_padding', False) for sample in sample_sets)

    # Llamar a las funciones de carga
        surf_data, surf_target = self.load_ocean_surface("thetao", sample_sets)
        atmos_data, atmos_target = self.load_ocean_atmos("thetao", sample_sets)
        static_data = self.load_static_var("lsm", sample_sets)

        times = [
        sample_set['time'].values[-1].astype('datetime64[s]').astype(datetime)
        for sample_set in sample_sets
        ]

        batch = Batch(
            surf_vars={"thetao": surf_data},
            static_vars={"lsm": static_data},
            atmos_vars={"thetao": atmos_data},
            metadata=Metadata(
                lat=torch.from_numpy(self.dataset['latitude'].values).float(),
                lon=torch.from_numpy(self.dataset['longitude'].values).float(),
                time=times,
                atmos_levels=self.dataset['depth'].values,
            )
        )
        batch.metadata.is_padding = is_padding

        batch_target = Batch(
        surf_vars={"thetao": surf_target},
        static_vars={"lsm": static_data},
        atmos_vars={"thetao": atmos_target},
        metadata=batch.metadata
        )
        batch_target.metadata.is_padding = is_padding

        return batch, batch_target

    def load_ocean_surface(self, v, sample_sets):
        data_list = []
        target_list = []
        for sample_set in sample_sets:
            sel_dict = {}
            if 'depth' in sample_set[v].dims:
                sel_dict['depth'] = 0
            data = sample_set[v].isel(**sel_dict).isel(time=slice(0, 2)).values
            data_tensor = torch.from_numpy(data).float()
            data_list.append(data_tensor)

            target = sample_set[v].isel(**sel_dict).isel(time=slice(2, None)).values
            target_tensor = torch.from_numpy(target).float()
            target_list.append(target_tensor)

        data_batch = torch.stack(data_list, dim=0)
        target_batch = torch.stack(target_list, dim=0)
        return data_batch, target_batch

    def load_ocean_atmos(self, v, sample_sets):
        data_list = []
        target_list = []
        for sample_set in sample_sets:
            sel_dict = {'depth': slice(0, 10)}
            data = sample_set[v].isel(**sel_dict).isel(time=slice(0, 2)).values
            data_tensor = torch.from_numpy(data).float()
            data_list.append(data_tensor)

            target = sample_set[v].isel(**sel_dict).isel(time=slice(2, None)).values
            target_tensor = torch.from_numpy(target).float()
            target_list.append(target_tensor)

        data_batch = torch.stack(data_list, dim=0)
        target_batch = torch.stack(target_list, dim=0)
        return data_batch, target_batch

    def load_static_var(self, v, sample_sets):
        sample_set = sample_sets[0]
        data_var = sample_set[v]
        dims_to_drop = [dim for dim in data_var.dims if dim not in ('latitude', 'longitude')]
        data_var = data_var.isel({dim: 0 for dim in dims_to_drop})
        data = data_var.values
        data_tensor = torch.from_numpy(data).float()
        return data_tensor

    def __iter__(self):
        for i in range(0, len(self.samples), self.batch_size):
            batch_samples = self.samples[i:i + self.batch_size]

            if len(batch_samples) < self.batch_size and self.padding:
                num_padding = self.batch_size - len(batch_samples)
                for _ in range(num_padding):
                    sample = self.samples[i % len(self.samples)]
                    sample = sample.copy()
                    sample.attrs['is_padding'] = True
                    batch_samples.append(sample)

            batch, batch_target = self.load_ocean_batch(batch_samples)
            yield batch, batch_target
