# Aurora Fine-Tuning with Oceanographic Data

This repository contains the code and resources for the fine-tuning of Microsoft's Aurora model with oceanographic data from the Copernicus Marine Service. This work is detailed in the following paper:

```
@article{medina2025leveraging,
  title={Leveraging an Atmospheric Foundational Model for Subregional Sea Surface Temperature Forecasting},
  author={Medina, V{\'\i}ctor and Cuervo-Londo{\~n}o, Giovanny A and S{\'a}nchez, Javier},
  journal={arXiv preprint arXiv:2510.25563},
  year={2025}
}
```
**Link to the paper:** http://arxiv.org/abs/2510.25563

## Project Overview

This project explores the adaptation of the Aurora model, originally designed for atmospheric forecasting, for oceanographic data applications. The primary focus is on predicting sea surface temperature (thetao) in the region of the Canary Islands and the surrounding Atlantic Ocean.

This represents a novel approach to oceanographic forecasting, leveraging a state-of-the-art AI model and adapting it to the unique challenges of marine prediction.

## Repository Structure

```
aiis-group-aurora_fine_tuning/
├── README.md
├── DatosCop.ipynb        # Data acquisition and processing notebook
├── LICENSE.txt           # MIT License
├── main.py               # Main execution script
├── requirements.txt      # Project dependencies
├── src/                  # Source code
│   ├── data/             # Data processing modules
│   ├── models/aurora/    # Aurora model implementation and adaptations
│   ├── training/         # Training and inference scripts
│   └── utils/            # Utility functions
└── tests/                # Testing modules
```

## Data

The data used for this project is the `cmems_mod_glo_phy_my_0.083deg_P1D-m` dataset from the Copernicus Marine Service. It provides:
- Daily sea surface temperature data (thetao)
- A spatial resolution of 0.083 degrees
- Geographic coverage of the Canary Islands and the surrounding Atlantic Ocean
- A temporal range from January 1, 2014, to January 1, 2021

## Methodology

### Data Processing
The `DatosCop.ipynb` notebook manages the data pipeline. It uses the `copernicusmarine` Python API for downloading data, followed by preprocessing, normalization, and splitting into training, validation, and test sets.

### Model Architecture

The Aurora model is a powerful deep learning architecture adapted here for oceanographic data. The model is a sophisticated transformer-based architecture with several key components:

- **Core Architecture:**
  - Encoder and decoder modules for processing inputs and generating predictions.
  - FiLM (Feature-wise Linear Modulation) layers for conditional computation.
  - Fourier feature encoding for handling spatial coordinates.
  - Position encoding mechanisms to maintain spatial awareness.
  - A Swin3D transformer architecture for integrated spatio-temporal processing.
  - Perceiver modules for an efficient attention mechanism.

- **Fine-Tuning Approach:**
  - A **LoRA (Low-Rank Adaptation)** fine-tuning strategy was employed.
  - This method allows for highly efficient parameter adjustment for domain adaptation.
  - It preserves the core capabilities of the pre-trained model while adapting it to the new oceanographic data domain.
  - This significantly reduces training time and computational requirements, enabling effective transfer learning.

This adaptation represents a novel application of advanced deep learning models to the field of oceanographic forecasting.

### Training & Evaluation
The model was trained with tuned hyperparameters and evaluated using metrics that account for geographical considerations:
- RMSE (Root Mean Squared Error) weighted by the cosine of latitude.
- Bias assessment weighted by the cosine of latitude.
- ACC (Anomaly Correlation Coefficient) weighted by the cosine of latitude.

This latitude-weighted approach is crucial for an accurate evaluation across different geographic regions, correcting for the grid cell distortion that occurs at higher latitudes.

## Getting Started

1.  Clone this repository.
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Execute the `DatosCop.ipynb` notebook to acquire and process the data.
    - This notebook handles the connection to the Copernicus Marine Service API.
    - It processes and normalizes the sea surface temperature (thetao) data.
    - It prepares the data splits for model training.
4.  Run the training script with desired parameters:
    ```bash
    python -m src.training.train
    ```
5.  Evaluate the model's performance using the latitude-weighted metrics:
    ```bash
    python -m src.training.inference
    ```
The evaluation script, `inference.py`, utilizes visualization tools to compare model predictions against the ground truth, with all metrics properly weighted by the cosine of latitude to account for geographic distortion.

## Results

The fine-tuned Aurora model demonstrates strong performance in oceanographic prediction tasks:

- Successfully adapted a foundational climate model architecture for oceanographic variables.
- Achieved a low Root Mean Squared Error (RMSE) of **0.119 K** and a high Anomaly Correlation Coefficient (ACC) of approximately **0.997**, indicating high accuracy and reliability.
- Effectively reproduces large-scale SST structures, though challenges remain in capturing finer details in complex coastal regions.
- Provides a solid foundation for future research in cross-domain model adaptation and shows significant potential for further development in marine forecasting applications.

## License

This project is licensed under the MIT License. Please see the [LICENSE.txt](LICENSE.txt) file for details.

## Contributing

Contributions are welcome. If you would like to contribute, please follow these steps:
1.  Fork the repository.
2.  Create a new branch for your feature.
3.  Commit your changes.
4.  Submit a pull request.

## Contact

For questions or suggestions, please feel free to reach out at:

victormedina2157@gmail.com
