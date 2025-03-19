# Aurora Fine-Tuning with Oceanographic Data

This repository contains the code and resources used for the first fine-tuning of Microsoft's Aurora model using oceanographic data from the Copernicus Marine Service, as described in:

```
@mastersthesis{medina2025prediccion,
  title={Predicci{\'o}n de variables oceanogr{\'a}ficas basada en m{\'e}todos de aprendizaje profundo},
  author={Medina Morales, V{\'\i}ctor},
  type={{B.S.} thesis},
  year={2025}
}
```

## Project Overview

This project adapts the Aurora model - originally designed for atmospheric prediction tasks - to work with oceanographic data from the Copernicus Marine Service. We've specifically focused on predicting sea surface temperature (thetao) around the Canary Islands and surrounding Atlantic Ocean regions.

This represents a novel approach to oceanographic forecasting by leveraging state-of-the-art AI models and adapting them to marine prediction challenges.

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

We used the `cmems_mod_glo_phy_my_0.083deg_P1D-m` dataset from Copernicus Marine Service, which provides:
- Daily sea surface temperature data (thetao)
- Spatial resolution of 0.083 degrees
- Geographic coverage of the Canary Islands and surrounding Atlantic Ocean
- Temporal range from January 1, 2014 to January 1, 2021

## Methodology

### Data Processing
The `DatosCop.ipynb` notebook handles downloading data using the `copernicusmarine` Python API, preprocessing, normalization, and splitting into training, validation, and test sets.

### Model Architecture

The Aurora model is a deep learning architecture originally designed for atmospheric prediction. In this project, we've adapted it to work with oceanographic data, specifically sea surface temperature. The model architecture includes:

- A transformer-based architecture with specialized components:
  - Encoder modules for processing input data
  - Decoder modules for generating predictions
  - FiLM (Feature-wise Linear Modulation) layers for conditional computation
  - Fourier feature encoding for handling spatial coordinates
  - Position encoding mechanisms for spatial awareness
  - Patch embedding for handling grid-based data
  - Swin3D transformer architecture for spatial-temporal processing
  - Perceiver modules for improved attention mechanisms

- LoRA (Low-Rank Adaptation) fine-tuning approach:
  - Efficient parameter adjustment for domain adaptation
  - Preserves core model capabilities while adapting to new data domain
  - Reduces training time and computational requirements
  - Enables effective transfer learning from atmospheric to oceanographic prediction

This adaptation represents a novel application of state-of-the-art deep learning models to oceanographic forecasting.

### Training & Evaluation
The model was trained with carefully tuned hyperparameters and evaluated using metrics that account for geographical considerations:
- RMSE (Root Mean Squared Error) weighted by cosine of latitude
- Bias assessment weighted by cosine of latitude
- ACC (Anomaly Correlation Coefficient) weighted by cosine of latitude

This latitude-weighted approach ensures proper evaluation across different geographic regions, accounting for the distortion that occurs at higher latitudes.

## Getting Started

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Follow the workflow in `DatosCop.ipynb` to acquire and process the data:
   - This notebook handles the Copernicus Marine Service API connection
   - Processes and normalizes the sea surface temperature (thetao) data
   - Prepares data splits for model training
   
4. Run the training script with your desired parameters:
   ```bash
   python -m src.training.train
   ```
   
5. Evaluate model performance using the latitude-weighted metrics:
   ```bash
   python -m src.training.inference
   ```
   
The evaluation in `inference.py` uses visualization tools from `visualization.py` to compare predictions against ground truth, with all metrics properly weighted by the cosine of latitude to account for geographic distortion.

## Results

The fine-tuned Aurora model demonstrates promising results in oceanographic prediction tasks:

- Successfully adapted a climate model architecture to oceanographic variables
- Properly accounts for geographical considerations through latitude-weighted metrics
- Shows potential for further development in marine forecasting applications
- Provides a foundation for further research in cross-domain model adaptation

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

## Contributing

Contributions are welcome! If you'd like to contribute, please:
1. Fork the repository
2. Create a new branch for your feature
3. Add your changes
4. Submit a pull request

## Contact

Questions or suggestions? Reach out to:

victormedina2157@gmail.com
