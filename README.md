# Hybrid Multi-modality Multi-task Learning for Forecasting Progression Trajectories in Subjective Cognitive Decline
![framework_v16-1](https://github.com/user-attachments/assets/2e4e1442-cb19-4a8d-b9b6-e3b6f6c06915)


# Usage
1. **To Train the Model**  
   Execute the following command:  
   ```bash
   python main.py

2. **File Descriptions**
- `main.py`: The main script to train and evaluate the Hybrid Multi-Modality Learning (HMML) model. It integrates data loading, model construction, and training procedures.

- `model.py`: Contains the implementation of the Hybrid Multi-Modality Learning (HMML) model architecture.

- `data_loader.py`: Handles all aspects of data preparation, including:
  - Reading and preprocessing input data (performing image cropping and normalization).
  - Checking for the presence of PET images and utilizing synthetic PET data for subjects with missing PET scans.
  - Arranging batch data to ensure balanced representation of positive and negative labels within each batch.
  - Defining data loaders for training and evaluation.

- `util.py`: Provides utility functions, such as checking for GPU availability and displaying GPU details.
