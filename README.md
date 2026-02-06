# Gaze Prediction from FaceMesh and Pupil Position

This project predicts where a user is looking on a screen using **FaceMesh** facial landmarks and pupil positions captured via a webcam. The prediction is performed using a **LightGBM** model trained on collected gaze data. 

##  Project Structure

```
├── coordinate_prediction.ipynb # Jupyter notebook for experimentation and model training
├── dataset                     # Processed datasets for training
│   ├── face_landmarks_2025-12-03T16-34-50-227Z..csv
│   ├── face_landmarks_2025-12-03T16-55-54-791Z..csv
│   ├── face_landmarks_2025-12-15T09-27-27-405Z..csv
│   └── face_landmarks_2025-12-15T09-40-57-367Z..csv
│
├── example.json                # Example input file format
├── json                        # Raw data (FaceMesh + pupil coordinates + metadata)
│   ├── face_landmarks_2025-12-03T16-34-50-227Z.json
│   ├── face_landmarks_2025-12-03T16-55-54-791Z.json
│   ├── face_landmarks_2025-12-15T09-27-27-405Z.json
│   └── face_landmarks_2025-12-15T09-40-57-367Z.json
│
├── lightgbm_model_x.pkl        # LightGBM model for X-coordinate prediction
├── lightgbm_model_y.pkl        # LightGBM model for Y-coordinate prediction
├── optimal_df                  # Optimized datasets for model training
│   └── df.csv
│
└── predict_coordinate.py        # Main script for real-time prediction

```

## How to Use 
### Prerequisites Ensure you have the following libraries installed: 
```bash pip install numpy pandas lightgbm scikit-learn opencv-python```

### Run the Prediction

To predict the gaze coordinates from a JSON file (e.g., `example.json`):

`python predict_coordinate.py "example.json"`

### Input Format

The input JSON file must follow the structure defined in `example.json`, which includes:

- **FaceMesh landmarks** (468 3D coordinates).
- **Pupil positions** (left and right eye).
- **Window size** (resolution of the screen)
- **Predictions** (calibration points to detect the corners of the screen) 

## How It Works

1. **Data Preprocessing**: The input JSON is parsed and  the facemesh is transformed into features for the model
2. **Prediction**: Two LightGBM models (`lightgbm_model_x.pkl` and `lightgbm_model_y.pkl`) predict the X and Y coordinates of the user's gaze.
3. **Calibration Correction**: A post-processing step adjusts the prediction based on calibration data to adjust to user's screen, resolution and size)

## Data

- **Raw Data**: Located in the `json/` directory, containing FaceMesh, pupil coordinates and metadata.
- **Processed Data**: Available in `dataset/` and `optimal_df/`, generated from `coordinate_prediction.ipynb`.

## Training the Model

To retrain the models:

1. Open `coordinate_prediction.ipynb` in Jupyter Notebook.
2. Follow the steps to preprocess the data and train the LightGBM models.
3. Export the trained models as `.pkl` files.

## Notes

- The models were trained on data collected from a webcam using **MediaPipe FaceMesh**.
- For best results, ensure the input JSON structure matches `example.json`.
- Calibration improves accuracy but requires initial user-specific data.
