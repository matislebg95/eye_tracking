import sys
import json
import pickle
import lightgbm as lgb
import numpy    as np
import pandas   as pd

from joblib import load

NB_OF_KEYPOINTS = 468
LABELS = ["x_center",
          "y_center", 
          "z_center", 
          "alpha",
          "beta",
          "gamma", 
          "x_left_eyes",
          "y_left_eyes",
          "z_left_eyes",
          "x_right_eyes",
          "y_right_eyes",
          "z_right_eyes",
          "x_left_pupils",
          "y_left_pupils",
          "x_right_pupils",
          "y_right_pupils",
          "x_left_look_vector",
          "y_left_look_vector",
          "x_right_look_vector",
          "y_right_look_vector",
          "pupils_distance",
          "left_right_ratio",
          "face_distance"]

def get_center_and_vector(point_cloud, reference_vector = None):
    centroid = np.mean(point_cloud, axis = 0)
    centered = point_cloud - centroid
    _, _, transposed_right_singular_vectors = np.linalg.svd(centered)
    normal_vector = transposed_right_singular_vectors[-1, :]
    unit_normal_vector = normal_vector / np.linalg.norm(normal_vector)

    scalar_product = np.dot(unit_normal_vector, reference_vector)
    if reference_vector is not None and scalar_product > 0:
        unit_normal_vector = -unit_normal_vector

    return(centroid, unit_normal_vector)

def get_eye(point_cloud):
    left_eye_indexes  = np.array([263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249])
    right_eye_indexes = np.array([ 33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163,   7])
    n_points = len(right_eye_indexes)
    left_eye  = np.zeros((n_points, 3))
    right_eye = np.zeros((n_points, 3))
    for i in range(n_points):
        left_eye[i]  = point_cloud[left_eye_indexes[i]]
        right_eye[i] = point_cloud[right_eye_indexes[i]]

    return(left_eye, right_eye)

def get_eyes_centroid(left_eye, right_eye):
    left_eye_centroid  = np.mean(left_eye, axis=0)
    right_eye_centroid = np.mean(right_eye, axis=0)
    eyes_centroid = np.array((left_eye_centroid, right_eye_centroid))
    return(eyes_centroid)

def get_distance(center, centered_eye_centroid, centered_pupils):
    distance = np.zeros(3)
    centered_left_eye  = centered_eye_centroid[0]
    centered_right_eye = centered_eye_centroid[1]

    pupil_distance = np.linalg.norm(centered_pupils[0] - centered_pupils[1])
    eye_distance   = np.linalg.norm(centered_left_eye[:2] - centered_right_eye[2:])

    left_distance  = np.linalg.norm(centered_pupils[0] - centered_left_eye[:2])
    right_distance = np.linalg.norm(centered_pupils[1] - centered_right_eye[:2])

    distance[0] = pupil_distance / eye_distance
    distance[1] = left_distance / right_distance
    distance[2] = np.linalg.norm(center)

    return(distance)

def get_input_vector(point_cloud, pupils):
    reference_vector = np.array([0, 0, 1])
    center, vector = get_center_and_vector(point_cloud, reference_vector)
    centered_point_cloud = point_cloud - center
    centered_left_eye, centered_right_eye = get_eye(centered_point_cloud)
    centered_eyes_centroid = get_eyes_centroid(centered_left_eye, centered_right_eye)
    centered_pupils = pupils - center[:2]
    centered_eye_centroid_without_z = np.delete(centered_eyes_centroid.flatten(), [2, 5])
    look_vector = centered_pupils.flatten() - centered_eye_centroid_without_z.flatten()    
    distance = get_distance(center, centered_eyes_centroid, centered_pupils)
    input_vector = np.concatenate((center, 
                                   vector, 
                                   centered_eyes_centroid.flatten(), 
                                   centered_pupils.flatten(), 
                                   look_vector.flatten(), 
                                   distance))
    
    return(input_vector.reshape(1, -1))

def get_homothety_params(window_size, positions, predictions):

    #######################################
    #                                     #
    # Shape of positions and predictions: #
    #                                     #
    # x_0, y_0                   x_1, y_1 #
    #                                     #
    # x_2, y_2                   x_3, y_3 #
    #                                     #
    #######################################
    
    positions   = np.asarray(positions, dtype=float)
    predictions = np.asarray(predictions, dtype=float)

    if positions.shape != (4, 2) or predictions.shape != (4, 2):
        raise ValueError("positions and predictions have to be arrays of shape (4,2)")

    positions_centroid   = positions.mean(axis=0)
    predictions_centroid = predictions.mean(axis=0)

    translation             = positions_centroid - predictions_centroid
    centered_predictions    = predictions + translation
    centered_positions      = positions - positions_centroid 
    centered_preds_for_size = centered_predictions - positions_centroid 

    pred_row1_width     = centered_preds_for_size[1, 0] - centered_preds_for_size[0, 0]
    pred_row2_width     = centered_preds_for_size[3, 0] - centered_preds_for_size[2, 0]
    pred_mean_row_width = np.mean([pred_row1_width, pred_row2_width])

    pos_row1_width     = centered_positions[1, 0] - centered_positions[0, 0]
    pos_row2_width     = centered_positions[3, 0] - centered_positions[2, 0]
    pos_mean_row_width = np.mean([pos_row1_width, pos_row2_width])

    pred_col1_height     = centered_preds_for_size[2, 1] - centered_preds_for_size[0, 1]
    pred_col2_height     = centered_preds_for_size[3, 1] - centered_preds_for_size[1, 1]
    pred_mean_col_height = np.mean([pred_col1_height, pred_col2_height])

    pos_col1_height     = centered_positions[2, 1] - centered_positions[0, 1]
    pos_col2_height     = centered_positions[3, 1] - centered_positions[1, 1]
    pos_mean_col_height = np.mean([pos_col1_height, pos_col2_height])

    def safe_div(numer, denom):
        denom = float(denom)
        return(numer / denom if abs(denom) > 1e-8 else 1.0)

    x_scale = safe_div(pos_mean_row_width, pred_mean_row_width)
    y_scale = safe_div(pos_mean_col_height, pred_mean_col_height)

    return(positions_centroid, translation, x_scale, y_scale)


def homothety_correction(prediction, positions_centroid, translation, x_scale, y_scale, window_size):
    prediction = np.asarray(prediction, dtype=float)
    if prediction.shape != (2, 1):
        raise ValueError("prediction have to be array-like of shape (2,)")

    centered_prediction = prediction.flatten() + translation
    prediction_vector   = centered_prediction - positions_centroid

    prediction_scaled = np.array([prediction_vector[0] * x_scale,
                                  prediction_vector[1] * y_scale])

    corrected_prediction = positions_centroid + prediction_scaled

    resized_prediction = [corrected_prediction[0] * window_size[0],
                          corrected_prediction[1] * window_size[1]]

    return(resized_prediction)

def get_prediction(input_vector, window_size, predictions):
    positions = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])
    positions_centroid, translation, x_scale, y_scale = get_homothety_params(window_size, positions, predictions)

    with open('lightgbm_model_x.pkl', 'rb') as f:
        model_x = pickle.load(f)
    
    with open('lightgbm_model_y.pkl', 'rb') as f:
        model_y = pickle.load(f)
    
    
    input_vector = pd.DataFrame(input_vector, columns=LABELS) 
    
    prediction_x = model_x.predict(input_vector)
    prediction_y = model_y.predict(input_vector)

    prediction = [prediction_x, prediction_y]
    resized_prediction = homothety_correction(prediction, positions_centroid, translation, x_scale, y_scale, window_size)
    return(resized_prediction)

def main(json_file_name):
    with open(json_file_name, 'r') as f:
        data = json.load(f)

    landmarks   = np.array(data['landmarks'])
    pupil_left  = np.array(data['pupil_left'])
    pupil_right = np.array(data['pupil_right'])
    window_size = np.array(data['window_size'])
    predictions = np.array(data['predictions'])
    
    input_vector = get_input_vector(landmarks, [pupil_left, pupil_right])
    prediction   = get_prediction(input_vector, window_size, predictions)
    return(prediction)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
