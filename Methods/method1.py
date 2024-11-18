import os
import re
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load Dataset
def load_images(dataset_path, train_split=1500):
    """
    Load images from multiple directories (figs_0, figs_1, ..., figs_7) and split into TRAIN and TEST sets.
    """
    train_data, test_data = [], []
    
    for dir_index in range(8):
        dir_path = os.path.join(dataset_path, f"figs_{dir_index}")
        if not os.path.exists(dir_path):
            print(f"Directory {dir_path} does not exist.")
            continue

        files = os.listdir(dir_path)
        ref_files = [f for f in files if re.match(r'^f\d{4}(_\d{2})?\.png$', f)]

        for ref_file in ref_files:
            file_index = ref_file.split('_')[0].replace('f', '')
            sub_file = ref_file.replace('f', 's')

            ref_img_path = os.path.join(dir_path, ref_file)
            sub_img_path = os.path.join(dir_path, sub_file)

            if os.path.exists(ref_img_path) and os.path.exists(sub_img_path):
                if int(file_index) <= train_split:
                    train_data.append((ref_img_path, sub_img_path))
                else:
                    test_data.append((ref_img_path, sub_img_path))

    return train_data, test_data

# Extract Features Using SIFT
def extract_sift_features(image_path):
    """
    Extract keypoints and descriptors using SIFT.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")
    
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    
    if descriptors is None:
        return np.zeros((1, 128))  # Return a zero feature if no descriptors are found
    return descriptors

# Compare Images Using SIFT Features
def compare_images_sift(image1_path, image2_path):
    """
    Compare two images using SIFT descriptors and cosine similarity.
    """
    features1 = extract_sift_features(image1_path)
    features2 = extract_sift_features(image2_path)
    
    # Compute the mean descriptor for each image
    mean_feature1 = np.mean(features1, axis=0).reshape(1, -1)
    mean_feature2 = np.mean(features2, axis=0).reshape(1, -1)
    
    similarity = cosine_similarity(mean_feature1, mean_feature2)[0][0]
    return similarity

# Evaluate System
def evaluate_system_sift(data, thresholds):
    """
    Evaluate the system for False Accept Rate (FAR), False Reject Rate (FRR), and Equal Error Rate (EER) over multiple thresholds.
    """
    results = []
    for threshold in thresholds:
        false_accepts = 0
        false_rejects = 0
        genuine_matches = 0
        impostor_matches = 0

        for ref_path, sub_path in data:
            score = compare_images_sift(ref_path, sub_path)
            print(score)

            if score >= threshold:
                if ref_path == sub_path:  # Genuine match
                    genuine_matches += 1
                else:  # Impostor match
                    false_accepts += 1
            else:
                if ref_path == sub_path:  # Genuine match rejected
                    false_rejects += 1
                else:  # Impostor match correctly rejected
                    impostor_matches += 1
        print("another down")
        far = false_accepts / (false_accepts + impostor_matches) if (false_accepts + impostor_matches) > 0 else 0
        frr = false_rejects / (false_rejects + genuine_matches) if (false_rejects + genuine_matches) > 0 else 0
        eer = far if np.abs(far - frr) < 0.01 else (far + frr) / 2
        
        results.append({
            "Threshold": threshold,
            "FAR": far,
            "FRR": frr,
            "EER": eer
        })
    
    return pd.DataFrame(results)

# Main Execution
if __name__ == "__main__":
    # Dataset path
    dataset_path = "C:\\Users\\Owner\\Documents\\2024 fall class notes\\Authentication\\NISTSpecialDatabase4GrayScaleImagesofFIGS\\NISTSpecialDatabase4GrayScaleImagesofFIGS\\sd04\\png_txt"
    train_data, test_data = load_images(dataset_path)

    # Set thresholds for evaluation
    thresholds = np.linspace(.90, .95, num=10)  # Adjust range as needed

    # Evaluate on TEST Set
    results_df = evaluate_system_sift(test_data, thresholds)

    # Summarize Results
    summary = results_df.describe().loc[["min", "max", "mean"]]
    print("Evaluation Summary (FAR, FRR, EER):")
    print(summary)
