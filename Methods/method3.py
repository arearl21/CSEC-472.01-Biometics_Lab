import os
import re
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

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

# Compute MSE
def compute_mse(image1_path, image2_path):
    """
    Compute the refined MSE (Mean Squared Error) between two images.
    The images are first normalized to float32 and resized to the same shape if necessary.
    """
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    
    if image1 is None or image2 is None:
        raise ValueError(f"One of the images {image1_path} or {image2_path} could not be loaded.")
    
    # Resize images to the same shape if necessary
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Normalize the images to the range [0, 1] for higher precision
    image1 = image1.astype(np.float32) / 255.0
    image2 = image2.astype(np.float32) / 255.0

    # Compute Mean Squared Error (MSE)
    mse = np.mean((image1 - image2) ** 2)

    return mse


# Compare Images Using MSE
def compare_images_mse(image1_path, image2_path):
    """
    Compare two images using Mean Squared Error (MSE).
    """
    mse = compute_mse(image1_path, image2_path)
    similarity_score = 1 / (1 + mse)  # Inverse of MSE for similarity (lower MSE means higher similarity)
    return similarity_score

# Evaluate System
def evaluate_system_mse(data, thresholds):
    """
    Evaluate the system for False Accept Rate (FAR), False Reject Rate (FRR), and Equal Error Rate (EER) over multiple thresholds.
    """
    results = []
    num = 0
    for threshold in thresholds:
        false_accepts = 0
        false_rejects = 0
        genuine_matches = 0
        impostor_matches = 0

        for ref_path, sub_path in data:
            score = compare_images_mse(ref_path, sub_path)
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
        
        num += 1
        print("")
        print(str(num) + " done")
        print("")
        
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
    thresholds = np.linspace(0.94, 0.955, num=10)  # Adjust range as needed

    # Evaluate on TEST Set
    results_df = evaluate_system_mse(test_data, thresholds)

    # Summarize Results
    summary = results_df.describe().loc[["min", "max", "mean"]]
    print("Evaluation Summary (FAR, FRR, EER):")
    print(summary)
