import os
import re
import cv2
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from concurrent.futures import ThreadPoolExecutor


# Load Dataset
def load_images(dataset_path, train_split=1500):
    train_data, test_data = [], []
    
    for dir_index in range(8):
        dir_path = os.path.join(dataset_path, f"figs_{dir_index}")
        
        if not os.path.exists(dir_path):
            print(f"Directory {dir_path} does not exist.")
            continue

        files = os.listdir(dir_path)
        ref_files = [f for f in files if re.match(r'^f\d{4}(_\d{2})?\.png$', f)]

        for ref_file in ref_files:
            file_index = ref_file.split('_')[0].replace('f','')  # e.g., '0001', '0002'
            sub_file = ref_file.replace('f', 's')
            ref_img_path = os.path.join(dir_path, ref_file)
            sub_img_path = os.path.join(dir_path, sub_file)

            if os.path.exists(ref_img_path) and os.path.exists(sub_img_path):
                ref_img = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)
                sub_img = cv2.imread(sub_img_path, cv2.IMREAD_GRAYSCALE)
                if int(file_index) <= train_split:
                    train_data.append((ref_img, sub_img))
                else:
                    test_data.append((ref_img, sub_img))

    return train_data, test_data


# Preprocess Image: Skeletonization
def preprocess_image(image):
    if image is None:
        raise ValueError("Image not loaded properly. Please check the file path.")
    
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    skeleton = cv2.ximgproc.thinning(binary_image)  # Thinning function to obtain skeleton
    return skeleton


# Minutiae Detection (optimized)
def detect_minutiae(skeleton):
    minutiae_points = []
    rows, cols = skeleton.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if skeleton[i, j] == 255:
                neighborhood = skeleton[i - 1:i + 2, j - 1:j + 2]
                crossing_number = np.sum(neighborhood) // 255
                if crossing_number == 2:
                    minutiae_points.append((i, j, "termination"))
                elif crossing_number > 2:
                    minutiae_points.append((i, j, "bifurcation"))
    return minutiae_points


# Minutiae Matching with KDTree Optimization
def compare_minutiae(minutiae1, minutiae2, max_distance=20):
    coords1 = [(point[0], point[1]) for point in minutiae1]
    coords2 = [(point[0], point[1]) for point in minutiae2]

    tree1 = KDTree(coords1)
    distances = []

    # Search for nearest neighbors within max_distance
    for point2 in coords2:
        dist, idx = tree1.query(point2, distance_upper_bound=max_distance)
        if dist < max_distance and minutiae1[idx][2] == minutiae2[coords2.index(point2)][2]:
            distances.append(dist)
    return np.mean(distances) if distances else float('inf')


# Efficient Evaluation: FAR, FRR, and EER with Parallelism
def evaluate_system(data, thresholds):
    results = []

    # Use ThreadPoolExecutor to parallelize the evaluation
    with ThreadPoolExecutor() as executor:
        futures = []
        for threshold in thresholds:
            futures.append(executor.submit(evaluate_threshold, data, threshold))

        for future in futures:
            results.append(future.result())

    return pd.DataFrame(results)


def evaluate_threshold(data, threshold):
    false_accepts = 0
    false_rejects = 0
    genuine_matches = 0
    impostor_matches = 0

    for ref_img, sub_img in data:
        ref_skeleton = preprocess_image(ref_img)
        sub_skeleton = preprocess_image(sub_img)
        ref_minutiae = detect_minutiae(ref_skeleton)
        sub_minutiae = detect_minutiae(sub_skeleton)
        score = compare_minutiae(ref_minutiae, sub_minutiae)

        if score < threshold:
            if ref_img is sub_img:  # Genuine match
                genuine_matches += 1
            else:  # Impostor match
                false_accepts += 1
        else:
            if ref_img is sub_img:  # Genuine match rejected
                false_rejects += 1
            else:  # Impostor match correctly rejected
                impostor_matches += 1

    far = false_accepts / (false_accepts + impostor_matches) if (false_accepts + impostor_matches) > 0 else 0
    frr = false_rejects / (false_rejects + genuine_matches) if (false_rejects + genuine_matches) > 0 else 0
    eer = far if np.abs(far - frr) < 0.01 else (far + frr) / 2

    return {
        "Threshold": threshold,
        "FAR": far,
        "FRR": frr,
        "EER": eer
    }


# Main Execution
if __name__ == "__main__":
    dataset_path = "/home/kali/fingerprints/sd04/png_txt"
    train_data, test_data = load_images(dataset_path)

    # Define threshold range for evaluation
    thresholds = np.arange(1, 2, 0.1)  # Modify as needed

    # Step 1: Evaluate on TEST Set with Parallel Execution
    results_df = evaluate_system(test_data, thresholds)

    # Step 2: Summarize Results
    summary = results_df.describe().loc[["min", "max", "mean"]]
    print("Evaluation Summary (FAR, FRR, EER):")
    print(summary)
