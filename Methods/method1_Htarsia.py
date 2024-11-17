import os
import re
import cv2
import numpy as np
import pandas as pd

# Load Dataset
def load_images(dataset_path, train_split=1500):
    """
    Load images from multiple directories (figs_0, figs_1, ..., figs_7) and split into TRAIN and TEST sets.
    """
    train_data, test_data = [], []
    
    # Loop through each figs_X directory (where X is 0-7)
    for dir_index in range(8):
        dir_path = os.path.join(dataset_path, f"figs_{dir_index}")
        
        # Ensure the directory exists
        if not os.path.exists(dir_path):
            print(f"Directory {dir_path} does not exist.")
            continue

        # List all files in the directory
        files = os.listdir(dir_path)

        # Filter for files starting with 'f' and ending with '.png'
        ref_files = [f for f in files if re.match(r'^f\d{4}(_\d{2})?\.png$', f)]

        # Process each matching file
        for ref_file in ref_files:
            # Extract the index part of the filename (e.g., 0001, 0002, 0010, etc.)
            file_index = ref_file.split('_')[0]  # This will give you '0001', '0002', etc.

            # Create corresponding sub image filename by replacing 'f' with 's'
            sub_file = ref_file.replace('f', 's')

            # Construct full file paths
            ref_img_path = os.path.join(dir_path, ref_file)
            sub_img_path = os.path.join(dir_path, sub_file)

            # Check if both reference and subject files exist
            if os.path.exists(ref_img_path) and os.path.exists(sub_img_path):
                # Split into train and test sets based on train_split threshold
                if int(file_index) <= train_split:
                    train_data.append((ref_img_path, sub_img_path))
                else:
                    test_data.append((ref_img_path, sub_img_path))

    return train_data, test_data

# Preprocess Image: Skeletonization
import cv2
import numpy as np

def preprocess_image(image):
    """
    Preprocess the fingerprint image: binarize and skeletonize.
    """
    if image is None:
        raise ValueError("Image not loaded properly. Please check the file path.")
    
    # Ensure the image is in grayscale
    if len(image.shape) != 2:
        raise ValueError("Image must be grayscale.")
    
    # Threshold to binary
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # Skeletonize the binary image using OpenCV's thinning function (if available)
    skeleton = cv2.ximgproc.thinning(binary_image)
    
    return skeleton

# Example usage
image_path = "path_to_your_image.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded correctly
if image is None:
    print("Error: Could not load image.")
else:
    skeleton_image = preprocess_image(image)
    cv2.imshow("Skeleton", skeleton_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Minutiae Detection
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

# Minutiae Matching
def compare_minutiae(minutiae1, minutiae2):
    distances = []
    for point1 in minutiae1:
        for point2 in minutiae2:
            if point1[2] == point2[2]:  
                distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
                distances.append(distance)
    return np.mean(distances) if distances else float('inf')

# Evaluate Performance: FAR, FRR, and EER
def evaluate_system(data, thresholds):
    """
    Evaluate the system for False Accept Rate (FAR), False Reject Rate (FRR), and Equal Error Rate (EER) over multiple thresholds.
    """
    results = []
    for threshold in thresholds:
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
        
        results.append({
            "Threshold": threshold,
            "FAR": far,
            "FRR": frr,
            "EER": eer
        })
    
    return pd.DataFrame(results)

# Main Execution
if __name__ == "__main__":
    dataset_path = "/home/kali/fingerprints/sd04/png_txt"
    train_data, test_data = load_images(dataset_path)

    # Step 1: Set thresholds to evaluate
    thresholds = np.arange(1, 51, 1)  # You can change this range as needed

    # Step 2: Evaluate on TEST Set
    results_df = evaluate_system(test_data, thresholds)

    # Step 3: Summarize Results
    summary = results_df.describe().loc[["min", "max", "mean"]]
    print("Evaluation Summary (FAR, FRR, EER):")
    print(summary)

    # Optional: Save to CSV
    results_df.to_csv('frr_far_eer_results.csv', index=False)
