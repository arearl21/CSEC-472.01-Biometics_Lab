import os
import cv2
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

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

        # Loop through the files in the current figs_X directory
        for i in range(1, 2001):  # Assuming you have 2000 pairs of images
            ref_img_path = os.path.join(dir_path, f"f{i:04d}.png")
            sub_img_path = os.path.join(dir_path, f"s{i:04d}.png")

            # Check if both files exist
            if not os.path.exists(ref_img_path) or not os.path.exists(sub_img_path):
                continue

            ref_img = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)
            sub_img = cv2.imread(sub_img_path, cv2.IMREAD_GRAYSCALE)
            
            # Split into train and test sets based on train_split threshold
            if i <= train_split:
                train_data.append((ref_img, sub_img))
            else:
                test_data.append((ref_img, sub_img))

    return train_data, test_data

# Preprocess Image: Skeletonization
def preprocess_image(image):
    """
    Preprocess the fingerprint image: binarize and skeletonize.
    """
    # Threshold to binary
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # Skeletonize the binary image
    skeleton = cv2.ximgproc.thinning(binary_image)
    return skeleton

# Minutiae Detection
def detect_minutiae(skeleton):
    """
    Detect minutiae points in the skeleton image.
    """
    minutiae_points = []
    rows, cols = skeleton.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if skeleton[i, j] == 255:  # Check for white pixel
                neighborhood = skeleton[i - 1:i + 2, j - 1:j + 2]
                crossing_number = np.sum(neighborhood) // 255
                if crossing_number == 2:  # Ridge ending
                    minutiae_points.append((i, j, "termination"))
                elif crossing_number > 2:  # Bifurcation
                    minutiae_points.append((i, j, "bifurcation"))
    return minutiae_points

# Minutiae Matching
def compare_minutiae(minutiae1, minutiae2):
    """
    Compare minutiae sets of two images using Euclidean distance.
    """
    distances = []
    for point1 in minutiae1:
        for point2 in minutiae2:
            if point1[2] == point2[2]:  # Match based on type
                distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
                distances.append(distance)
    return np.mean(distances) if distances else float('inf')

# Evaluate Performance
def evaluate_system(data, threshold):
    """
    Evaluate the system for False Accept Rate, False Reject Rate, and EER.
    """
    true_labels = []
    scores = []
    for ref_img, sub_img in data:
        ref_skeleton = preprocess_image(ref_img)
        sub_skeleton = preprocess_image(sub_img)
        ref_minutiae = detect_minutiae(ref_skeleton)
        sub_minutiae = detect_minutiae(sub_skeleton)
        score = compare_minutiae(ref_minutiae, sub_minutiae)
        scores.append(score)
        true_labels.append(1 if score < threshold else 0)
    
    # Calculate ROC Curve
    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
    return fpr, tpr, eer

# Main Execution
if __name__ == "__main__":
    # Step 1: Load Dataset
    dataset_path = "/home/kali/fingerprints/sd04/png_txt"
    train_data, test_data = load_images(dataset_path)

    # Step 2: Process Train Data
    threshold = 30  # Initial threshold for decision
    print("Processing TRAIN data...")
    for ref_img, sub_img in train_data:
        ref_skeleton = preprocess_image(ref_img)
        sub_skeleton = preprocess_image(sub_img)
        ref_minutiae = detect_minutiae(ref_skeleton)
        sub_minutiae = detect_minutiae(sub_skeleton)
        _ = compare_minutiae(ref_minutiae, sub_minutiae)  # Extract scores
    
    # Step 3: Evaluate on TEST Set
    print("Evaluating system on TEST data...")
    fpr, tpr, eer = evaluate_system(test_data, threshold)
    
    # Step 4: Visualize Results
    plt.plot(fpr, tpr, label=f"EER: {eer:.2f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    print(f"Equal Error Rate (EER): {eer:.4f}")
