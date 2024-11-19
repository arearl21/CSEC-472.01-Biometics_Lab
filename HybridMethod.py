import os
import re
import cv2
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.metrics.pairwise import cosine_similarity

# Load Dataset - This will be used in all methods
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
            file_index = ref_file.split('_')[0].replace('f','')  # This will give you '0001', '0002', etc.

            # Create corresponding sub image filename by replacing 'f' with 's'
            sub_file = ref_file.replace('f', 's')

            # Construct full file paths
            ref_img_path = os.path.join(dir_path, ref_file)
            sub_img_path = os.path.join(dir_path, sub_file)

            # Check if both reference and subject files exist
            if os.path.exists(ref_img_path) and os.path.exists(sub_img_path):
                ref_img = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)
                sub_img = cv2.imread(sub_img_path, cv2.IMREAD_GRAYSCALE)
                # Split into train and test sets based on train_split threshold
                if int(file_index) <= train_split:
                    train_data.append((ref_img, sub_img))
                else:
                    test_data.append((ref_img, sub_img))

    return train_data, test_data

#======== Method 1 ========

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

# Match Descriptors Using FLANN
def match_descriptors(descriptors1, descriptors2):
    """
    Match descriptors between two images using FLANN-based matcher.
    """
    # FLANN parameters
    index_params = dict(algorithm=1, trees=10)  # FLANN-based index parameters (KDTREE)
    search_params = dict(checks=50)  # Search parameters (number of times the trees are traversed)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    
    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:  # Lowe's ratio test threshold
            good_matches.append(m)
    
    return good_matches

#======== Method 2 ========

# Preprocess Image: Skeletonization
def preprocess_image(image):
    """
    Preprocess the fingerprint image: binarize and skeletonize.
    """
    if image is None:
        raise ValueError("Image not loaded properly. Please check the file path.")
    
    # Threshold to binary
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # Skeletonize the binary image using OpenCV's thinning function (if available)
    skeleton = cv2.ximgproc.thinning(binary_image)
    
    return skeleton

#======== Method 3 ========

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

#======== Compare Each Method ========

def compare_m1(img1,img2):
    """
    Compare two images using SIFT descriptors and FLANN-based matching.
    """
    keypoints1, descriptors1 = extract_sift_features(img1)
    keypoints2, descriptors2 = extract_sift_features(img2)
    
    if len(descriptors1) == 0 or len(descriptors2) == 0:
        return 0.0  # If no descriptors found, return similarity 0
    
    good_matches = match_descriptors(descriptors1, descriptors2)
    
    # Calculate similarity score as the ratio of good matches to total keypoints
    similarity_score = len(good_matches) / min(len(keypoints1), len(keypoints2))
    return similarity_score


def compare_m2(img1,img2):
    # Resize to the same dimensions (if necessary)
    img2 = cv2.resize(img2, img1.shape[::-1])

    # Calculate cross-correlation
    similarity = np.corrcoef(img1.ravel(), img2.ravel())[0, 1]
    return similarity < 0.04

def compare_m3(img1,img2):
    """
    Compare two images using Mean Squared Error (MSE).
    """
    mse = compute_mse(img1, img2)
    similarity_score = 1 / (1 + mse)  # Inverse of MSE for similarity (lower MSE means higher similarity)
    return similarity_score

#Returns True or False based on the three methods
def majority_voting(img1, img2):
    # Run all three methods
    decision1 = compare_m1(img1,img2)
    decision2 = compare_m2(img1,img2)
    decision3 = compare_m3(img1,img2)

    # Collect decisions in a list
    decisions = [decision1, decision2, decision3]

    # Count occurrences of each decision
    from collections import Counter
    decision_counts = Counter(decisions)

    # Select the majority decision (most common)
    majority_decision = decision_counts.most_common(1)[0][0]  # Returns the most common decision

    return majority_decision

if __name__ == "__main__":
    #Change Based on user
    dataset_path = "C:\\Users\\Jacob Patterson\\College Word Documents\\Year 5\\472_Lab4\\NISTSpecialDatabase4GrayScaleImagesofFIGS\\NISTSpecialDatabase4GrayScaleImagesofFIGS\\sd04\\png_txt"
    train_data, test_data = load_images(dataset_path)

    #Get images
    for ref_img, sub_img in test_data:
        ref_skeleton = preprocess_image(ref_img)
        sub_skeleton = preprocess_image(sub_img)
        score = majority_voting(ref_skeleton, sub_skeleton)

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
    
    results_df = pd.DataFrame(results)

    summary = results_df.describe().loc[["min", "max", "mean"]]
    print("Evaluation Summary (FAR, FRR, EER):")
    print(summary)
