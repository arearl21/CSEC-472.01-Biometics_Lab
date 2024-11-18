import os
import re
import cv2
import numpy as np
import pandas as pd
from scipy.spatial import KDTree


def compareM1(img1,img2):
    return


def compareM2(img1,img2):
    # Resize to the same dimensions (if necessary)
    img2 = cv2.resize(img2, img1.shape[::-1])

    # Calculate cross-correlation
    similarity = np.corrcoef(img1.ravel(), img2.ravel())[0, 1]
    return similarity < 0.04

def compareM3(img1,img2):
    return

#Returns True or False based on the three methods
def majority_voting(img1, img2):
    # Run all three methods
    decision1 = compareM1(img1,img2)
    decision2 = compareM2(img1,img2)
    decision3 = compareM3(img1,img2)

    # Collect decisions in a list
    decisions = [decision1, decision2, decision3]

    # Count occurrences of each decision
    from collections import Counter
    decision_counts = Counter(decisions)

    # Select the majority decision (most common)
    majority_decision = decision_counts.most_common(1)[0][0]  # Returns the most common decision

    return majority_decision

if __name__ == "__main__":
    dataset_path = "D:\\FIGS\\NISTSpecialDatabase4GrayScaleImagesofFIGS\\NISTSpecialDatabase4GrayScaleImagesofFIGS\\sd04\\png_txt"
    train_data, test_data = load_images(dataset_path)

    for ref_img, sub_img in data:
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
