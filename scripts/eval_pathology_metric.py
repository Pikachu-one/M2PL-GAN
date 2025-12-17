import os
import cv2
import numpy as np
from scipy.stats import pearsonr

def calculate_pearson_r_for_folders(generated_folder, gt_folder, resize_dim=None):
    generated_files = sorted(os.listdir(generated_folder))
    gt_files = sorted(os.listdir(gt_folder))

    results = []

    for gen_file, gt_file in zip(generated_files, gt_files):
        gen_path = os.path.join(generated_folder, gen_file)
        gt_path = os.path.join(gt_folder, gt_file)

        gen_image = cv2.imread(gen_path, cv2.IMREAD_GRAYSCALE)
        gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        if gen_image is None or gt_image is None:
            continue

        if resize_dim:
            gen_image = cv2.resize(gen_image, resize_dim)
            gt_image = cv2.resize(gt_image, resize_dim)

        gen_flat = gen_image.flatten()
        gt_flat = gt_image.flatten()

        corr, _ = pearsonr(gen_flat, gt_flat)
        results.append(corr)

    return np.array(results).mean()




gt_folder = "XXXXX"

fake_folders = {

#"MODEL1":'XXXXXX',
"MODEL2":'XXXXX',
"MODEL3":'XXXXX',
"MODEL4":'XXXXX',}

resize_dim = (1024, 1024)

# ------------------------ 运行计算 ------------------------

print("\n====== Pearson-R Evaluation ======\n")
for name, fake_path in fake_folders.items():
    score = calculate_pearson_r_for_folders(fake_path, gt_folder, resize_dim)
    print(f"{name:20s}:  Pearson-R = {score:.6f}")
