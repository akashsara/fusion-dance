import cv2 as cv
import numpy as np
import os
import sys
from tqdm import tqdm

src_dir = sys.argv[1]
mode = sys.argv[2]  

# Allowed modes
if mode not in ["erosion", "dilation", "opening", "closing", "gradient", "resize", "bilateral"]:
    print("Invalid mode!")
    sys.exit(0)

# Create target directory
if src_dir[-1] == "\\":
    base = src_dir.split("\\")[:-2]
else:
    base = src_dir.split("\\")[:-1]
target_dir = os.path.join(*base, "pixellated", mode)

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 4x4 Kernel of 1s
kernel = np.ones((4, 4), np.uint8)

for file in tqdm(os.listdir(src_dir)):
    src_path = os.path.join(src_dir, file)
    dest_path = os.path.join(target_dir, file)
    image = cv.imread(src_path)
    if mode == "erosion":
        image = cv.erode(image, kernel=kernel, iterations=1)
    elif mode == "dilation":
        image = cv.dilate(image, kernel=kernel, iterations=1)
    elif mode == "opening":
        image = cv.morphologyEx(image, op=cv.MORPH_OPEN, kernel=kernel)
    elif mode == "closing":
        image = cv.morphologyEx(image, op=cv.MORPH_CLOSE, kernel=kernel)
    elif mode == "gradient":
        image = cv.morphologyEx(image, op=cv.MORPH_GRADIENT, kernel=kernel)
    elif mode == "resize":
        image = cv.resize(image, dsize=(16, 16), interpolation=cv.INTER_AREA)
        image = cv.resize(image, dsize=(64, 64), interpolation=cv.INTER_CUBIC)
    elif mode == "bilateral":
        image = cv.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    result = cv.imwrite(dest_path, image)
    if not result:
        print(f"Didn't save file: {src_path}:{dest_path}")