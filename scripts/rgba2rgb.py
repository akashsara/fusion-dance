import os
import sys
from PIL import Image

file_dir = sys.argv[1]

for file in os.listdir(file_dir):
    filepath = os.path.join(file_dir, file)
    Image.open(filepath).convert('RGB').save(filepath)
