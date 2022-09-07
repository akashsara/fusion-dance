"""
Disclaimer: This script contains older code and may not work as is.
This is a small script to scrape fusions from AlexOnsager.
Please be respectful of other people's websites and don't send tons of requests
per second.
"""
import requests
import time
from PIL import Image
from io import BytesIO
import os

base_url = "https://images.alexonsager.net/pokemon/fused/{base}/{base}.{fusion}.png"
output_dir = "alex_onsager_fusions/"

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

failed = []
for base in range(1, 152):
    for fusion in range(1, 152):
        save_dir = os.path.join(output_dir, f"{base}.{fusion}.png")
        if os.path.exists(save_dir):
            continue
        response = requests.get(base_url.format(base=base, fusion=fusion))
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            image.save(save_dir)
            print(f"{base}.{fusion}")
        else:
            failed.append(f"{base}.{fusion}")
        # Respect the website; don't DoS it
        time.sleep(1)

print("Failed Downloads:")
print(failed)