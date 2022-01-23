import os
from PIL import Image
import numpy as np
import sys
import joblib

sys.path.append("./")

# Path to the unprocessed data (original_data.zip)
original_data_dir = "data\\Pokemon\\original_data"
# Path to the data used for training the VQVAE
standard_data_dir = "data\\Pokemon\\final\\standard"
# Path to output dir; will be created if it doesn't exist
output_data_dir = "data\\Pokemon\\inpainting"

# Create output data dir if it doesn't exist
if not os.path.exists(output_data_dir):
    os.makedirs(output_data_dir)

for folder in ["train", "test", "val"]:
    for subfolder in ["inputs", "labels"]:
        dir = os.path.join(output_data_dir, folder, subfolder)
        if not os.path.exists(dir):
            os.makedirs(dir)

train_dir = os.path.join(standard_data_dir, "train")
val_dir = os.path.join(standard_data_dir, "val")
test_dir = os.path.join(standard_data_dir, "test")


def get_data_ids(data_dir):
    all_ids = []
    for file in os.listdir(data_dir):
        id = file.split(".")[0].split("_")[0].split("-")[0]
        if id not in all_ids:
            all_ids.append(id)
    return all_ids


train_ids = get_data_ids(train_dir)
val_ids = get_data_ids(val_dir)
test_ids = get_data_ids(test_dir)

train_index = []
val_index = []
test_index = []
for input_file in os.listdir(original_data_dir):
    if "shiny" in input_file:
        continue
    print(input_file)
    # Load file
    file_path = os.path.join(original_data_dir, input_file)
    target_img = Image.open(file_path).convert("RGBA")

    # Find directory to save to
    input_file_name = input_file.split(".")[0]
    image_id = input_file_name.split("_")[0].split("-")[0]

    # Create silhouette & target image
    background = Image.new("RGBA", target_img.size, (0, 0, 0))
    np_sprite = np.asarray(target_img)
    mask = np_sprite[:, :, 3] == 255
    np_sprite[mask] = 255
    source_img = Image.fromarray(np_sprite)

    # Create final images
    source_img = Image.alpha_composite(background, source_img).convert("RGB")
    target_img = Image.alpha_composite(background, target_img).convert("RGB")
    # Flip images horizontally
    source_img_flipped = source_img.transpose(Image.FLIP_LEFT_RIGHT)
    target_img_flipped = target_img.transpose(Image.FLIP_LEFT_RIGHT)
    # Define filenames to save as
    source_filename = f"{input_file_name}.png"
    target_filename = f"{input_file_name}.png"
    source_flipped_filename = f"{input_file_name}_flipped.png"
    target_flipped_filename = f"{input_file_name}_flipped.png"
    # Figure out save dir
    if image_id in test_ids:
        save_dir = os.path.join(output_data_dir, "test")
        test_index.extend([source_filename, source_flipped_filename])
    elif image_id in val_ids:
        save_dir = os.path.join(output_data_dir, "val")
        val_index.extend([source_filename, source_flipped_filename])
    else:
        save_dir = os.path.join(output_data_dir, "train")
        train_index.extend([source_filename, source_flipped_filename])
    # Save
    source_img.save(os.path.join(save_dir, "inputs", source_filename))
    target_img.save(os.path.join(save_dir, "labels", target_filename))
    source_img_flipped.save(os.path.join(save_dir, "inputs", source_flipped_filename))
    target_img_flipped.save(os.path.join(save_dir, "labels", target_flipped_filename))
# Save index files
joblib.dump(train_index, os.path.join(output_data_dir, "train", "index.joblib"))
joblib.dump(val_index, os.path.join(output_data_dir, "val", "index.joblib"))
joblib.dump(test_index, os.path.join(output_data_dir, "test", "index.joblib"))

print(
    f"Train: {len(train_index)} ({len(train_ids)})\nVal: {len(val_ids)} ({len(val_index)})\nTest: {len(test_ids)} ({len(test_index)})"
)
