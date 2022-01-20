import os

source_dir = "data\\Sprites\\frames"  # Path to the tiny hero directory
target_dir = os.path.join(source_dir, "processed")

if not os.path.exists(target_dir):
    os.mkdir(target_dir)

for folder in ["slash", "spellcard", "walk"]:
    prefix = folder
    folder = os.path.join(source_dir, folder)
    for file in os.listdir(folder):
        src_path = os.path.join(folder, file)
        pose, id, animation_frame = file.split(".")[0].split("_")
        new_name = f"{id}_{pose}_{prefix}{animation_frame}.png"
        target_path = os.path.join(target_dir, new_name)
        print(src_path, target_path)
        os.rename(src_path, target_path)
    os.rmdir(folder)
