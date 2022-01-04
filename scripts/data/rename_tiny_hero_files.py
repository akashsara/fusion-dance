import os

source_dir = "data\\TinyHero\\final" # Path to the tiny hero directory
target_dir = os.path.join(source_dir, "processed")

if not os.path.exists(target_dir):
    os.mkdir(target_dir)

for folder, suffix in zip(["0", "1", "2", "3"], ["back", "left", "front", "right"]):
    folder = os.path.join(source_dir, folder)
    for file in os.listdir(folder):
        src_path = os.path.join(folder, file)
        new_name = file.split(".")[0] + '_' + suffix + '.png'
        target_path = os.path.join(target_dir, new_name)
        print(src_path, target_path)
        os.rename(src_path, target_path)
    os.rmdir(folder)