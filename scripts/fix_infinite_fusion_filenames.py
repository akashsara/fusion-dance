# Simply renames the misnamed files
# By using the mapping csv 
import pandas as pd
import os
import shutil

target_dir = "data\\Infinite Fusion"
df = pd.read_csv("utils\\mapping.csv")
mapping = {str(x["Num"]): str(x["TrueNum"]) for x in df.to_dict(orient="records")}

errors = []
folder_mapping = {}
folders = os.listdir(target_dir)
for i, folder in enumerate(folders):
    files = os.listdir(os.path.join(target_dir, folder))
    base_dir = os.path.join(target_dir, folder)
    temp_dir = os.path.join(target_dir, folder, "temp")
    os.mkdir(temp_dir)
    for file in files:
        file_path = os.path.join(base_dir, file)
        if file == f"{folder}.png":
            os.remove(file_path)
            print(f"Remove: {file_path}")
        else:
            first, second, _ = file.split(".")
            first = first.split("-")[0]
            second = second.split("-")[0]
            skip = True
            if int(first) > 420 or int(second) > 420:
                os.remove(file_path)
                print(f"Remove: {file_path}")
            elif first == folder:
                if int(first) > 251:
                    first = mapping[first]
                    skip = False
                if int(second) > 251:
                    second = mapping[second]
                    skip = False
                new_file = f"{first}.{second}.png"
                if skip:
                    continue
                new_path = os.path.join(temp_dir, new_file)
                j = 1
                while os.path.exists(new_path):
                    new_path = os.path.join(temp_dir, new_file[:-4] + f"-{j}.png")
                    j += 1
                os.rename(file_path, new_path)
                print(f"Rename: {file} to {new_path}")
            else:
                errors.append(file)
                print(f"Error: {file}")
    for file in os.listdir(temp_dir):
        src = os.path.join(temp_dir, file)
        dest = os.path.join(base_dir, file)
        os.rename(src, dest)
    shutil.rmtree(temp_dir)
    if int(folder) > 251:
        temp_folder = f"TEMP-{i}"
        folder_mapping[temp_folder] = mapping[folder]
        os.rename(base_dir, os.path.join(target_dir, temp_folder))
        print(folder, temp_folder)

# Rename Folders
for folder, new_name in folder_mapping.items():
    folder = os.path.join(target_dir, folder)
    new_name = os.path.join(target_dir, new_name)
    os.rename(folder, new_name)
    print(folder, new_name)