"""
Disclaimer: This script contains older code and may not work as is.
This script is responsible for generating classification labels for any dataset
specified in the config.
"""

import os
import sys
import pandas as pd
import joblib

sys.path.append("./")

################################################################################
#################################### Config ####################################
################################################################################

# Data Config
dataset = "sprites" # "pokemon", "tinyhero", "sprites"
data_folder_lookup = {
    "pokemon": "data\\pokemon\\final\\standard",
    "tinyhero": "data\\TinyHero\\final",
    "sprites": "data\\Sprites\\final"
}
output_folder_lookup = {
    "pokemon": "data\\pokemon\\classification",
    "tinyhero": "data\\TinyHero\\classification",
    "sprites": "data\\Sprites\\classification"
}
data_prefix = data_folder_lookup[dataset]
output_dir = output_folder_lookup[dataset]
pokedex_url = "data\\Pokemon\\pokedex_(Update_04.21).csv"

################################################################################
#################################### Config ####################################
################################################################################

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Preprocess Pokedex
if "pokemon" in dataset:
    pokedex = pd.read_csv(pokedex_url)
    for forbidden in ["Mega ", "Partner ", "Alolan ", "Galarian "]:
        pokedex.drop(pokedex[pokedex['name'].str.contains(forbidden)].index, inplace=True)
    pokedex.drop_duplicates(["pokedex_number", "name"], inplace=True)
    pokedex.index = pokedex.pokedex_number
    names = pokedex["name"].to_dict()
    height = pokedex["height_m"].to_dict()
    weight = pokedex["weight_kg"].to_dict()
    type1 = pokedex["type_1"].to_dict()
    type2 = pokedex["type_2"].to_dict()
    egg1 = pokedex["egg_type_1"].to_dict()
    egg2 = pokedex["egg_type_2"].to_dict()

# Create labels
for folder in ["train", "val", "test"]:
    all_files = os.listdir(os.path.join(data_prefix, folder))
    final = {}
    for filename in all_files:
        print(filename)
        if "pokemon" in dataset:
            type_bypass = False
            pokemon_id = filename.split(".")[0].split("_")[0]
            if "-" in pokemon_id:
                pokemon_id, *form = pokemon_id.split("-")
                form = "-".join(form)
                type_bypass = True
                # No changes needed
                if form in ["beta", "spiky-eared"] or pokemon_id in ["201", "386", "412", "421", "487", "422", "423", "550", "585", "586", "641", "642", "645", "646", "647", "649"]:
                    type_bypass = False
                # Castform
                elif pokemon_id == "351":
                    bypass_type2 = float("nan")
                    if form == "rainy":
                        bypass_type1 = "Water"
                    elif form == "sunny":
                        bypass_type1 = "Fire"
                    elif form == "snowy":
                        bypass_type1 = "Ice"
                    else:
                        bypass_type1 = "Normal"
                # Wormadan
                elif pokemon_id == "413":
                    bypass_type1 = "Bug"
                    if form == "sandy":
                        bypass_type2 = "Ground"
                    elif form == "trash":
                        bypass_type2 = "Steel"
                    else:
                        bypass_type2 = "Grass"
                # Rotom
                elif pokemon_id == "479":
                    bypass_type1 = "Electric"
                    if form == "fan":
                        bypass_type2 = "Flying"
                    elif form == "frost":
                        bypass_type2 = "Ice"
                    elif form == "heat":
                        bypass_type2 = "Fire"
                    elif form == "mow":
                        bypass_type2 = "Grass"
                    elif form == "wash":
                        bypass_type2 = "Water"
                    else:
                        bypass_type2 = "Ghost"
                # Shaymin
                elif pokemon_id == "492":
                    bypass_type1 = "Grass"
                    if form == "sky":
                        bypass_type2 = "Flying"
                    else:
                        bypass_type2 = float("nan")
                # Arceus
                elif pokemon_id == "493":
                    bypass_type2 = float("nan")
                    if form == "unknown":
                        bypass_type1 = "Normal"
                    else:
                        bypass_type1 = form.capitalize()
                # Darmanitan
                elif pokemon_id == "555":
                    bypass_type1 = "Fire"
                    if form == "zen":
                        bypass_type2 = float("nan")
                    else:
                        bypass_type2 = "Psychic"
                # Meloetta
                elif pokemon_id == "646":
                    bypass_type1 = "Normal"
                    if form == "pirouette":
                        bypass_type2 = "Psychic"
                    else:
                        bypass_type2 = "Fighting"
            pokemon_id_int = int(pokemon_id)
            if type_bypass:
                final[filename] = {
                    "height": height[pokemon_id_int],
                    "weight": weight[pokemon_id_int],
                    "type1": bypass_type1,
                    "type2": bypass_type2,
                    "egg1": egg1[pokemon_id_int],
                    "egg2": egg2[pokemon_id_int],
                }
            else:
                final[filename] = {
                    "height": height[pokemon_id_int],
                    "weight": weight[pokemon_id_int],
                    "type1": type1[pokemon_id_int],
                    "type2": type2[pokemon_id_int],
                    "egg1": egg1[pokemon_id_int],
                    "egg2": egg2[pokemon_id_int],
                }
        elif dataset == "tinyhero":
            id, pose, background, rotation = filename.split('.')[0].split('_')
            final[filename] = {
                "id": id,
                "pose": pose,
                "background": background,
                "rotation": rotation
            }
        elif dataset == "sprites":
            id, pose, anim, background, rotation = filename.split('.')[0].split('_')
            final[filename] = {
                "id": id,
                "pose": pose,
                "anim": anim,
                "background": background,
                "rotation": rotation
            }
    joblib.dump(final, os.path.join(output_dir, f"{folder}.joblib"))