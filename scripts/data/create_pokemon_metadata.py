"""
This files compiles data from several datasets and combines them.
In addition it handles several exceptions and edge cases.
The final data contains the following attributes for all Pokemon:
    - height
    - weight
    - egg group 1
    - egg group 2
    - color
    - shape
    - type 1
    - type 2

This information is used in any script that uses a conditional PixelCNN.
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
data_prefix = "data\\pokemon\\final\\standard"
output_dir = "data\\Pokemon"
pokedex_url = "data\\Pokemon\\pokedex_(Update_04.21).csv"
# Path to file containing different pokemon shapes
pokemon_shapes = "data\\Pokemon\\pokemon_shapes.csv" 
# Path to file containing different pokemon colors
pokemon_colors = "data\\Pokemon\\pokemon_colors.csv" 
# Path to file to link shapes/colors to pokemon IDs
pokemon_connector = "data\\Pokemon\\pokemon_species.csv" 

################################################################################
#################################### Config ####################################
################################################################################

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

connector = pd.read_csv(pokemon_connector)
# Drop newer Pokemon that aren't in our data 
connector.drop(connector[connector.id > 649].index, inplace=True)
shapes = pd.read_csv(pokemon_shapes)
colors = pd.read_csv(pokemon_colors)
connector['shape'] = connector.shape_id.map(shapes.set_index('id').identifier)
connector['color'] = connector.color_id.map(colors.set_index('id').identifier)
connector['id'] = connector['id'].apply(lambda x: '0' * (3 - len(str(x))) + str(x))
connector = connector[['id', 'color', 'shape']].to_dict('records')
connector = {item['id']:{key: value for key, value in item.items() if key != 'id'} for item in connector}

# Preprocess Pokedex
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
final = {}
for folder in ["train", "val", "test"]:
    all_files = os.listdir(os.path.join(data_prefix, folder))
    for filename in all_files:
        print(filename)
        type_bypass = False
        shape_bypass = False
        pokemon_id = filename.split(".")[0].split("_")[0]
        # If Pokemon has multiple forms
        if "-" in pokemon_id:
            pokemon_id, *form = pokemon_id.split("-")
            form = "-".join(form)
            type_bypass = True
            # No changes needed
            if form in ["beta", "spiky-eared"] or pokemon_id in ["201", "386", "412", "421", "422", "423", "550", "585", "586", "646", "647", "649"]:
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
            # Giratina
            elif pokemon_id == "487":
                type_bypass = False
                bypass_shape = "squiggle"
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
            # Tornadus
            elif pokemon_id == "641":
                type_bypass = False
                bypass_shape = "wings"
            # Thundurus
            elif pokemon_id == "642":
                type_bypass = False
                bypass_shape = "upright"
            # Landorus
            elif pokemon_id == "645":
                type_bypass = False
                bypass_shape = "quadruped"
            # Meloetta
            elif pokemon_id == "646":
                bypass_type1 = "Normal"
                if form == "pirouette":
                    bypass_type2 = "Psychic"
                else:
                    bypass_type2 = "Fighting"
        pokemon_id_int = int(pokemon_id)
        final[filename] = {
            "height": height[pokemon_id_int],
            "weight": weight[pokemon_id_int],
            "egg1": egg1[pokemon_id_int],
            "egg2": egg2[pokemon_id_int],
            "color": connector[pokemon_id]['color']
        }
        if type_bypass:
            final[filename]["type1"] = bypass_type1
            final[filename]["type2"] = bypass_type2
        else:
            final[filename]["type1"] = type1[pokemon_id_int]
            final[filename]["type2"] = type2[pokemon_id_int]
        if shape_bypass:
            final[filename]["shape"] = bypass_shape
        else:
            final[filename]["shape"] = connector[pokemon_id]['shape']
joblib.dump(final, os.path.join(output_dir, f"metadata.joblib"))