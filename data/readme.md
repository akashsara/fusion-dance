# Pokemon Sprites

1. Extract the files from `Pokemon/original_data.zip`
2. Run the preprocessing script `../preprocess.py` on the dataset.
3. If you need images of Pokemon fusions, follow steps 4-7. Otherwise stop here.
4. Download the fusion dataset from https://github.com/Aegide/FusionSprites
5. Run `../scripts/data/fix_infinite_fusion_filenames.py` on the `Japeal` folder within that repository.
6. Run `../scripts/data/split_fusion_data.py` on the folder from above and the extracted files from step 1.
7. Run `../scripts/data/split_training_fusion_data.py` on the two training directories resulting from the above.

# Pokemon Sugimori

1. Download and extract the dataset from https://veekun.com/static/pokedex/downloads/pokemon-sugimori.tar.gz
2. Remove any empty images in the extracted directory.
3. Run the preprocessing script `../preprocess.py` on the dataset.

# The Sprite Dataset

1. Clone the repository from https://github.com/YingzhenLi/Sprites. 
2. Run the `random_character.py` in this codebase.
3. Pass the `frames` directory generated from the above to `../scripts/data/process_sprite_dataset_files.py`
4. Run the preprocessing script `../preprocess.py` on the dataset.

# Tiny Hero
1. Extract and download the files from https://github.com/AgaMiko/pixel_character_generator.
2. Run `rename_tiny_hero_files.py` on the dataset.
3. Run the preprocessing script `../preprocess.py` on the dataset.

# Pokemon Data CSVs

### For Embedding Analysis & Conditioning Information:

https://www.kaggle.com/mariotormo/complete-pokemon-dataset-updated-090420

File: `pokedex_(Update_04.21).csv`

This dataset is used in the following scripts: 
```
fusion-dance\scripts\compute_vae_embeddings.py
fusion-dance\scripts\compute_vqvae_embeddings.py
fusion-dance\scripts\data\create_classification_labels.py
fusion-dance\scripts\data\create_pokemon_metadata.py
```

### For Conditioning Information:

https://github.com/PokeAPI/pokeapi

Files: `pokemon_colors.csv`, `pokemon_shapes.csv`, `pokemon_species.csv`

These datasets are used in the following script:

```
fusion-dance\scripts\data\create_pokemon_metadata.py
```