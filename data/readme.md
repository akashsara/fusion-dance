# Pokemon Sprites

1. Extract the files from `Pokemon/original_data.zip`
2. Run the preprocessing script `../preprocess.py` on the dataset.
3. Download the fusion dataset from https://github.com/Aegide/FusionSprites
4. Run `../scripts/data/fix_infinite_fusion_filenames.py` on the `Japeal` folder within that repository.
5. Run `../scripts/data/split_fusion_data.py` on the folder from above and the extracted files from step 1.
6. Run `../scripts/data/split_training_fusion_data.py` on the two training directories resulting from the above.

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

# Pokemon Data CSV (for Embedding Analysis)
1. https://www.kaggle.com/mariotormo/complete-pokemon-dataset-updated-090420