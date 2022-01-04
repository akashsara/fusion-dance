# Pokemon Sprites

1. Extract the files from `../../data/Pokemon/original_data.zip`
2. Run the preprocessing script `../../preprocess.py` on the dataset.
3. Download the fusion dataset from https://github.com/Aegide/FusionSprites
4. Run `fix_infinite_fusion_filenames.py` on the `Japeal` folder within that repository.
5. Run `split_fusion_data.py` on the folder from above and the extracted files from step 1.
6. Run `split_training_fusion_data.py` on the two training directories resulting from the above.

# Pokemon Sugimori

1. Download and extract the dataset from the link in `../../data/Pokemon-Sugimori/source.txt`. 
2. Remove any empty images in the extracted directory.
2. Run the preprocessing script `../../preprocess.py` on the dataset.# Pokemon Sugimori

# Tiny Hero
1. Extract the files from `../../data/TinyHero/original.zip`
2. Run `rename_tiny_hero_files.py` on the dataset.
2. Run the preprocessing script `../../preprocess.py` on the dataset.