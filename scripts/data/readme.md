1. Extract the files from `../../data/Pokemon/original_data.zip`
2. Download the fusion dataset from https://github.com/Aegide/FusionSprites
3. Run `fix_infinite_fusion_filenames.py` on the `Japeal` folder within that repository.
4. Run `split_fusion_data.py` on the folder from above and the extracted files from step 1.
5. Run `split_training_fusion_data.py` on the two training directories resulting from the above.