# Fusion Dance

This is the repository for "Fusion Dance: Generating New Content via Fusion of Existing Content". The goal of this project is to generate new Pokemon by blending latent representations of different Pokemon. We experiment with a wide variety of models for this task.

## Abstract

Creating new content for video games is a time and resource-intensive task. While machine learning models have performed admirably in other fields, a lack of data and controllability for these models makes it difficult to adapt them for video game content creation. We propose an autoencoder-based approach to learning representations of game assets, and then explore how the combination or blending of these representations can lead to new assets. We experimentally verify this approach using Pokemon from the Pokemon video game series. Our experimental results indicate that the blending of representations is a worthwhile pursuit towards the generation of novel content.

## Usage

Install the required packages via `pip install -r requirements.txt`.

The original data can be found in the original_data folder. The final preprocessed data used for modeling can be obtained by running the `preprocess.py` script. Fusion data can be obtained from the [FusionSprites repository](https://github.com/Aegide/FusionSprites). Note that you will need to run the `fix_infinite_fusion_filenames.py` to fix the filenames.

To train the models, fill up the config section of the respective model script and run. 

<!-- MSE and SSIM scores are computed in the notebooks. We use the [pytorch_fid](https://github.com/mseitzer/pytorch-fid) package to compute the FID scores. -->

All models automatically generate and save outputs for the test set (including fusions if the test set uses them). 