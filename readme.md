# Fusion Dance

This is the repository for "Fusion Dance: Generating New Content via Fusion of Existing Content". The goal of this project is to generate new Pokemon by blending latent representations of different Pokemon. We work with both a convolutional autoencoder and a convolutional variational autoencoder.

## Abstract

Creating new content for video games is a time and resource-intensive task. While machine learning models have performed admirably in other fields, a lack of data and controllability for these models makes it difficult to adapt them for video game content creation. We propose an autoencoder-based approach to learning representations of game assets, and then explore how the combination or blending of these representations can lead to new assets. We experimentally verify this approach using Pokemon from the Pokemon video game series. Our experimental results indicate that the blending of representations is a worthwhile pursuit towards the generation of novel content.

## Usage

Install the required packages via `pip install -r requirements.txt`.

The original data can be found in the original_data folder. The final preprocessed data used for modeling can be obtained by running the `preprocess.py` script.

To train the models, fill up the config section of the respective notebook and run. MSE and SSIM scores are computed in the notebooks. We use the [pytorch_fid](https://github.com/mseitzer/pytorch-fid) package to compute the FID scores.

To play with/generate fusions, use the notebook from the fusions folder.