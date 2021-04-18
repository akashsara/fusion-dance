# Fusion Dance

This is the repository for "Fusion Dance: Generating New Content via Fusion of Existing Content". The goal of this project is to generate new Pokemon by blending latent representations of different Pokemon. We work with both a convolutional autoencoder and a convolutional variational autoencoder.

Although the initial results aren't the greatest, we identify a number of possible avenues for future work. Broadly speaking, this covers methods of improving the VAE in order to reduce fuzziness, biasing of the latent space to learn more structure and the use of a larger corpus of pre-existing handcrafted fusions

The original data can be found in the original_data folder. The final preprocessed data used for modeling can be obtained by running the `preprocess.py` script.