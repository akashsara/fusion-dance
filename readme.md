# Fusion Dance

This is the repository for "Fusion Dance: Generating New Content via Fusion of Existing Content".

The original data can be found in the original_data folder. The final preprocessed data used for modeling can be obtained by running the `preprocess.py` script.

TODO:
* Basic first draft model
* Compute metrics - MSE & SSIM
* Combine representations & see initial results
* Revisit data being used
    Look at other augmentations - maybe a noisy background instead of white or black?
    Or use other constant colors such as magenta and cyan to make the model learn to ignore the background
* Revisit the model
    Is a ConvVAE doing well enough?
    Are the hyperparameters good?