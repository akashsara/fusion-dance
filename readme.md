# Pixel VQ-VAEs for Improved Pixel Art Representation

This is the repository for "Pixel VQ-VAEs for Improved Pixel Art Representation". In short, we introduce the Pixel VQ-VAE, a modified Vector Quantized VAE that has possesses special improvements to better work with pixel art. This work will be presented at EXAG 2022 and our paper can be found [here](https://arxiv.org/abs/2203.12130). 

## Citation

If you found this paper or this project useful for your research, please cite us:

```
@inproceedings{saravanan2022pixelvqvae,
  title={Pixel VQ-VAEs for Improved Pixel Art Representation},
  author={Saravanan, Akash and Guzdial, Matthew},
  booktitle= {{Experimental AI in Games (EXAG), AIIDE 2022 Workshop}},
  year={2022}
}
```

## Abstract

Machine learning has had a great deal of success in image processing. However, the focus of this work has largely been on realistic images, ignoring more niche art styles such as pixel art. Additionally, many traditional machine learning models that focus on groups of pixels do not work well with pixel art, where individual pixels are important. We propose the Pixel VQ-VAE, a specialized VQ-VAE model that learns representations of pixel art. We show that it outperforms other models in both the quality of embeddings as well as performance on downstream tasks.

## Usage

Install the required packages via `pip install -r requirements.txt`.

The original data can be found in the original_data folder. This consists of the data obtained by merging different Pokemon game data and running a deduplication program. This data was obtained from [veekun](https://veekun.com/dex/downloads). For specific instructions on recreating this work please refer to [this](data/readme.md).

`vq_vae.py`

This runs the training script for the Pixel VQ-VAE. Simply fill up the config and run!

`scripts/compute_metrics_from_images.py`

This computes the SSIM and MSE metrics.

`gated_pixelcnn_prior.py`

This script trains the PixelCNN for generating images using the Pixel VQ-VAE encodings.

`python -m pytorch_fid path/to/dataset1 path/to/dataset2`

This computes the FID scores for generated outputs.

All models automatically generate and save outputs for the test set.

## Fusion Dance?

If you're wondering about the name, that is because this project was originally started to attempt to generate new Pokemon via the fusion of existing Pokemon, much like the hand-crafted fusions of old. This is also why you might find some seemingly unrelated scripts and code in this repository. While I cannot guarantee that they all will still work simply due to the significant changes that have happened since then, feel free to get in touch or open up an issue if you're interested in fusing Pokemon (or other content) using AI!