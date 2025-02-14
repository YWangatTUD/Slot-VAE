# **Slot-VAE: Object-Centric Compositional Image Generation with Slot Attention**

## **We propose Slot-VAE, an unsupervised approach that discovers compositional concepts from images with slot attention and reuses them for compositional image generation.**
![](/slot-vae_overview.png)

[Paper](https://arxiv.org/abs/2306.06997) | [Project Page]


## **Qualitative Results**
![](/slot-vae_teaser.png)





## **Setup**

Run the following to create and activate a conda environment:

```
conda env create -f environment.yml
conda activate slot-vae_env
```

## **Training**

To train the model, run 

```
python train.py --config-file config/arrow.yaml
```
