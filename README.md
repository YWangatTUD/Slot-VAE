# **Slot-VAE: Object-Centric Compositional Image Generation with Slot Attention**

## **We propose Slot-VAE, an generative approach that discovers compositional concepts from images with slot attention and reuses them for compositional image generation.**
![](/slot-vae_overview.png)

[Slot-VAE: Object-Centric Scene Generation with Slot Attention](https://arxiv.org/abs/2306.06997)

[Yanbo Wang](https://sps.ewi.tudelft.nl/People/bio.php?id=810) <sup>1</sup>, [Letao Liu](https://sps.ewi.tudelft.nl/People/bio.php?id=744) <sup>2</sup>, [Justin Dauwels](https://sps.ewi.tudelft.nl/People/bio.php?id=744)  <sup>1</sup>

<sup>1</sup> TU Delft, <sup>2</sup> NTU


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
