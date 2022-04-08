# Splitdim Adversarial Disentanglement of Speaker Emeddings

Repository for the paper: 

> **Investigating the contribution of speaker attributes to speaker separability using disentangled speaker representations** 

by Luu, Renals and Bell, to be presented at Interspeech 2022, Incheon, South Korea. 

# Concept

This work disentangles speaker attributes in a supervised fashion by targeting specific dimensions of speaker embeddings. 

Unlike some other work, all embeddings are trained from scratch (and this was found to be crucial in obtaining truly disentangled dimensions).

In the figure below in the `(c)` configuration, speaker attributes are isolated to specified dimensions by training classifier-discriminator pairs on complementary dimensions of the speaker embedding. For example, speaker gender is encoded in the first dimension via the classifier, and the adversary discourages the rest of the embedding dimensions for containing this information. This can be repeated for a number of different attributes.

<p align="center">
  <img src="figures/splitdim3diag.png?raw=true">
</p>

# Data Preparation and Setup

For more details, see [MTL-Speaker-Embeddings](https://github.com/cvqluu/MTL-Speaker-Embeddings/), since the data preparation is identical.

Training models is also performed in an identical fashion, with the format:

```sh
python train.py --cfg <path-to-cfg>
```

# Training Configuration

Here this repository differs from [MTL-Speaker-Embeddings](https://github.com/cvqluu/MTL-Speaker-Embeddings/), since now multi-task heads can act on splits of the embedding dimensions.

An example configuration file can be found in `configs/example_config.cfg`, which we will also describe below. Since all the configuration options are detailed in the other repository, we will only describe the options that pertain to this work.

## Model specification
```cfg
[Model]
model_type = XTDNN
classifier_heads = speaker,gender,gender,nationality,nationality
```

- `model_type` defines the architecture. `XTDNN` is the only model type that was used in this work (equivalent to the x-vector architecture)
- `classifier_heads` defines all the classifier heads that act upon the speaker embedding. In this example, we have a speaker classification head, along with classifier-adversary pairs for gender and nationality. The ordering and length of this option will propagate forward, and all other head related options must be specified for each head defined here.


## Classifier head parameters
```cfg
[Optim]
classifier_types = adm,xvec,xvec,xvec,xvec
classifier_lr_mults = [1.0,1.0,1.0,1.0,1.0]
classifier_loss_weights = [1.0,0.05,-20.0,0.05,-10.0]
classifier_smooth_types = none,none,none,none,none
```

- `classifier_types` indicates the architecture of each classifier specified in `classifier_heads`
- `classifier_lr_mults` is the learning rate for each head.
- `classifier_loss_weights` is the loss weighting for each head in the total loss sum.
- `classifier_smooth_types` is an option for label smoothing that was always left at `none` for this work.


## Embedding extractor parameters
```cfg
[Hyperparams]
embedding_dim = 64
```
- `embedding_dim` specifies the number of dimensions in the speaker embedding.
- For more details on this section, see the other repo.


## Dimension split specification
```cfg
[Misc]
split_embedding_dims_per_task = True
split_embedding_dims = [[[0,64]],[[0,1]],[[1,64]],[[1,11]],[[0,1], [11,64]]]
```

Here we define the split of dimensions.
- `split_embedding_dims_per_task` is a boolean that must be set to `True` to enable this feature
- `split_embedding_dims` is a nested list defining which embedding dimensions should be used as input to each classifier head.
  - The length of this list should be the same as the number of heads.
  - Each element of this list is itself a list (with variable length >=1).
    - The elements of these lists are pairs of integers representing the start and end dimensions to slice, which are then to be concatenated for input into the classifier head.

In this example:
- The `speaker` classifier uses all dimensions `[0,64]`
- The `gender` (classifier) uses the first dimension `[0,1]`
- The `gender` (adversary) uses the remaining dimensions `[1,64]`
- The `nationality` (classifier) uses dimensions `[1,11]`
- And crucially, the `nationality` (adversary) uses the following dimensions:
  - The first dimension `[0,1]`
  - The dimensions `[11,64]`
  - These are concatenated together as input


