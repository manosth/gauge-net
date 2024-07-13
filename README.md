# Constructing gauge-invariant neural networks for scientific applications
Official code repository for "[Constructing gauge-invariant neural networks for scientific applications](https://openreview.net/forum?id=1fwEsyICb3)", an extended abstract accepted at the [AI4Science](https://ai4sciencecommunity.github.io) and [GRaM](https://gram-workshop.github.io) workshops at ICML 2024.

It consists of an gauge-invariant architecture for predicting the energy configuration in the 2D XY model. The repository contains a folder `data_gen` wuth various files to generate and test datasets:
- - `gen_gauge` is the main file: it generates a 2D XY model. It also, in the later part, adds a gauge field to the XY model, where the angles at each grid point are smoothly perturbed,
  - `gen_dataset` uses the logic of `gen_gauge` to generate a whole dataset of 2D XY samples (without the smooth gauge),
  - `gen_dataset_gauge` updates an _already created_ dataset to have the smooth gauge,
  - `sample_dataset` randomly samples a created dataset, and
  - `test_dataset` is similar to `sample_dataset`, but iterates until a set number of high-, mid-, and low-energy states have been found.
  
The rest of the files are training and testing files for different architectures and they come in pairs. `egnn_clean` is from the [official EGNN repo](https://github.com/vgsatorras/egnn) and is necessary for `*_egnn`.
The models that are available are:
- [EGNN](https://github.com/vgsatorras/egnn) in `train_egnn` and `test_egnn`,
- [EMLP](https://github.com/mfinzi/equivariant-MLP/tree/master) in `train_emlp` and `test_emlp`,
- [(Pretrained) ResNets](https://arxiv.org/abs/1512.03385) in `train_pretrained` and `test_pretrained`, and
- Our proposed architecture to estimate the energy in the 2D XY model in `train_egnn_ours` and `test_egnn_ours`.

