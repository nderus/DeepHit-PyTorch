# DeepHit (Unofficial PyTorch Implementation)

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An unofficial PyTorch port of **DeepHit**, a deep neural network for survival
analysis that handles **competing risks** without restrictive parametric
assumptions about the underlying stochastic process. The model directly learns
the joint distribution over event types and discrete event times, from which
cause-specific cumulative incidence functions can be read off.

Based on the original TensorFlow implementation by Lee et al.
([chl8856/DeepHit](https://github.com/chl8856/DeepHit)) — see [Citation](#citation)
below. The architecture, losses, and evaluation protocol mirror the original;
any deviations from the upstream implementation are unintentional.

## Installation

Clone the repo and create the conda environment from the provided spec:

```bash
git clone https://github.com/nderus/DeepHit-PyTorch.git
cd DeepHit-PyTorch
conda env create -f environment.yml
conda activate deephit-torch
```

The environment pins Python 3.8, PyTorch 2.0.1, lifelines, scikit-learn, and
pandas.

## Quickstart

Two sample datasets are bundled under `sample data/`:

- `SYNTHETIC` — the competing-risks synthetic benchmark from the original paper.
- `METABRIC` — breast-cancer cohort with two event types.

Pick a dataset by editing the `data_mode` variable near the top of
`main_RandomSearch.py` (and `summarize_results.py`), then:

```bash
# Run random hyperparameter search: 5 outer splits × 50 random configs each.
python main_RandomSearch.py

# Aggregate cause-specific weighted C-index and Brier scores across splits.
python summarize_results.py
```

Best checkpoints land in `<DATASET>/results/itr_<i>/models/` and the winning
hyperparameters for each outer split in
`<DATASET>/results/itr_<i>/hyperparameters_log.txt`.

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{lee2018deephit,
  title     = {DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks},
  author    = {Lee, Changhee and Zame, William R. and Yoon, Jinsung and van der Schaar, Mihaela},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2018}
}
```

## License

This repository does not currently include a license file. Refer to the
[original DeepHit repository](https://github.com/chl8856/DeepHit) for the
canonical implementation's licensing terms.
