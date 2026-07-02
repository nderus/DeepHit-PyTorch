# DeepHit (Unofficial PyTorch Implementation)

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Lint](https://github.com/nderus/DeepHit-PyTorch/actions/workflows/lint.yml/badge.svg)](https://github.com/nderus/DeepHit-PyTorch/actions/workflows/lint.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An unofficial PyTorch port of **DeepHit**, a deep neural network for survival
analysis that handles **competing risks** without restrictive parametric
assumptions about the underlying stochastic process. The model directly learns
the joint distribution over event types and discrete event times, from which
cause-specific cumulative incidence functions can be read off.

Based on the original TensorFlow implementation by Lee et al.
([chl8856/DeepHit](https://github.com/chl8856/DeepHit)); see [Citation](#citation)
below. The architecture, losses, and evaluation protocol mirror the original,
except for one documented correction (see [Reproduction](#reproduction)).

## Relation to other implementations

This repository is a faithful PyTorch translation of the original TensorFlow code
([chl8856/DeepHit](https://github.com/chl8856/DeepHit)), preserving its
architecture, loss, masks, and integer-time discretization with a padded horizon.
Other packages re-implement DeepHit with different design choices, so their
results are not directly comparable.

## Installation

Clone the repo and create the conda environment from the provided spec:

```bash
git clone https://github.com/nderus/DeepHit-PyTorch.git
cd DeepHit-PyTorch
conda env create -f environment.yml
conda activate deephit-torch
```

The environment pins Python 3.8, PyTorch 2.0.1, lifelines, scikit-learn, and
pandas. Without conda, `pip install -r requirements.txt` covers the core
runtime dependencies.

## Quickstart

Two sample datasets are bundled under `sample data/`:

- `SYNTHETIC`: competing-risks synthetic benchmark from the original paper (two event types).
- `METABRIC`: breast-cancer cohort with a single event type.

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

## Reproduction

Protocol: the original random search (5 outer splits, 50 configs each), best
config per split by validation C-index, evaluated on the held-out test set.
Metric is the cause-specific time-dependent concordance index ($C^{td}$),
reported per horizon plus a risk-set-weighted aggregate, averaged over the 5
splits.

One correction vs. upstream: the original pipeline standardizes features over
the full dataset before splitting, which leaks test statistics into training.
Here the standardization is fit on the training split and applied to validation
and test. Impact on these datasets is minor.

SYNTHETIC (two competing events), horizons {12, 24, 36}:

| Event | @12 | @24 | @36 | Aggregate |
|:-----:|:-----:|:-----:|:-----:|:---------:|
|   1   | 0.765 | 0.733 | 0.720 | **0.728** |
|   2   | 0.755 | 0.731 | 0.712 | **0.724** |

METABRIC (single event), horizons {144, 288, 432} in months:

|  @144 | @288  | @432  | Aggregate |
|:-----:|:-----:|:-----:|:---------:|
| 0.684 | 0.637 | 0.638 | **0.638** |

All 5 splits reuse the fixed `seed=1234` split (as upstream), so the spread is
small (std under 0.014). These are point estimates on one split, not
generalization variance. Values sit in the DeepHit literature range ($C^{td}$
roughly 0.66 to 0.73). The bundled `sample data/` may differ from the paper
cohorts, so treat this as a fidelity check, not an exact replication.

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

Released under the [MIT License](LICENSE).
