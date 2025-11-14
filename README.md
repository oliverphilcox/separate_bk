# separate-bk

This code computes separable three-dimensional representations for arbitrary input bispectra, as discussed in [Philcox, Zhong \& Sirletti](https://arxiv.org/2511.XXXXX).  Given an input shape function, the code computes a set of neural network basis functions and weights that accurately reproduce the input spectra in a factorizable manner. This can be interfaced with CMB estimators to provide fast estimation of fNL amplitudes, such as the [PolySpec](https://github.com/oliverphilcox/PolySpec) code, as described below. 

### Example Usage
The main code can be run on the command line as follows:
```
python main.py --config experiment_fiducial.yaml
```
The `.yaml` file describes the main code inputs and outputs, which are described in detail in the [example file](experiment_fiducial.yaml). Many of these parameters can also be specified on the command line. The key parameters are as follows:
- `datafile`: This specifies the input datafile, which contains {k1, k2, k3, S(k1,k2,k3)}, where S is a dimensionless shape function. The input k values should span the entire range of interest.
- `num_terms`: The maximum number of terms in the separable representation. This is typically 5 or less.
- `threshold`: The code scans over models with increasing numbers of terms, starting from one. When the accuracy level reaches `threshold` (or `num_terms` is reached), the code will exit.
- `symm_kind`: This specifies the type of symmetry to assume. The main options are `1` (full symmetry; 6 permutations) or `2` (cyclic symmetry; 3 permutations).

In the [Usage](Usage.ipynb) notebook, we demonstrate how to compute the `separate-bk` inputs, and visualize the outputs. We also show how to interface the code with [PolySpec](https://github.com/oliverphilcox/PolySpec), for measure the template amplitudes from CMB datasets.

### Authors
- [Kunhao Zhong](mailto:kunhaoz@sas.upenn.edu) (Penn)
- [Oliver Philcox](mailto:ohep2@cantab.ac.uk) (Stanford)
- Salvatore Samuel Sirletti (Ferrara)

### Dependencies
- Python 3
- [pytorch](https://pytorch.org/)
- yaml
- tqdm

### Reference
- Philcox, O. H. E., Zhong, K., Sirletti, S. S., "Separating the Inseparable: Constraining Arbitrary Primordial Bispectra with Cosmic Microwave Background Data", (2025) ([arXiv](https://arxiv.org/abs/2511.XXXXX))