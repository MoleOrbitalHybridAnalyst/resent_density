Archive for https://doi.org/10.48550/arXiv.2402.12335

# software/pyscf
A copy of PySCF for data generation. Not needed for model inference.

# software/resnet
Implementation of the ResNet model for density prediction.

# data/
Some examples of data and scripts for generating them. Filenames with atomh1e indicate the SAD guess density, atomh1e2 indicates the worsened SAD guess, and otherwise means the DFT converged density. \
Full data are available at \
(1) Li, C., Sharir, O., Yuan, S., & Chan, G. K. (2024). QM9 DFT Electron Density [Data set]. CaltechDATA. https://doi.org/10.22002/7vr2f-0r732 \
(2) Li, C., Sharir, O., Yuan, S., & Chan, G. K. (2024). C7O2H10 Isomer DFT Electron Density [Data set]. CaltechDATA. https://doi.org/10.22002/04d1z-1d608 \
(3) Li, C., Sharir, O., Yuan, S., & Chan, G. K. (2024). GTP-in-Microtubule DFT Electron Density [Data set]. CaltechDATA. https://doi.org/10.22002/v5tec-g6p43

# examples/
An example of training and an example of inference.

# Pre-trained Models
Pre-trained models are available at doi.org/10.6084/m9.figshare.25365508.

# Software Dependencies
All training and prediction were run with PyTorch 1.11.0, Python 3.10.10 on Ubuntu 22.04.3 LTS.\
All data generation was performed with PySCF (see software/pyscf), NumPy 1.24.2, SciPy 1.8.0, H5py 3.8.0 and Python 3.10.4 on Fedora 7.2 (Maipo).

# Installation
Add the path to software/resnet to the environmental variable PYTHONPATH
