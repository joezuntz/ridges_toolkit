# Emulator
Emulator for tangential shear prediction from ridges. **atm works for one bin pair only**
A few examples can be found in `notebooks/`.


### Structure
- `build_dataset.py` uses the ridge analysis output to create two dataset, one for training and validating the emulator, and one for testing. Saved in `data/`.
- `load_data.py` contains functions to loads and process a dataset.
- `model.py` builds a NN model.
- `train.py` trains the model. The result is saved in `models/`.
- `prediction.py` makes a shear signal prediction given a cosmology as input.
- `plot_prediciton.py` compares two predicted and true signals, showing also their fractional error.



### How to use
**work in progress, a few things needs to be changed.**

