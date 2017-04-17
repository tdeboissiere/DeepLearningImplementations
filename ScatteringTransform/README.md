# Scaling the Scattering Transform: Deep Hybrid Networks

- Pure tensorflow implementation of the scattering transform and hybrid networks.
- No need for external dependencies, only tensorflow functions are used
- Depending on the settings (e.g. use_XLA), the code is between 3x to 5x slower than [pyscatwave](https://github.com/edouardoyallon/pyscatwave).
- Since only tensorflow primitives are involved, it is possible to backprop through the Scattering Transform (this functionality will soon be added to [pyscatwave](https://github.com/edouardoyallon/pyscatwave)).


**Sources:**

- [Scaling the Scattering Transform: Deep Hybrid Networks](https://arxiv.org/abs/1703.08961)
- [pyscatwave](https://github.com/edouardoyallon/pyscatwave)

This repository is a simple adapatation of [pyscatwave](https://github.com/edouardoyallon/pyscatwave) with tensorflow primitives only.

**Copyright (c) 2017, Eugene Belilovsky (INRIA), Edouard Oyallon (ENS) and Sergey Zagoruyko (ENPC)**
**All rights reserved.**

# Requirements

## python modules

- tensorflow 1.0
- h5py
- matplotlib
- opencv 3
- numpy
- tqdm
- colorama
- terminaltables
- parmap

You should also install the dependencies of [pyscatwave](https://github.com/edouardoyallon/pyscatwave) to run the tests.


# Running the code

Follow [these instructions](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/ScatteringTransform/src/model)


# Running tests

Follow [these instructions](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/ScatteringTransform/test)