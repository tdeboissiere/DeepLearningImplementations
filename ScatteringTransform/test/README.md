# Check implementation of the Scattering Transform

The pytorch code has been copied from [pyscatwave](https://github.com/edouardoyallon/pyscatwave).

The dependencies of the above library should be installed prior to running the tests.

# Test principle

- Since this repository is an exact port of [pyscatwave](https://github.com/edouardoyallon/pyscatwave), we make sure we get the same outputs for all the main functions.

# Running tests

    python run_test.py

optional arguments:

        -h, --help  show this help message and exit
        --M M
        --N N
        --J J


# Tests contents

- We check that the filter banks are equal for all filters
- Given a random input image, we check that the Scattering transform output is the same up to a tolerance of 1E-6.