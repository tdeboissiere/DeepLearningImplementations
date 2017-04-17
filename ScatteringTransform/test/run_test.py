import run_tf
import run_pytorch
import argparse
import numpy as np
import sys
sys.path.append("../src/utils")
import logging_utils as lu


def test_filters_bank(M, N, J):

    d_tf = run_tf.run_filter_bank(M, N, J)
    d_pytorch = run_pytorch.run_filter_bank(M, N, J)

    for key in d_tf["phi"].keys():
        arr_tf = d_tf["phi"][key]
        arr_pytorch = d_pytorch["phi"][key]
        try:
            assert np.all(np.isclose(arr_tf, arr_pytorch))
            lu.print_bright_green("phi, key: %s" % key, "OK")
        except AssertionError:
            lu.print_bright_red("phi, key: %s" % key, "FAIL")
            return

    for idx, (elem_tf, elem_pytorch) in enumerate(zip(d_tf["psi"], d_pytorch["psi"])):
        for key in elem_tf:
            arr_tf = elem_tf[key]
            arr_pytorch = elem_pytorch[key]
            try:
                assert np.all(np.isclose(arr_tf, arr_pytorch))
                lu.print_bright_green("psi, index: %s, key: %s" % (idx, key), "OK")
            except AssertionError:
                lu.print_bright_red("psi, index: %s, key: %s" % (idx, key), "FAIL")
                return

    lu.print_green("All filters equal")


def test_scattering(M, N, J, use_cuda, use_XLA):

    np.random.seed(0)
    X = np.random.normal(0, 1, size=(16, 3, M, N))

    S_pytorch = run_pytorch.run_scattering(X, use_cuda)
    S_tf = run_tf.run_scattering(X, use_XLA)

    for idx, (Stf, Spytorch) in enumerate(zip(S_tf, S_pytorch)):
        try:
            assert np.all(np.isclose(Stf, Spytorch, atol=1E-6))
            lu.print_bright_green("S coeff, index: %s" % idx, "OK")
        except AssertionError:
            lu.print_bright_red("S coeff, index: %s" % idx, "FAIL")
            return

    lu.print_green("Scattering results equal")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--M', type=int, default=32)
    parser.add_argument('--N', type=int, default=32)
    parser.add_argument('--J', type=int, default=2)
    parser.add_argument('--use_cuda', action="store_true")
    parser.add_argument('--use_XLA', action="store_true")

    args = parser.parse_args()

    test_filters_bank(args.M, args.N, args.J)
    test_scattering(args.M, args.N, args.J, args.use_cuda, args.use_XLA)
