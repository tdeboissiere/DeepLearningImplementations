import os
# Disable Tensorflow's INFO and WARNING messages
# See http://stackoverflow.com/questions/35911252
if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import sys
sys.path.append("../src/utils")
import filters_bank
import scattering


def run_filter_bank(M, N, J):

    filters = filters_bank.filters_bank(M, N, J)
    d_save = {}
    # Save phi
    d_save["phi"] = {}
    for key in filters["phi"].keys():
        val = filters["phi"][key]
        if isinstance(val, tf.Tensor):
            val_numpy = val.eval(session=tf.Session())
            d_save["phi"][key] = val_numpy
    # Save psi
    d_save["psi"] = []
    for elem in filters["psi"]:
        d = {}
        for key in elem.keys():
            val = elem[key]
            if isinstance(val, tf.Tensor):
                val_numpy = val.eval(session=tf.Session())
                d[key] = val_numpy
        d_save["psi"].append(d)

    return d_save


def run_scattering(X, use_XLA=False):

    # Ensure NCHW format
    assert X.shape[1] < min(X.shape[2:])

    M, N = X.shape[2:]

    X_tf = tf.placeholder(tf.float32, shape=[None,] + list(X.shape[1:]))

    scat = scattering.Scattering(M=M, N=N, J=2, check=True)
    S = scat(X_tf)

    # Create session
    config = tf.ConfigProto()
    if use_XLA:
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    sess = tf.Session(config=config)
    S_out = sess.run(S, feed_dict={X_tf: X})

    return S_out
