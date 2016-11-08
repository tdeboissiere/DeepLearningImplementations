# Implementation of the Eve optimizer


This is a keras implementation of [Improving Stochastic Gradient Descent With Feedback](https://arxiv.org/pdf/1611.01505v1.pdf).

Check [this page](https://github.com/jayanthkoushik/sgd-feedback/blob/master/src/eve.py) for the authors' original implementation of Eve.

## Usage

You can either import this optimizer:

    from Eve import Eve
    Eve_instance = Eve(lr=0.001, beta_1=0.9, beta_2=0.999,
                        beta_3=0.999, small_k=0.1, big_K=10,
                        epsilon=1e-8)


Or copy the Eve class to keras/optimizers.py and use it as any other optimizer.