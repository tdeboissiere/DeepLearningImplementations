# Sobolev training for neural networks

theano + lasagne implementation of [Sobolev Training for Neural Networks](https://arxiv.org/abs/1706.04859)

![Sobolev with 20 pts](figures/tang_20pts.gif)
![Sobolev with 100 pts](figures/tang_100pts.gif)


## Dependencies

- lasagne (tested on 0.2.dev1)
- theano (tested on 0.9.0)
- matplotlib (tested on 2.0.1)
- tqdm
- numpy
- natsort


## Uage

Main command:

	python main,py

Arguments:


	--nb_epoch NB_EPOCH   Number of training epochs
	--batch_size BATCH_SIZE
	                      Batch size
	--npts NPTS         Number of training points
	--learning_rate LEARNING_RATE
	                      Learning rate
	--sobolev_weight SOBOLEV_WEIGHT
	                      How much do we weight the Sobolev function


## Run a batch of experiments

	bash run_experiments.sh


## Create gif

	bash run_experiments
	python make_gif.py


## Notes

- The architecture of the NN is the same as in the original paper.
- We plot the loss curves to give some more perspective.
- Initially had a hard time reproducing results. Inspection of loss curves show you just have to train longer until Soboleb loss and MSE loss have similar magnitude. Or increase the weight on the Sobolev loss.