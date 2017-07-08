# Experiments with MNIST and SELU

Pytorch implementation of some experiments from [Self-Normalizing Networks](https://arxiv.org/pdf/1706.02515.pdf)

## Dependencies

- python (tested on Anaconda python 3.6.1)
- pytorch (tested on 0.1.12_2)
- sklearn (tested on 0.18.1)
- matplotlib (tested on 2.0.1)
- tqdm
- numpy


## Uage

Main command:

	python main.py

Arguments:

	  --model MODEL         Model name, RELUNet or SELUNet
	  --n_inner_layers N_INNER_LAYERS
	                        Number of inner hidden layers
	  --hidden_dim HIDDEN_DIM
	                        Hidden layer dimension
	  --dropout DROPOUT     Dropout rate
	  --use_cuda            Use CUDA
	  --nb_epoch NB_EPOCH   Number of training epochs
	  --batchnorm           Whether to use BN for RELUNet
	  --batch_size BATCH_SIZE
	                        Batch size
	  --optimizer OPTIMIZER
	                        Optimizer
	  --learning_rate LEARNING_RATE
	                        Learning rate


## Run a batch of experiments

Modify `run_experiments.sh` as needed then run

	bash run_experiments.sh


## Plot results

Run a few experiments. Results are saved in a `results` folder.
Modify `plot_results.py` to select your experiments, then run:


	python plot_results.py


## Notes

- The architecture of the NN is the same as in the original paper.
- We plot the loss curves to give some more perspective.
- Initially had a hard time reproducing results. Inspection of loss curves show you just have to train longer until Soboleb loss and MSE loss have similar magnitude. Or increase the weight on the Sobolev loss.