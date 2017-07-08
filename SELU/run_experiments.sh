# RELU
python main.py --use_cuda --n_inner_layers 4 --model RELUNet --learning_rate 1E-5 --nb_epoch 20 --batchnorm --optimizer Adam
python main.py --use_cuda --n_inner_layers 8 --model RELUNet --learning_rate 1E-5 --nb_epoch 20 --batchnorm --optimizer Adam
python main.py --use_cuda --n_inner_layers 16 --model RELUNet --learning_rate 1E-5 --nb_epoch 20 --batchnorm --optimizer Adam
python main.py --use_cuda --n_inner_layers 32 --model RELUNet --learning_rate 1E-5 --nb_epoch 20 --batchnorm --optimizer Adam
# SELU
python main.py --use_cuda --n_inner_layers 4 --model SELUNet --learning_rate 1E-5 --nb_epoch 20 --optimizer Adam
python main.py --use_cuda --n_inner_layers 8 --model SELUNet --learning_rate 1E-5 --nb_epoch 20 --optimizer Adam
python main.py --use_cuda --n_inner_layers 16 --model SELUNet --learning_rate 1E-5 --nb_epoch 20 --optimizer Adam
python main.py --use_cuda --n_inner_layers 32 --model SELUNet --learning_rate 1E-5 --nb_epoch 20 --optimizer Adam
# SELU + dropout
python main.py --use_cuda --n_inner_layers 4 --model SELUNet --learning_rate 1E-5 --nb_epoch 200 --dropout 0.05
python main.py --use_cuda --n_inner_layers 8 --model SELUNet --learning_rate 1E-5 --nb_epoch 200 --dropout 0.05
python main.py --use_cuda --n_inner_layers 16 --model SELUNet --learning_rate 1E-5 --nb_epoch 200 --dropout 0.05
python main.py --use_cuda --n_inner_layers 32 --model SELUNet --learning_rate 1E-5 --nb_epoch 200 --dropout 0.05
