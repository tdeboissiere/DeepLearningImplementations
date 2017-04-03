import os
import cv2
from tqdm import tqdm
from keras.datasets import mnist

data_dir = "/home/tmain/Desktop/DeepLearning/Data/mnist"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

X = mnist.load_data()[0][0]

for i in tqdm(range(len(X))):
    cv2.imwrite(os.path.join(data_dir, "%s.jpg" % i), X[i])
