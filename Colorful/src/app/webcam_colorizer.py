import os
import cv2
import sys
import h5py
import argparse
import numpy as np
from skimage import color
sys.path.append("../model")
import models_colorful as models_colorful


def webcam_colorizer(data_file, model_name, epoch, T, out_h, out_w, video_path=None):

    # Load the array of quantized ab value
    q_ab = np.load("../../data/processed/pts_in_hull.npy")
    nb_q = q_ab.shape[0]
    batch_size = 1

    # Load and rescale data
    with h5py.File(data_file, "r") as hf:
        X = hf["validation_lab_data"]
        c, h, w = X.shape[1:]

    img_size = int(os.path.basename(data_file).split("_")[1])

    # Load colorization model
    color_model = models_colorful.load(model_name, nb_q, (1, h, w), batch_size)
    color_model.load_weights("../../models/%s/%s_weights_epoch%s.h5" %
                             (model_name, model_name, epoch))

    #################
    # APP
    #################
    if video_path != "":
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0)

    print '\n\nPRESS q/Q to QUIT\n'

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret is True:
            img = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_AREA)
            img = img[:, :, ::-1]  # BGR to RGB

            # Convert to Lab
            img = color.rgb2lab(img)

            img_gray = img[:, :, 0].reshape((1, 1, img_size, img_size))

            X_black = img_gray / 100.
            X_colorized = color_model.predict(X_black)[:, :, :, :-1]

            # Format X_colorized
            X_colorized = X_colorized.reshape((1 * h * w, nb_q))

            # Reweight probas
            X_colorized = np.exp(np.log(X_colorized) / T)
            X_colorized = X_colorized / np.sum(X_colorized, 1)[:, np.newaxis]

            # Reweighted
            q_a = q_ab[:, 0].reshape((1, nb_q))
            q_b = q_ab[:, 1].reshape((1, nb_q))

            X_a = np.sum(X_colorized * q_a, 1).reshape((1, 1, h, w))
            X_b = np.sum(X_colorized * q_b, 1).reshape((1, 1, h, w))

            X_colorized = np.concatenate((100 * X_black, X_a, X_b), axis=1).transpose(0, 2, 3, 1)
            X_colorized = color.lab2rgb(X_colorized[0])

            X_black = X_black[0]
            X_black = np.repeat(X_black, 3, axis=0).transpose(1,2,0)
            X = np.concatenate((X_black, X_colorized[:, :, ::-1]), 1)
            X = cv2.resize(X, (out_w, out_h))

            cv2.imshow('Colorizer (q/Q: Quit)',X)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:

            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Colorize app')
    parser.add_argument('data_file', type=str, help="Path to HDF5 containing the data")
    parser.add_argument('epoch', type=int, help='Epoch of saved weights')
    parser.add_argument('--model_name', type=str, default="simple_colorful",
                        help="Model name. Choose simple_colorful or colorful")
    parser.add_argument('--T', default=0.1, type=float,
                        help="Temperature to change color balance. If T = 1: desaturated. If T~0 vivid")
    parser.add_argument('--out_h', default=420, type=int, help="Width of ouput image")
    parser.add_argument('--out_w', default=640, type=int, help="Height of ouput image")
    parser.add_argument('--video_path', default="", type=str, help="Path to B&W video to colorize")

    args = parser.parse_args()

    assert args.model_name in ["colorful", "simple_colorful"]

    webcam_colorizer(args.data_file,
                     args.model_name,
                     args.epoch,
                     args.T,
                     args.out_h,
                     args.out_w,
                     video_path=args.video_path)
