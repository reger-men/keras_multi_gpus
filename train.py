# USAGE
# python train.py --output single_gpu.png
# python train.py --output multi_gpu.png --gpus 4

# set the matplotlib backend so figures can be saved in the background
# (uncomment the lines below if you are using a headless server)
# import matplotlib
# matplotlib.use("Agg")

# import the necessary packages
from pyimagesearch.minigooglenet import MiniGoogLeNet
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.utils.training_utils import multi_gpu_model
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to output plot")
ap.add_argument("-g", "--gpus", type=int, default=1,
	help="# of GPUs to use for training")
args = vars(ap.parse_args())

# grab the number of GPUs and store it in a conveience variable
G = args["gpus"]

# definine the total number of epochs to train for along with the
# initial learning rate
NUM_EPOCHS = 70
INIT_LR = 5e-3

def poly_decay(epoch):
	# initialize the maximum number of epochs, base learning rate,
	# and power of the polynomial
	maxEpochs = NUM_EPOCHS
	baseLR = INIT_LR
	power = 1.0

	# compute the new learning rate based on polynomial decay
	alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power

	# return the new learning rate
	return alpha

# load the training and testing data, converting the images from
# integers to floats
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")
testX = testX.astype("float")

# apply mean subtraction to the data
mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# construct the image generator for data augmentation and construct
# the set of callbacks
aug = ImageDataGenerator(width_shift_range=0.1,
	height_shift_range=0.1, horizontal_flip=True,
	fill_mode="nearest")
callbacks = [LearningRateScheduler(poly_decay)]

# check to see if we are compiling using just a single GPU
if G <= 1:
	print("[INFO] training with 1 GPU...")
	model = MiniGoogLeNet.build(width=32, height=32, depth=3,
		classes=10)

# otherwise, we are compiling using multiple GPUs
else:
	print("[INFO] training with {} GPUs...".format(G))

	# we'll store a copy of the model on *every* GPU and then combine
	# the results from the gradient updates on the CPU
	with tf.device("/cpu:0"):
		# initialize the model
		model = MiniGoogLeNet.build(width=32, height=32, depth=3,
			classes=10)
	
	# make the model parallel
	model = multi_gpu_model(model, gpus=G)

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=INIT_LR, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=64 * G),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // (64 * G),
	epochs=NUM_EPOCHS,
	callbacks=callbacks, verbose=2)

# grab the history object dictionary
H = H.history

# plot the training loss and accuracy
N = np.arange(0, len(H["loss"]))
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H["loss"], label="train_loss")
plt.plot(N, H["val_loss"], label="test_loss")
plt.plot(N, H["acc"], label="train_acc")
plt.plot(N, H["val_acc"], label="test_acc")
plt.title("MiniGoogLeNet on CIFAR-10")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()

# save the figure
plt.savefig(args["output"])
plt.close()