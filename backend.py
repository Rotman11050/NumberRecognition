import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import tensorflow as tf
from keras.datasets import mnist

import tkinter as tk
from tkinter import messagebox

mnist = (train_X, train_y), (test_X, test_y) = mnist.load_data() 

print('X_train ' + str(train_X.shape))
print('Y_train ' + str(train_y.shape))
print('X_test ' + str(test_X.shape))
print('Y_test ' + str(test_y.shape))



