import tensorflow as tf, keras
import numpy as np
import os

# Train model on disease labels + original image data

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import model 
import cv2

def getCNN(data, labels, img_width, split=.8):   
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    n = len(labels)
    print("Number of possible classes: ", num_classes)

    # map categorical labels to integers
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    # convert string labels to integers
    labels = np.asarray(list(map(label_mapping.get, labels)))
    labels = labels.astype(np.int32) 

    # split data into train,test sets
    train_split = int(np.floor(.8*n))
    print("Train split: ", train_split, " | Test split: ", n-train_split)

    train_data = data[:train_split]
    train_labels = labels[:train_split]

    test_data = data[train_split:]
    test_labels = labels[train_split:]

    print("== Start Training ==")

    # train using labels and data
    cnn, feature_extractor = model.getModel(train_data, train_labels, test_data, test_labels, num_classes, img_width)

    print("== End Training ==")

    return cnn, feature_extractor
    

