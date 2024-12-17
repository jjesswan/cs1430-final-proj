import cv2
import pandas as pd
import tensorflow as tf, keras
import numpy as np
import os
import cnn as OurCNN
import argparse

folder_a_default = 'images/original'
folder_b_default = 'images/reconstructed'
downsampled_default = 'images/downsampled_images'
true_labels = pd.read_csv('disease_true_labels.csv')
IMG_WIDTH = 224

def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--classify',
        action='store_true',
        help='''Run our custom cnn to both classify and feature extract between 
        downsampled images and their upsampled counterparts. This is turned off
        by default as it must train the CNN first.''')
    
    parser.add_argument(
        '--original',
        default=folder_a_default,
        help='''Name of directory to original dataset''')
    
    parser.add_argument(
        '--reconstructed',
        default=folder_b_default,
        help='''Name of directory to reconstructed dataset''')
    
    parser.add_argument(
        '--downsampled',
        default=downsampled_default,
        help='''Name of directory to downsampled dataset''')

    return parser.parse_args()

def validate_and_pair_images(folder, img_width):
    """ Filters out invalid data and matches images with their true labels.
        Returns nd array of valid image paths, integer labels, and raw image data.  """

    print("Process Folder: ", folder)
    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]

    directory = os.listdir(folder)
    data = [] 
    labels = []
    paths = []

    # Match labels with available training data
    for filename in directory:
        img_path= f"{folder}/{filename}"

        ext = os.path.splitext(filename)[1].lower()
        if ext not in image_extensions: continue

        row = true_labels[true_labels['Image Index'] == filename]
        if row.empty: continue
        else: 
            img = cv2.imread(img_path)
            if img is None: continue
            img = cv2.resize(img, dsize=(img_width, img_width), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            img = img.astype('float32')  
            label = row['Finding Labels'].iloc[0].split("|")[0] 
            labels.append(label)
            data.append(img)
            paths.append(img_path)


    labels = np.asarray(labels)
    paths = np.asarray(paths)
    unique_labels = np.unique(labels)
    raw_data = np.asarray(data)

    # map categorical labels to integers
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    # use map to convert the string labels to integers
    labels = np.asarray(list(map(label_mapping.get, labels)))
    labels = labels.astype(np.int32) 

    return paths, labels, raw_data


def load_and_preprocess_image(path):
    """ Preprocess from filepath into tensor image"""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_WIDTH, IMG_WIDTH])
    img = keras.applications.resnet50.preprocess_input(img)
    return img


def main():
    """ Main function. """

    folder_a = ARGS.original
    folder_b = ARGS.reconstructed
    downsampled = ARGS.downsampled


    # Make list of image paths in each folder
    # image_paths_a = [os.path.join(folder_a, img) for img in os.listdir(folder_a) if img.endswith(('.jpg', '.jpeg', '.png'))]
    # image_paths_b = [os.path.join(folder_b, img) for img in os.listdir(folder_b) if img.endswith(('.jpg', '.jpeg', '.png'))]


    # Validate img format and paths, pair with true labels
    image_paths_a, labels_a, data_a = validate_and_pair_images(folder_a, IMG_WIDTH)
    image_paths_b, labels_b, data_b = validate_and_pair_images(folder_b, IMG_WIDTH)
    downsampled, labels_down, data_down = validate_and_pair_images(downsampled, IMG_WIDTH)

    # Build tensor data from valid images
    path_ds_a = tf.data.Dataset.from_tensor_slices(image_paths_a)
    path_ds_b = tf.data.Dataset.from_tensor_slices(image_paths_b)

    image_ds_a = path_ds_a.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    image_ds_b = path_ds_b.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    batch_size = 32
    image_ds_a = image_ds_a.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    image_ds_b = image_ds_b.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    if ARGS.classify:
        # Build our CNN Model
        cnn, base_model = OurCNN.getCNN(data_a, labels_a, IMG_WIDTH)

        # Classification on data B
        test_loss, test_accuracy = cnn.evaluate(data_b, labels_b, batch_size=32)
        print("Classification loss on b : ", test_loss, "Accuracy on b: ", test_accuracy)

        # Classification on downsampled data:
        test_loss, test_accuracy = cnn.evaluate(data_down, labels_down, batch_size=32)
        print("Classification loss on downsampled : ", test_loss, "Accuracy on downsampled: ", test_accuracy)
    else: 
        # ResNet50 Model
        base_model = keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(IMG_WIDTH, IMG_WIDTH, 3))


    # Feature Extraction
    features_a = base_model.predict(image_ds_a, verbose=1)
    features_b = base_model.predict(image_ds_b, verbose=1)

    normalized_features_a = features_a / np.linalg.norm(features_a, axis=1, keepdims=True)
    normalized_features_b = features_b / np.linalg.norm(features_b, axis=1, keepdims=True)

    # Compute similarity matrix
    similarity_matrix = np.dot(normalized_features_a, normalized_features_b.T)

    most_similar_indices = np.argmax(similarity_matrix, axis=1)
    most_similar_scores = np.max(similarity_matrix, axis=1)
    most_similar_images = [image_paths_b[idx] for idx in most_similar_indices]

    # Results
    count = 0
    worked = 0
    total = 0
    for idx, (img_a, img_b, score) in enumerate(zip(image_paths_a, most_similar_images, most_similar_scores)):
        print(f"Image {img_a} is most similar to {img_b} with a similarity score of {score:.4f}")
        count += 1
        total += score

        slice_amount = len(folder_b) - 1
        b_slice = img_b[slice_amount:]

        # print(img_a[8:])
        # print(b_slice)

        if img_a[8:] == b_slice:
            worked += 1

        print("count: ", count)
        print("matched: ", worked)

    result = total/count
    print("average score:", result)    


# Make arguments global
ARGS = parse_args()
main()







