import tensorflow as tf, keras
import numpy as np
import os

# Step 1: Define the image folders
folder_a = 'images/'
folder_b = 'downsampled_images/'

# Get list of image paths in each folder
image_paths_a = [os.path.join(folder_a, img) for img in os.listdir(folder_a) if img.endswith(('.jpg', '.jpeg', '.png'))]
image_paths_b = [os.path.join(folder_b, img) for img in os.listdir(folder_b) if img.endswith(('.jpg', '.jpeg', '.png'))]

print(f"Number of images in {folder_a}: {len(image_paths_a)}")
print(f"Number of images in {folder_b}: {len(image_paths_b)}")

# Step 2: Load and preprocess images
def load_and_preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [224, 224])
    img = keras.applications.resnet50.preprocess_input(img)
    return img

# Create datasets
path_ds_a = tf.data.Dataset.from_tensor_slices(image_paths_a)
path_ds_b = tf.data.Dataset.from_tensor_slices(image_paths_b)

image_ds_a = path_ds_a.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
image_ds_b = path_ds_b.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

# Batch and prefetch
batch_size = 32
image_ds_a = image_ds_a.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
image_ds_b = image_ds_b.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

# Step 3: Load the pretrained model
base_model = keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

# Step 4: Extract features
print("Extracting features for images in folder A...")
features_a = base_model.predict(image_ds_a, verbose=1)

print("Extracting features for images in folder B...")
features_b = base_model.predict(image_ds_b, verbose=1)

# Step 5: Normalize features
normalized_features_a = features_a / np.linalg.norm(features_a, axis=1, keepdims=True)
normalized_features_b = features_b / np.linalg.norm(features_b, axis=1, keepdims=True)

# Step 6: Compute similarity matrix
similarity_matrix = np.dot(normalized_features_a, normalized_features_b.T)
print("Similarity matrix shape:", similarity_matrix.shape)

# Step 7: Find most similar images
most_similar_indices = np.argmax(similarity_matrix, axis=1)
most_similar_scores = np.max(similarity_matrix, axis=1)
most_similar_images = [image_paths_b[idx] for idx in most_similar_indices]

# Step 8: Display results
for idx, (img_a, img_b, score) in enumerate(zip(image_paths_a, most_similar_images, most_similar_scores)):
    print(f"Image {img_a} is most similar to {img_b} with a similarity score of {score:.4f}")
