import os
import cv2
import numpy as np
from tqdm import tqdm
import time

def downsample_image(image_path, output_dir='downsampled_images', scale=0.25, noise_prob=1, noise_std=5, blur_ksize=3, artifact_prob=1, artifact_quality=80):
    """
    Downsample image with noise, blur, and artifacts
    
    Args:
        image_path: Image path
        output_dir: Directory to save the downsampled image
        scale: Downsampling factor
        noise_prob: probability of adding Gaussian noise
        noise_std: standard deviation of Gaussian noise
        blur_ksize: kernel size for Gaussian blur
        artifact_prob: probability of adding compression artifacts
        artifact_quality: quality of compression artifacts
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load & downsample
    image = cv2.imread(image_path)
    if image is None: return
    # print("image path:", image_path)
    height, width = image.shape[:2]
    downsampled = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    
    # Gaussian noise
    if np.random.uniform(0, 1) < noise_prob:
        noise = np.random.normal(0, noise_std, downsampled.shape).astype(np.uint8)
        noisy_image = cv2.add(downsampled, noise)
    else:
        noisy_image = downsampled

    # Gaussian blur
    blurred_image = cv2.GaussianBlur(noisy_image, (blur_ksize, blur_ksize), 0)

    # Compression artifacts
    if np.random.uniform(0, 1) < artifact_prob:
        _, compressed_image = cv2.imencode('.jpg', blurred_image, [cv2.IMWRITE_JPEG_QUALITY, artifact_quality])
        final_image = cv2.imdecode(compressed_image, cv2.IMREAD_COLOR)
    else:
        final_image = blurred_image

    # Save image
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, final_image)


def downsample_image_from_zip(zip_ref, image_path, output_dir='downsampled_images', scale=0.25, noise_prob=1, noise_std=5, is_blur=True, blur_ksize=3, artifact_prob=1, artifact_quality=80):
    """
    Downsample image with noise, blur, and artifacts
    
    Args:
        image_path: Image path
        output_dir: Directory to save the downsampled image
        scale: Downsampling factor
        noise_prob: probability of adding Gaussian noise
        noise_std: standard deviation of Gaussian noise
        blur_ksize: kernel size for Gaussian blur
        artifact_prob: probability of adding compression artifacts
        artifact_quality: quality of compression artifacts
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load & downsample
    # image = cv2.imread(image_path)
    with zip_ref.open(image_path) as file:
        image_bytes = file.read()
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), 1)
    height, width = image.shape[:2]
    downsampled = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    
    # Gaussian noise
    if np.random.uniform(0, 1) < noise_prob:
        noise = np.random.normal(0, noise_std, downsampled.shape).astype(np.uint8)
        noisy_image = cv2.add(downsampled, noise)
    else:
        noisy_image = downsampled

    # Gaussian blur
    if is_blur:
        blurred_image = cv2.GaussianBlur(noisy_image, (blur_ksize, blur_ksize), 0)
    else:
        blurred_image = noisy_image

    # Compression artifacts
    if np.random.uniform(0, 1) < artifact_prob:
        _, compressed_image = cv2.imencode('.jpg', blurred_image, [cv2.IMWRITE_JPEG_QUALITY, artifact_quality])
        final_image = cv2.imdecode(compressed_image, cv2.IMREAD_COLOR)
    else:
        final_image = blurred_image

    # Save image
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, final_image)


image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]

def main(input_dir='images/original', output_dir='downsampled_images1', scale=0.5, noise_std=5, blur_ksize=3):
    """Process all images in input_dir"""

    image_files = os.listdir(input_dir)
    # print(f'Processing {len(image_files)} images...')
    
    # start_time = time.time()

    for image_file in tqdm(image_files, desc='Processing images'):
        image_path = os.path.join(input_dir, image_file)
        ext = os.path.splitext(image_file)[1].lower()
        if ext not in image_extensions: continue
        downsample_image(image_path, output_dir, scale, noise_std, blur_ksize)

    # elapsed_time = time.time() - start_time
    #print(f'Downsampling complete. Time elapsed: {elapsed_time:.2f} seconds.')


if __name__ == '__main__':
    main(input_dir='images/original', output_dir='downsampled_images1', scale=0.5, noise_std=5, blur_ksize=3)