# cs1430-final-proj

This project performs medical image superresolution through diffusion and compares their classification accuracy

## Authors
- Jessica Wan
- Michelle Liu
- Yen Chu
- Zilai Zeng

## Data
- [NIH Chest X-Rays images_001](https://www.kaggle.com/datasets/nih-chest-xrays/data?resource=download&select=images_001)
  - So far just 4999 images found in `images_001`
  - To use: put images in `images/`

## Corruption-Restoration Pipeline
- Setup ResShift environment following `ResShift/README.md`
- Under ResShift folder, run `pip install -e .`
- Download zipped image data, for example, download NIH `images_001.zip` and put it under `images/source` folder
- Run corruption-restoration pipeline, examples are shown below:
  - 4x Downsample --> Mildest Corruption
    ```shell
    python corrupt_to_restore.py \
      -i images/source/images_001.zip \
      -c images/corrupted \
      -o images/reconstructed \
      --downscale 0.25 \
      --upscale 4 \
      --task realsr \
      --version v3 \
      --num_samples 50
    ```
  - Blurring (w/ Gaussian Kernel Size 25)
    ```shell
    python corrupt_to_restore.py \
      -i images/source/images_001.zip \
      -c images/corrupted \
      -o images/reconstructed \
      --downscale 1 \
      --upscale 1 \
      --task deblur \
      --num_samples 50 \
      --blur_ksize 25
    ```
  - 4x Downsample + Compression (w/ 10% quality) --> Hardest Corruption
    ```shell
    python corrupt_to_restore.py \
      -i images/source/images_001.zip \
      -c images/corrupted \
      -o images/reconstructed \
      --downscale 0.25 \
      --upscale 4 \
      --task realsr \
      --version v3 \
      --num_samples 50 \
      --artifact_prob 1 \
      --artifact_quality 10
    ```

## Metric Calculation
Run the following command to calculate the quality of reconstructed images:
```shell
python calc_metrics.py -s images/source/images_001.zip -r images/reconstructed
```