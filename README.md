# Image Matrix Completion via Iterative SVT

This project implements an image reconstruction and matrix completion algorithm using **Iterative Singular Value Thresholding (SVT)** and **Projected Gradient Descent (PGD)**. It demonstrates how to recover missing pixel data from masked images by leveraging low-rank matrix properties and nuclear norm regularization.

## Overview

In this project, images are treated as matrices that are partially observed due to random patch masking. The reconstruction process aims to solve the following optimization problem:

$$\min_{X} \frac{1}{2} \| P_{\Omega}(X - M) \|_F^2 + \tau \| X \|_*$$

where:
- $M$ is the observed image data.
- $P_{\Omega}$ is the projection onto the observed indices.
- $\| X \|_*$ is the nuclear norm (sum of singular values), serving as a convex proxy for the rank of the matrix.

## Key Features

- **Iterative SVT Algorithm**: Implements a Projected Gradient Descent approach combined with Soft-Thresholding of singular values.
- **Patch Masking**: A custom PyTorch `PatchMaskWrapper` that divides images into patches and randomly masks them based on a specified ratio.
- **RGB Reconstruction**: Per-channel reconstruction logic to handle standard color images.
- **Performance Tracking**: Records reconstruction loss and matrix rank evolution throughout the iterations.
- **Visualization**: Tools to compare Original, Masked, and Reconstructed images.

## Requirements

To run this notebook, you will need the following libraries:

- `torch`
- `torchvision`
- `matplotlib`
- `numpy`

```bash
pip install torch torchvision matplotlib numpy
```

## Methodology

### 1. Patch Masking
The `PatchMaskWrapper` class takes a standard dataset (such as STL-10) and applies a random binary mask to the images.

* **Patch Size**: Customizable (default is $8 \times 8$).
* **Mask Ratio**: Defines the percentage of patches to be hidden (e.g., $0.5$ for 50% masking).

### 2. SVT & Projected Gradient Descent
The algorithm iterates through the following steps to recover the low-rank structure of the image:

1.  **Gradient Step**: Update the current estimate $X$ by moving it closer to the ground truth pixels in the observed set $\Omega$.
2.  **Singular Value Decomposition (SVD)**: Perform SVD on the current estimate to obtain the singular values.
3.  **Soft-Thresholding**: Apply the threshold $\tau$ to the singular values: 
    $$S_\tau(\sigma) = \text{sgn}(\sigma)\max(|\sigma| - \tau, 0)$$
    This step forces the matrix to be low-rank by shrinking small singular values to zero.
4.  **Reconstruction**: Re-assemble the matrix using the thresholded singular values and the original $U$ and $V$ matrices.

## Results

The implementation includes visualization of the following:

* **Ground Truth**: The original uncorrupted image.
* **Masked Input**: The image with missing patches, serving as the input for the completion algorithm.
* **SVT Reconstruction**: The final result after iterative completion, demonstrating how the low-rank assumption helps fill in structured information and textures.

## Usage

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/sihhanchao/SVT-MAE-reconstruction.git
    ```
2.  **Install Dependencies**:
    Ensure you have `torch`, `torchvision`, `matplotlib`, and `numpy` installed.
3.  **Open the Notebook**:
    Launch `SVT_MAE_Completion.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
4.  **Run the Cells**:
    Execute the cells sequentially to download the STL-10 dataset and run the reconstruction demonstration.


