# Super-Resolution and Texture Transfer for Image Processing

This project leverages a pre-trained ESRGAN (Enhanced Super-Resolution Generative Adversarial Network) model to enhance image resolution and apply a texture transfer effect. The code processes an on-model image, upscales it for high-quality output, and applies a texture detail from a close-up texture image to the specified garment area in the on-model image.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)

## Overview

This project accomplishes two main tasks:

1. **Image Super-Resolution**: Using ESRGAN, it enhances the resolution of both an on-model image and a close-up texture image.
2. **Texture Transfer**: Applies texture details from the close-up image to the garment area in the on-model image using mask-based blending techniques.

## Requirements

The project requires the following dependencies:

- Python 3.x
- PyTorch
- OpenCV
- NumPy

You can install the necessary libraries using:

```bash
pip install torch torchvision opencv-python-headless numpy

```
