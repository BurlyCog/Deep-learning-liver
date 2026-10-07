# 3D Liver Segmentation from CT Images

A deep learning pipeline for **automatic liver segmentation from 3D CT scans** using a volumetric **3D U-Net** implemented with PyTorch and MONAI.

## What It Does

The system takes volumetric CT scans and produces a predicted liver segmentation mask. The pipeline covers the complete workflow:

**DICOM → NIfTI → Preprocessing → 3D U-Net → Training → Inference → Visualization**

## How We Built It

- Converted DICOM CT slices into **NIfTI volumes** using `dicom2nifti`.
- Standardized volumes with MONAI using:
  - voxel spacing normalization
  - RAS orientation
  - CT intensity windowing
  - foreground cropping
  - fixed-size volume resampling
- Built a **3D U-Net** with five encoder/decoder levels and residual units.
- Trained using **Dice Loss** and the Adam optimizer with CUDA acceleration.
- Used cached datasets to reduce repeated preprocessing overhead.
- Evaluated predictions using the **Dice coefficient**.
- Saved the best-performing model checkpoint.
- Used **sliding-window inference** to process 3D volumes within GPU memory constraints.
- Generated slice-wise visualizations comparing the CT image, ground truth, and predicted segmentation.

## What We Achieved

Compared with a basic image-segmentation workflow, the project provides a complete **volumetric medical-imaging pipeline** rather than treating CT scans as independent 2D images.

Key improvements include:

- **3D spatial context** through volumetric U-Net processing.
- Standardized CT preprocessing for more consistent model input.
- **GPU-accelerated training and inference** with PyTorch.
- Memory-efficient **sliding-window inference** for full CT volumes.
- Automated model selection through best-Dice checkpointing.
- Reproducible preprocessing through deterministic MONAI transforms.

> **Note:** The current implementation performs binary foreground segmentation (`background` vs `liver/tumor foreground`). It does not separately classify liver and tumor regions.
