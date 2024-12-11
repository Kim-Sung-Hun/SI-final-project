# VITAL: Versatile Image Transformation and Automated Labeling

## Introduction
VITAL is a novel framework designed to enhance medical image processing by combining **multi-conditional augmentation** and **automated labeling**. This ensures high-quality image generation tailored to specific conditions, improving both dataset quality and diagnostic efficiency.

## Key Features
- **Multi-Conditional Augmentation**: Tailors data augmentation based on diverse conditions, such as image modality and medical attributes.
- **Automated Labeling**: Uses conditional augmentation to generate precise labels, reducing manual effort and errors.
- **High Fidelity and Precision**: Ensures generated images retain clinical relevance through intelligent integration of multiple conditions.

## Methodology

### Diffusion-Based Backbone
- Utilizes **DiffAE** for high-quality image generation.

### Multi-Conditional Decomposition
VITAL decomposes images into distinct elements for precise control:
- **Mask**: Adjusted using ground truth (GT) and convolution layers.
- **Color**: Extracted via ColorThief using color quantization and selection.
- **Fovea/Optic Disc (OD)**: Coordinates manually labeled for precision.
- **Style**: Captures stylistic and texture features with **SimSiam** and jigsaw augmentation.

### Inference Workflow
1. **Mask Generation**: Generated using DDIM for high-quality synthesis.
2. **Feature Estimation**: OD, fovea, and color estimated with Kernel Density Estimation (KDE).
3. **Condition Integration**: All elements combined in DiffAE for controlled image generation.
4. **Validation and Output**: Ensures compliance with input conditions and applies fine-tuning if needed.

## Results
- **Dataset**: Used 1,745 image-mask pairs from datasets like DRIVE, CHASEDB, and ARIA.
- **Controlled Variation**: Demonstrated precise control by modifying specific conditions (e.g., mask, color, OD).
- **High-Quality Outputs**: Ensured consistent quality and clinical relevance across scenarios.

## Future Directions
- Enhance the **mask condition** by preserving spatial details with 2D tensor representations.
- Develop advanced conditioning mechanisms to improve segmentation accuracy and robustness.

## Applications
VITAL can be applied to:
- **Medical Image Augmentation**: Generate clinically relevant data for training.
- **Feature Detection**: Train models for fovea or optic disc detection.
- **Diagnostic Support**: Generate high-fidelity images to aid medical diagnosis.



