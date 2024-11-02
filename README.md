# Generating High-Fidelity Retinal Images with Vessel-Based Conditioning Using Pix2Pix and Outpainting

## Authors
Sunghun Kim, Junphyo Im

## Abstract
Retinal image generation is critical in medical imaging, assisting in diagnostics and treatment planning. Traditional diffusion-based generative models struggle to accurately incorporate vessel structures when these are used as conditioning elements, often leading to blurred or inaccurate representations in the final image. To address this limitation, we propose a two-stage generation approach. First, we create a detailed vessel structure using a Pix2Pix model, converting grayscale vessel masks into color vessel masks. Next, we employ outpainting techniques to complete the remaining regions of the retinal image, using the generated vessel structure as a foundation. This approach ensures the vessel structures are accurately represented, enhancing both the realism and diagnostic value of the generated retinal images.

## Introduction
In medical imaging and analysis, generating high-quality retina images is critical for diagnosis and treatment planning. Traditional diffusion-based generative methods face challenges when vessels are used as conditioning elements; the vessel structures are often not reflected accurately in the final output, leading to less reliable results. This project aims to address this limitation by first generating a vessel structure, which is then used as the basis for outpainting the remaining retinal regions. This two-step process enhances vessel clarity and improves overall image quality.

## Related Works
Previous studies have explored various methods for generative retinal imaging, often focusing on diffusion models for their ability to produce high-quality images. However, these models tend to have limitations in preserving vessel details when used with conditional information. Recent advancements, such as Pix2Pix, have shown promise in translating image styles, and conditional outpainting methods have gained attention for enhancing peripheral details. Our work builds on these methods by focusing on vessel-based conditioning.

## Methodology
Our proposed method consists of a two-step process to generate high-quality fundus images by conditioning on vessel masks. By first generating a colorized mask from a grayscale mask and subsequently outpainting the rest of the fundus image, we ensure that the vessel structures are accurately represented in the final output. Below are the details of each step.

### Step 1: Generating Color Mask from Grayscale Mask
In the first step, we transform a grayscale mask of the vessels into a color mask. To train this model, we prepare a dataset containing pairs of grayscale and corresponding color masks. The color mask is derived from the full fundus image by isolating only the pixels that correspond to the white regions in the grayscale mask. This allows us to create a color mask that accurately reflects the vessel structures.

We then employ the **Pix2Pix** model for this task, training it with grayscale mask and color mask pairs. The Pix2Pix model, a conditional GAN architecture, learns to translate the grayscale input into a colorized mask that matches the vessels in the fundus image. This process ensures that the vessels are well-defined in the color space, setting a strong foundation for the next outpainting step.

### Step 2: Outpainting the Fundus Image
In the second step, we use the **LaMa** model to complete the rest of the fundus image based on the color mask generated in Step 1. Our training setup involves feeding both the color mask from Step 1 and the complete fundus image as inputs, with the task being to fill in the non-vessel regions while preserving the structure and context of the vessels.

The original LaMa framework is designed for generic inpainting tasks, where an image \( x \) and a random mask \( x' \) are combined to create a partially occluded image. For our purposes, however, we modify this dataset composition to better suit our architecture. Specifically, we condition the LaMa model on structured vessel regions represented by the color mask, rather than random masks, which ensures that vessel structures remain intact while the surrounding fundus regions are outpainted.

## Datasets
The dataset for this project will include retinal fundus images and corresponding vessel segmentation masks. These images are essential for training models that can accurately capture vessel structure and generate realistic retinal images around them. We will use existing public datasets, such as DRIVE or STARE, which contain both vessel masks and full fundus images to facilitate the two-step generation process.

## License
This project is licensed under [MIT License](LICENSE).

## Acknowledgments
We thank the authors of Pix2Pix and LaMa for their contributions to generative models and inpainting methodologies.
