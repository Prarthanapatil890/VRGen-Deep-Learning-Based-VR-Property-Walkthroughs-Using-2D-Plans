# VRGen: Deep Learning Based VR Property Walkthroughs Using 2D Plans

VRGen is an end-to-end deep learning pipeline that automatically converts 2D architectural floor plan images into interactive 3D VR walkthroughs — no manual CAD/Blender modeling required. It uses a Conditional GAN (Pix2Pix-style) with an enhanced U-Net generator to predict heightmaps from floor plans, refines them with post-processing, and reconstructs a 3D mesh exported in VR-ready `.glb` format.

> Project Report: *VRGen: Deep Learning-Based VR Property Walkthroughs Using 2D Plans*
> A.P. Shah Institute of Technology, Thane — Dept. of CSE (AI & ML), University of Mumbai, 2025–2026

## Overview

Traditional 2D-to-3D architectural visualization relies on manual tools like CAD and Blender — a process that is time-consuming, expertise-dependent, and costly to iterate on. VRGen addresses this by using a **Conditional Generative Adversarial Network (cGAN)** to automatically learn the mapping between 2D floor plan images and their corresponding heightmap (depth) representations, which are then converted into fully navigable 3D VR environments.

This project contributes to:
- **SDG 9** — Industry, Innovation and Infrastructure
- **SDG 11** — Sustainable Cities and Communities
- **SDG 4** — Quality Education
- **SDG 12** — Responsible Consumption and Production

## Features

- 📐 Fully automated 2D floor plan → 3D mesh pipeline
- 🧠 Enhanced U-Net generator with attention mechanisms
- 🎯 PatchGAN discriminator for locally realistic outputs
- 🧹 Post-processing: denoising, wall enhancement, floor leveling, morphological cleanup
- 🕶️ VR-ready `.glb` export for immersive walkthroughs
- ⚙️ Multi-loss training strategy (L1 + Edge + Perceptual + Total Variation + GAN loss)
- 💻 Supports both GPU (CUDA) training and CPU inference

### Main Components

| Module | Responsibility |
|---|---|
| **Input Module** | Accepts floorplan images (PNG/JPG) |
| **Preprocessing Module** | Resizes to 256×256, normalizes, augments, enhances edges |
| **Dataset Loader** | Loads floorplans + heightmaps, synthesizes targets when missing |
| **Generator (Enhanced U-Net)** | Encoder–decoder with attention; outputs heightmap + edge map |
| **Discriminator (PatchGAN)** | Distinguishes real vs. generated patches |
| **Post-processing Module** | Noise removal, wall/door enhancement, morphological ops |
| **Heightmap Enhancement** | Thresholds walls vs. floors for clean 3D separation |
| **Mesh Generation** | Marching cubes algorithm converts heightmap → 3D mesh |
| **Output Module** | Exports final model as `.glb` for VR viewers |

## Pipeline

1. **Upload** a 2D floorplan image (PNG/JPG)
2. **Preprocess**: resize (256×256), normalize, augment, detect edges
3. **Generate heightmap** using the trained Enhanced U-Net Generator
4. **Post-process** the heightmap (denoise, sharpen, enhance walls/floors)
5. **Convert to 3D mesh** via the marching cubes algorithm
6. **Export** as `.glb` and view in a VR headset or browser-based viewer

## Tech Stack

**Language:** Python

**Core Libraries:**
- [PyTorch](https://pytorch.org/) — model design, training, optimization
- [Torchvision](https://pytorch.org/vision/) — transforms & dataset utilities
- [OpenCV](https://opencv.org/) — edge detection, morphological operations
- [NumPy](https://numpy.org/) / [SciPy](https://scipy.org/) — numerical & scientific computing
- [Pillow (PIL)](https://python-pillow.org/) — image I/O and preprocessing
- [Trimesh](https://trimesh.org/) — 3D mesh generation
- [PyRender](https://pyrender.readthedocs.io/) — 3D rendering/visualization
- [Plotly](https://plotly.com/python/) — interactive debugging visualizations
- [PyYAML](https://pyyaml.org/) — configuration management
- [TQDM](https://tqdm.github.io/) — training progress bars

## Dataset

- **Source:** [Text-to-Floorplan using GANs](https://www.kaggle.com/code/aarykeskar/text-to-floorplan-using-gans) (Kaggle)
- **Total images:** 4,003
- **Training set:** 1,000 images
- **Test set:** 500 images
- Remaining images support validation/augmentation

**Preprocessing steps:**
- Resize to 256×256
- Normalization
- Grayscale conversion, contrast enhancement, edge detection
- Synthetic heightmap target generation (since annotated heightmaps aren't available)

**Augmentation techniques:**
- Horizontal/vertical flipping
- Rotation (90°, 180°, 270°)
- Brightness/contrast adjustment

## Model Details

- **Architecture:** Conditional GAN (Pix2Pix-based)
- **Generator:** Enhanced U-Net with attention gates, dual output heads (heightmap + edge map)
- **Discriminator:** PatchGAN (local patch realism)

**Loss functions:**

| Loss Function | Purpose |
|--------------|---------|
| L1 (Reconstruction) | Ensures pixel-wise reconstruction accuracy between the generated heightmap and the target. |
| Edge Loss | Preserves architectural boundaries such as walls, doors, and edges. |
| Perceptual Loss | Maintains high-level structural similarity using deep feature representations. |
| Total Variation (TV) | Reduces noise and promotes smoothness in the generated heightmaps. |
| Adversarial (GAN) | Encourages the generator to produce realistic outputs through discriminator feedback. |


### Viewing the Output

Open the generated `.glb` file in any glTF-compatible viewer, a WebXR browser viewer, or a standalone VR headset app.


## Comparison with Existing Systems

| Aspect | Existing Systems | VRGen |
|---|---|---|
| Input & Output | Limited to 2D or partial 3D | Full 2D → 3D (.glb) |
| 3D Reconstruction | Partial / not fully implemented | Full end-to-end |
| Model Approach | CNN / GAN / rule-based (varies) | Conditional GAN (Pix2Pix-based) |
| Loss & Evaluation | Limited or unreported | Explicit multi-loss evaluation |
| Output Quality | Moderate, weak edges | High structural accuracy, sharp edges |
| Automation | Semi-automated | Fully automated |
| VR Integration | Not supported / limited | Fully VR-compatible |


## Future Work

- Expand dataset diversity (more architectural styles/layouts)
- Semantic segmentation for doors, windows, furniture, room types
- Advanced texturing and material mapping for realism
- Real-time inference optimization (compression, pruning)
- Deployment on mobile devices and standalone VR headsets
- Multi-modal input support (hand-drawn sketches, text, voice)
- Enhanced VR interaction (navigation controls, annotations, guided tours)

## Team

Developed by students of the Department of Computer Science & Engineering (AI & ML), A.P. Shah Institute of Technology, Thane, under the guidance of **Prof. Vijesh Mundokalam**.

| Name | Roll No. |
|---|---|
| Yash Penkar | 22106118 |
| Prarthana Patil | 22106035 |
| Rutuja Pawar | 22106043 |
| Dhruv Sawant | 22106015 |

## References

Key references (full list in the project report):

1. Isola et al., *Image-to-Image Translation with Conditional Adversarial Networks*, CVPR 2017
2. Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015
3. Johnson et al., *Perceptual Losses for Real-Time Style Transfer and Super-Resolution*, ECCV 2016
4. Park & Kim, *3DPlanNet: Generating 3D Models from 2D Floor Plan Images Using Ensemble Methods*, Electronics 2021
5. Kippers et al., *Automatic 3D Building Model Generation Using Deep Learning Methods Based on CityJSON and 2D Floor Plans*, ISPRS 2021
6. Rudin, Osher & Fatemi, *Nonlinear Total Variation Based Noise Removal Algorithms*, Physica D, 1992
7. Sobel & Feldman, *An Isotropic 3×3 Image Gradient Operator*, Stanford AI Project, 1968

