# DCIN-CQG

## Code Coming Soon!

## Color-Quality Invariance for Robust Medical Image Segmentation
link to paper coming soon

**Abstract**

Single-source domain generalization (SDG) in medical image segmentation remains a significant challenge, particularly for images with varying color distributions and qualities. 
Previous approaches often struggle when models trained on high-quality images fail to generalize to low-quality test images due to these color and quality shifts. 
In this work, we propose two novel techniques to enhance generalization: Dynamic Color Image Normalization (DCIN) and Color-Quality Generalization (CQG) loss. 
The DCIN dynamically normalizes the color of test images using a dual-strategy reference selection pipeline. 
Specifically, the DCIN utilizes a Global Reference Image Selection (GRIS), which finds a universal reference image, and Local Reference Image Selection (LRIS), which selects a semantically similar reference image per test sample. 
Additionally, CQG loss enforces invariance to color and quality variations by ensuring consistent segmentation predictions across transformed image pairs. 
Experimental results show that our approach significantly improves segmentation performance over the baseline, producing strong, usable results even under substantial domain shifts. 
Notably, our full DCIN pipeline (GRIS + LRIS) achieves superior generalization compared to existing methods, including those using human-expert-selected references. 
Our work contributes to the development of more robust medical image segmentation models that generalize across unseen domains.

## Installation

Clone this repo.
```bash
git clone https://github.com/Kaiseem/SLAug.git
cd SLAug/
```

This code requires PyTorch and PyTorch lightning. Please install dependencies with
```bash
pip install -r requirements.txt
```

## Organization

The `DCIN_CQG` directory is structured to facilitate efficient organization of training, evaluation, visualization, and data processing tasks. The `main` folder contains scripts and subdirectories related to model training, including preprocessing utilities, data augmentation, custom loss functions, and evaluation scripts. The `visualizations` folder provides tools for analyzing and interpreting results, with a dedicated script for visualizing outputs and a collection of sample images. The `GRIS` and `LRIS` directories contain important scripts for pre-computing elements of the DCIN. 

```none
DCIN_CQG
├── main
│   ├── preprocess
│   │   ├── datasets/dataloaders 
│   │   ├── augmentations
│   │   ├── DCIN helpers
│   ├── CQG Loss
│   ├── training scripts
│   ├── evaluation scripts
├── visualizations
│   ├── visualize results scripts
│   ├── paper figures
├── GRIS
│   ├── color histogram computation
│   ├── GRIS selector
├── LRIS
│   ├── embedding extractor
│   ├── LRIS selector
```

## Training (source domain)

To run training, simply update the dataset class of the source domain in the main/preprocess directory to read in your images. For standard models, update `dataset_hq.py`. For color-quality generalization training, update `dataset_hq_cqg.py`

Then run the following commands: Coming soon

## Inference (non-source domain)

To run inference non-source domains, simply update the dataset class of a non-source domain (`dataset_lq.py` or `dataset_sp.py`) in the main/preprocess directory to read in your images. 

Then run the following commands: Coming soon

## Data and Experimental Results

The data used for this project is private, proprietary, and the exclusive property of Aillis Inc.

Information about the Data and Experimental Results can be found in our paper here: coming soon. 

## Acknowledgements

All authors are with the AI Development Department, Aillis, Inc., Tokyo, Japan. 

This project was made possible by the data provided by Aillis Inc. To learn more visit: https://aillis.jp/

We would like to thank all researchers at Aillis Inc.,
especially the AI Development team, Dr. Memori Fukuda
(MD), Dr. Kei Katsuno (MD), and Dr. Takaya Hanawa (MD)
for their valuable comments and feedback.


## Citation
Coming Soon
