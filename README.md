# DCIN-CQG Pipeline

## Color-Quality Invariance for Robust Medical Image Segmentation

**Authors**: Ravi Shah, Atsushi Fukuda, Quan Huu Cap

**Paper**: https://arxiv.org/abs/2502.07200<br>

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
git clone https://github.com/RaviShah1/DCIN-CQG.git
cd CIN-CQG/
```

This code requires PyTorch and PyTorch lightning. To create an environment, you can use:
```bash
conda env create -f environment.yml
conda activate dcin_cqg
```

## Organization

The `main` folder contains scripts and subdirectories related to model training, including preprocessing utilities, data augmentation, custom loss functions, and evaluation scripts. The `visualizations` folder provides tools for analyzing and interpreting results. The `GRIS` and `LRIS` directories contain important scripts for pre-computing elements of the DCIN. 

```none
DCIN_CQG
├── GRIS
│   ├── color_histogram_computation.py
│   ├── gris_selector.py
├── LRIS
│   ├── create_feature_vector_db.py
├── main
│   ├── preprocess
│   │   ├── augmentations.py 
│   │   ├── color_transfer.py
│   │   ├── constants.py
│   │   ├── dataset_cqg.py
│   │   ├── dataset_test.py
│   │   ├── dataset_train.py
│   │   ├── dataset_val.py
│   │   ├── lris_utils.py
│   ├── loss.py
│   ├── test.py
│   ├── train_cqg.py
│   ├── train_pl.py
│   ├── train.py
├── visualizations
│   ├── visualize_results.py
```

## Running the Code

**Step 1: Global Reference Image Selection**

Navigate to GRIS utils and update the paths in `gris_selector.py` with the paths to your dataset. Then run the selector and copy the best filepath that is output.

```
cd GRIS
python gris_selector.py
cd ..
```

**Step 2: Local Reference Image Selection**

Navigate to the LRIS utils and update the paths in `create_feature_vector_db.py` with the paths to your dataset. Then run the file to create the database.

```
cd LRIS
python create_feature_vector_db.py --model_name swinv2_large_window12to24_192to384.ms_in22k_ft_in1k
cd ..
```

**Step 3: Training**

Navigate to the main directory. In `main/preprocess/constants.py` update the paths with paths to your dataset. You may need to change the `dataset_train.py` file if your data is in a different format. Then run the `train.py` file. You can specify the backbone, batch size, learning rate, number of epochs, augmentations, and name via the flags.

```
cd main
python train.py --augs CQG
```

**Step 4: Evaluating**

Make further updates to the `constants.py` file if necessary. Make further updates to `dataset_test.py` if necessary. Then go to the `test.py` file and place the paths to the checkpoints of the models you would like to evaluate in the `weight_list`. Then run the tester.

```
python test.py
cd ..
```

**Step 5: Visualizing Results**

Navigate to the visualizations directory. Select your image and options in the main function at the bottom. Then run the file.

```
cd visualizations
python visualize_results.py
```


## Data and Experimental Results

The data used for this project is the private property of Aillis Inc.

Information about the Data and Experimental Results can be found in our paper here: https://arxiv.org/abs/2502.07200

## Acknowledgements

All authors are with the AI Development Department, Aillis, Inc., Tokyo, Japan. 

This project was made possible by the data provided by Aillis Inc. To learn more visit: https://aillis.jp/

We would like to thank all researchers at Aillis Inc.,
especially the AI Development team, Dr. Memori Fukuda
(MD), Dr. Kei Katsuno (MD), and Dr. Takaya Hanawa (MD)
for their valuable comments and feedback.


## Citation

```
@article{shah2025color,
  title={Color-Quality Invariance for Robust Medical Image Segmentation},
  author={Shah, Ravi and Fukuda, Atsushi and Cap, Quan Huu},
  journal={arXiv preprint arXiv:2502.07200},
  year={2025}
}
```