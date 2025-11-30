# Plant Disease Recognition from the Perspective of Bangladesh

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

This repository contains the official implementation and resources for the research paper **"Plant Disease Recognition from the Perspective of Bangladesh: A Comparative Study of Deep Learning Models and Ensemble Techniques"**, presented at the **2025 International Conference on Electrical, Computer and Communication Engineering (ECCE)**.

## 📋 Table of Contents
- [Abstract](#-abstract)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Results](#-results)
- [Citation](#-citation)
- [Keywords](#-keywords)

## 📄 Abstract
Agriculture is the backbone of Bangladesh's economy, yet it faces significant challenges from crop diseases that threaten food security. This study proposes a robust deep learning framework to recognize diseases in five major crops: **Corn (Maize), Potato, Rice, Tomato, and Wheat**.

We curated the **Bangladeshi Crops Disease Dataset (BCDD)**, a comprehensive collection of **8,992 images** across **19 classes**, covering both healthy and diseased states. Our research evaluates six state-of-the-art transfer learning models and introduces a high-performance **Ensemble Model** that achieves superior accuracy.

## 💾 Dataset
The **Bangladeshi Crops Disease Dataset (BCDD)** is a curated collection of images sourced from multiple open-access repositories (Kaggle Wheat Leaf, Rice Leaf, and Plant Village).

- **Total Images:** 8,992
- **Classes:** 19 (Healthy & Diseased)
- **Crops:** Corn, Potato, Rice, Tomato, Wheat
- **Preprocessing:** Resized to 96x96, Normalized to `[-1, 1]`
- **Split:** 70% Train, 15% Validation, 10% Test

### Download Links
| Platform | Link |
| :--- | :--- |
| **Kaggle** | [![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue?style=flat&logo=kaggle)](https://www.kaggle.com/datasets/musfiqurtuhin/bangladeshi-crops-disease-dataset-bcdd) |
| **Hugging Face** | [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Dataset-yellow?style=flat&logo=huggingface)](https://huggingface.co/datasets/musfiqurtuhin/BCDD) |

### Class Breakdown
<details>
<summary>Click to view all 19 classes</summary>

1.  **Corn (Maize)**
    -   `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot`
    -   `Corn_(maize)___Common_rust_`
    -   `Corn_(maize)___Northern_Leaf_Blight`
    -   `Corn_(maize)___healthy`
2.  **Potato**
    -   `Potato___Early_blight`
    -   `Potato___Late_blight`
    -   `Potato___healthy`
3.  **Rice**
    -   `Rice_bacterial_leaf_blight`
    -   `Rice_brown_spot`
    -   `Rice_leaf_blast`
    -   `Rice_healthy`
4.  **Tomato**
    -   `Tomato___Bacterial_spot`
    -   `Tomato___Early_blight`
    -   `Tomato___Late_blight`
    -   `Tomato___healthy`
5.  **Wheat**
    -   `Wheat Brown rust`
    -   `Wheat Yellow rust`
    -   `Wheat Loose Smut`
    -   `Wheat Healthy`
</details>

## 🧠 Methodology
We employed Transfer Learning with fine-tuning on six pre-trained architectures. To further boost performance, we developed a weighted average **Ensemble Model**.

### Models Implemented
- **ConvNeXtBase**
- **ResNet101V2**
- **ResNet152V2**
- **VGG16**
- **VGG19**
- **Xception**
- **Ensemble:** (Xception + VGG19 + ResNet152V2)

## 🛠️ Project Structure
```bash
.
├── notebooks/                  # Jupyter notebooks for training & evaluation
│   ├── 01_ConvNeXtBase_Transfer_Learning.ipynb
│   ├── ...
│   └── 07_Ensemble_Model.ipynb
├── Dataset/                    # Dataset directory (ignored in git)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- GPU recommended for training

### Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/MusfiqurTuhin/Plant-Disease-Recognition-from-the-Perspective-of-Bangladesh.git
    cd Plant-Disease-Recognition-from-the-Perspective-of-Bangladesh
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup Dataset:**
    Download the dataset from Kaggle and extract it into the `Dataset/` folder.

### Usage
Run the notebooks in the `notebooks/` directory to train or evaluate models:
```bash
jupyter notebook notebooks/01_ConvNeXtBase_Transfer_Learning.ipynb
```

## 📊 Results
Our proposed **Ensemble Model** outperformed individual models, demonstrating the effectiveness of combining diverse architectures.

| Model | Accuracy |
| :--- | :--- |
| **Ensemble (Xception + VGG19 + ResNet152V2)** | **99.33%** 🏆 |
| Xception | 99.11% |
| VGG19 | 98.67% |
| ResNet152V2 | 98.44% |
| ConvNeXtBase | 98.22% |
| ResNet101V2 | 98.00% |
| VGG16 | 97.78% |

## 📜 Citation
If you use this code or dataset in your research, please cite our paper:

```bibtex
@INPROCEEDINGS{11013222,
  author={Rahman, Md. Musfiqur and Tusher, Md Mahbubur Rahman and Rinky, Susmita Roy and Mokit, Junaid Rahman and Biswas, Sudipa},
  booktitle={2025 International Conference on Electrical, Computer and Communication Engineering (ECCE)}, 
  title={Plant Disease Recognition from the Perspective of Bangladesh: A Comparative Study of Deep Learning Models and Ensemble Techniques}, 
  year={2025},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/ECCE64574.2025.11013222}
}
```

## 🔑 Keywords
`Deep learning` `Measurement` `Plant diseases` `Accuracy` `Transfer learning` `Crops` `Predictive models` `Robustness` `Agriculture` `Ensemble learning` `Plant Disease Detection` `Xception` `VGG16` `VGG19` `ResNet152V2` `ConvNeXtBase` `Crop Management`

---
<div align="center">
  <p>Maintained by <a href="https://github.com/MusfiqurTuhin">Musfiqur Tuhin</a></p>
</div>
