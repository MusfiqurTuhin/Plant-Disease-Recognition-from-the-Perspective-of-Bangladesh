# Plant Disease Recognition from the Perspective of Bangladesh

This repository contains the code and resources for the research paper **"Plant Disease Recognition from the Perspective of Bangladesh: A Comparative Study of Deep Learning Models and Ensemble Techniques"** (IEEE EICT 2024).

## 📄 Abstract
Agriculture is a vital sector in Bangladesh, yet crop diseases significantly impact yield and food security. This study presents a comprehensive approach to recognizing diseases in five major crops: **Corn (Maize), Potato, Rice, Tomato, and Wheat**. We utilized a custom dataset, the **Bangladeshi Crops Disease Dataset (BCDD)**, comprising 19 classes (healthy and diseased) with 8,992 images.

We implemented and compared six state-of-the-art deep learning models:
- **ConvNeXtBase**
- **ResNet101V2**
- **ResNet152V2**
- **VGG16**
- **VGG19**
- **Xception**

Additionally, we developed an **Ensemble Model** combining Xception, VGG19, and ResNet152V2, which achieved the highest accuracy of **99.33%**.

## 📂 Dataset
The dataset used in this research is the **Bangladeshi Crops Disease Dataset (BCDD)**. It aggregates images from multiple sources (Wheat Leaf Disease, Rice Leaf Disease, and Plant Village) and includes 19 classes.

- **Kaggle:** [Bangladeshi Crops Disease Dataset (BCDD)](https://www.kaggle.com/datasets/musfiqurtuhin/bangladeshi-crops-disease-dataset-bcdd)
- **Hugging Face:** [musfiqurtuhin/BCDD](https://huggingface.co/datasets/musfiqurtuhin/BCDD)

### Classes
1. **Corn:** Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy
2. **Potato:** Early Blight, Late Blight, Healthy
3. **Rice:** Bacterial Leaf Blight, Brown Spot, Leaf Blast, Healthy
4. **Tomato:** Bacterial Spot, Early Blight, Late Blight, Healthy
5. **Wheat:** Brown Rust, Yellow Rust, Loose Smut, Healthy

## 🛠️ Project Structure
```
.
├── notebooks/                  # Jupyter notebooks for each model
│   ├── 01_ConvNeXtBase_Transfer_Learning.ipynb
│   ├── 02_ResNet101V2_Transfer_Learning.ipynb
│   ├── 03_ResNet152V2_Transfer_Learning.ipynb
│   ├── 04_VGG16_Transfer_Learning.ipynb
│   ├── 05_VGG19_Transfer_Learning.ipynb
│   ├── 06_Xception_Transfer_Learning.ipynb
│   └── 07_Ensemble_Model.ipynb
├── Dataset/                    # Dataset directory (ignored in git)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- TensorFlow / Keras

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/MusfiqurTuhin/Plant-Disease-Recognition-from-the-Perspective-of-Bangladesh.git
   cd Plant-Disease-Recognition-from-the-Perspective-of-Bangladesh
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/musfiqurtuhin/bangladeshi-crops-disease-dataset-bcdd) and extract it into the `Dataset/` folder.

### Running the Notebooks
Navigate to the `notebooks/` directory and run the desired notebook using Jupyter:
```bash
jupyter notebook notebooks/01_ConvNeXtBase_Transfer_Learning.ipynb
```

## 📊 Results
| Model | Accuracy |
| :--- | :--- |
| **Ensemble (Xception + VGG19 + ResNet152V2)** | **99.33%** |
| Xception | 99.11% |
| VGG19 | 98.67% |
| ResNet152V2 | 98.44% |
| ConvNeXtBase | 98.22% |
| ResNet101V2 | 98.00% |
| VGG16 | 97.78% |

## 📜 Citation
If you find this work useful, please cite our paper:

```bibtex
@INPROCEEDINGS{11013222,
  author={Tuhin, Musfiqur Rahman and Islam, Md. Monirul and Islam, Md. Saiful and Islam, Md. Sherazul},
  booktitle={2024 6th International Conference on Electrical Information and Communication Technology (EICT)}, 
  title={Plant Disease Recognition from the Perspective of Bangladesh: A Comparative Study of Deep Learning Models and Ensemble Techniques}, 
  year={2024},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/EICT68925.2024.11013222}
}
```

## 🔗 Links
- **Paper:** [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/11013222)
- **GitHub:** [MusfiqurTuhin/Plant-Disease-Recognition-from-the-Perspective-of-Bangladesh](https://github.com/MusfiqurTuhin/Plant-Disease-Recognition-from-the-Perspective-of-Bangladesh)
