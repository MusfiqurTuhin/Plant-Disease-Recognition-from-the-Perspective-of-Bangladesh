# 📓 Experiment Notebooks

This directory contains the Jupyter notebooks used for the experimental analysis, training, and evaluation of various deep learning models for plant disease recognition.

## 📋 Notebooks Overview

| # | Notebook | Model Architecture | Description |
| :--- | :--- | :--- | :--- |
| **01** | [ConvNeXtBase](01_ConvNeXtBase_Transfer_Learning.ipynb) | ConvNeXtBase | Transfer learning with fine-tuning on ConvNeXtBase. |
| **02** | [ResNet101V2](02_ResNet101V2_Transfer_Learning.ipynb) | ResNet101V2 | Deep residual network with 101 layers. |
| **03** | [ResNet152V2](03_ResNet152V2_Transfer_Learning.ipynb) | ResNet152V2 | Deeper residual network with 152 layers. |
| **04** | [VGG16](04_VGG16_Transfer_Learning.ipynb) | VGG16 | Classic CNN architecture with 16 layers. |
| **05** | [VGG19](05_VGG19_Transfer_Learning.ipynb) | VGG19 | Deeper variant of VGG with 19 layers. |
| **06** | [Xception](06_Xception_Transfer_Learning.ipynb) | Xception | Depthwise separable convolutions. **(Best Individual Model)** |
| **07** | [Ensemble Model](07_Ensemble_Model.ipynb) | **Ensemble** | Weighted average of Xception, VGG19, and ResNet152V2. **(Best Overall)** |

## ⚙️ Prerequisites

To run these notebooks, you will need:

-   **Python 3.8+**
-   **Jupyter Notebook** or **JupyterLab**
-   **TensorFlow 2.x**
-   **GPU Support** (Highly recommended for faster training)

## 🏃 Usage

1.  Ensure the dataset is placed in the `../Dataset/` directory relative to this folder.
2.  Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
3.  Open and run the desired notebook. The notebooks are self-contained and include data loading, preprocessing, model definition, training, and evaluation steps.
