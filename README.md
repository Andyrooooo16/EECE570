# EECE570 Final Project – Retinal Fundus Disease Classifier  
**Author:** Andrew Feng (ID: 12039012)  
**Course:** EECE 570 - Computer Vision  

## Project Overview
This project implements a multi-label disease classifier for retinal fundus images using a fine-tuned ResNet-50 model. It explores classification performance on a medical dataset through training, evaluation, and fine-tuning pipelines. The goal is to effectively detect key retinal diseases, addressing class imbalance and real-world diagnostic challenges.

## Dataset
The dataset used in this project is available here:  
🔗 [Google Drive Dataset](https://drive.google.com/drive/folders/1_S10EHnr1WJhG0syPcaV3XSWHx25IY8N?usp=sharing)

After cloning this repository, download the **entire `data/` folder** from the link above and place it inside the root directory of the project (`EECE570/`). This is required to run the code successfully and reproduce results.

---

## Getting Started

### 1. Clone the Repository

git clone https://github.com/yourusername/EECE570.git
cd EECE570

### 2. Set Up the Dataset
Download the data/ folder from the Google Drive link
Place the folder directly inside the project root:

css
Copy
Edit
EECE570/
├── data/
├── src/
├── outputs/
└── README.md

### 3. Code Structure
All source code is located in the src/ directory:

css
Copy
Edit
EECE570/
└── src/
    ├── crop_pictures.py
    ├── dataset.py
    ├── disease_breakdown.py
    ├── model.py
    ├── train_model.py
    ├── evaluate.py
    ├── finetuning_model.py
    ├── evaluate_finetuned.py
    └── test_finetuned.py

### 4. How to Run the Project

#### 1. Image Preprocessing
Run 'crop_pictures.py' on the Training, Validation, and Test datasets to generate cropped images for model training, evaluation, and testing.
#### 2. Data Preparation
Run 'dataset.py' to preprocess all three subsets of data.
#### 3. Dataset Analysis
Run 'disease_breakdown.py' to generate three visualization graphs in your browser showing the dataset composition and top disease classes in each subset.
#### 4. Model Setup
The 'model.py' file contains the pretrained ResNet50 model with adjusted hyperparameters for initial training.
#### 5. Model Training
Run 'train_model.py' to train the multi-label classifier using the cropped Training Dataset and the model from 'model.py'. This will save 'best_fundus_model.pth' in the EECE570/outputs/Training folder.
#### 6. Model Evaluation
Run 'evaluate.py' to assess the performance of 'best_fundus_model.pth'. Evaluation metrics will be saved in the EECE570/outputs/Evaluation folder.
#### 7. Model Fine-tuning
Run 'finetuning_model.py' to fine-tune the model. This will generate 'fine_tuned_model.pth'.
#### 8. Fine-tuned Model Evaluation
Run 'evaluate_finetuned.py' to evaluate the fine-tuned model. Results will be saved under EECE570/outputs/Fine_Tuned.
#### 9. Final Testing
Run 'test_finetuned.py' to perform the final evaluation of the fine-tuned model on the test dataset. Final evaluation metrics will be saved in the EECE570/outputs/Test folder.



