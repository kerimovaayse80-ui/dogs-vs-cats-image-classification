# Dogs vs Cats Image Classification

### A Deep Learning Approach Using CNN and Transfer Learning

This project implements a binary image classification system that distinguishes between cat and dog images using deep learning. It explores the complete machine learning workflow, from dataset preparation and exploratory data analysis to model training, performance evaluation, and comparison.

Two approaches are implemented and evaluated: a custom Convolutional Neural Network (CNN) trained from scratch and a MobileNetV2 model using transfer learning.

The project was developed as part of the Artificial Intelligence course at **Azerbaijan State Oil and Industry University (ASOIU)**.

---

## Project Overview

Image classification is a fundamental task in computer vision that enables machines to recognize and categorize visual information.

The goal of this project is to classify an input image into one of two categories:

- **Cat (0):** The image contains a cat.
- **Dog (1):** The image contains a dog.

The project investigates whether a CNN trained from scratch can achieve competitive performance compared with a model that uses pre-trained visual features.

### Objectives

- Explore and clean an image dataset containing cats and dogs.
- Perform exploratory data analysis (EDA) to investigate class distribution and image dimensions.
- Apply image preprocessing and data augmentation.
- Build and train a custom CNN architecture.
- Implement transfer learning using MobileNetV2 pre-trained on ImageNet.
- Compare the models using accuracy, loss, confusion matrices, classification metrics, and ROC-AUC.
- Analyze model performance, limitations, and potential improvements.

---

## Technologies and Libraries

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow and Keras | Deep learning models and training |
| NumPy | Numerical operations and image arrays |
| Pandas | Dataset organization and data manipulation |
| Matplotlib | Data visualization and learning curves |
| Seaborn | Confusion matrix visualization |
| Scikit-learn | Train-validation splitting and model evaluation |
| Pillow (PIL) | Image loading, inspection, and validation |
| Jupyter Notebook | Interactive development and experimentation |

---

## Dataset

The project uses the **Microsoft Cats and Dogs Dataset**, distributed through the Kaggle Dogs vs. Cats competition.

**Dataset source:** [Kaggle — Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data)

### Dataset Characteristics

- Original dataset: approximately 25,000 JPEG images.
- Classes: cats and dogs.
- Image format: RGB JPEG.
- Original image dimensions: variable.
- Cleaned dataset: 24,998 images according to the project report.
- Class distribution after cleaning: 12,499 cat images and 12,499 dog images.

The balanced class distribution provides a suitable starting point for binary image classification without a major class imbalance.

---

## Project Workflow

### 1. Data Collection and Preparation

The dataset is downloaded from Microsoft's dataset archive and extracted into separate `Cat` and `Dog` directories.

Image paths and their corresponding labels are collected into a Pandas DataFrame.

The labels are encoded as follows:

- `0` — Cat
- `1` — Dog

This structure makes it easier to manage image paths, labels, and dataset splits throughout the workflow.

### 2. Data Cleaning

Before model training, the dataset is checked for non-image files and corrupted images.

The cleaning process includes:

- Identifying files that are not valid images.
- Removing Windows thumbnail files (`Thumbs.db`).
- Removing corrupted image files that cannot be opened correctly.
- Organizing image paths and labels into a consistent DataFrame.

This step helps prevent errors during image loading and model training.

### 3. Exploratory Data Analysis (EDA)

EDA is performed to understand the structure and quality of the dataset before building the models.

The analysis includes:

- **Class distribution:** Comparing the number and percentage of cat and dog images.
- **Image dimensions:** Examining the distribution of image widths and heights.
- **Sample visualization:** Displaying grids of cat and dog images.
- **Image inspection:** Reviewing visual variety, image quality, poses, and backgrounds.
- **Data augmentation examples:** Comparing original images with transformed versions.

The findings highlight the need to resize images to consistent dimensions before feeding them into neural networks.

### 4. Image Preprocessing and Data Augmentation

Images are resized and normalized to prepare them for model training.

Pixel values are rescaled from the original range to the interval `[0, 1]` by dividing by 255.

Data augmentation is applied to the training images to expose the model to different visual variations and help reduce overfitting.

The transformations include:

- Random rotation.
- Shearing.
- Zooming.
- Horizontal flipping.
- Nearest-neighbor filling for newly exposed image regions.

The custom CNN uses images of size `128 × 128 × 3`, while MobileNetV2 uses `224 × 224 × 3` inputs.

The dataset is split into training and validation subsets using an 80/20 split with `random_state=42`. Both models use the same underlying split for comparison.

### 5. Model 1 — Custom Convolutional Neural Network (CNN)

The first model is a sequential CNN developed and trained from scratch.

Its architecture consists of:

1. Convolutional layer with 16 filters and ReLU activation.
2. Max-pooling layer.
3. Convolutional layer with 32 filters and ReLU activation.
4. Max-pooling layer.
5. Convolutional layer with 64 filters and ReLU activation.
6. Max-pooling layer.
7. Flatten layer.
8. Fully connected Dense layer with 512 neurons and ReLU activation.
9. Output Dense layer with one neuron and sigmoid activation.

**Training configuration:**

- Input size: `128 × 128 × 3`
- Optimizer: Adam
- Loss function: Binary cross-entropy
- Batch size: 512
- Training duration: 10 epochs
- Total parameters: 6,447,137

The model learns visual patterns directly from the training images without relying on a pre-trained feature extractor.

### 6. Model 2 — MobileNetV2 Transfer Learning

The second model uses MobileNetV2, a lightweight CNN architecture pre-trained on ImageNet.

Instead of learning all visual features from scratch, the model reuses the pre-trained convolutional base as a feature extractor.

A custom classification head is added to adapt the network to the cat-versus-dog task.

**Architecture:**

1. Pre-trained MobileNetV2 backbone.
2. Global Average Pooling layer.
3. Dense layer with 128 neurons and ReLU activation.
4. Output Dense layer with one neuron and sigmoid activation.

The pre-trained base is frozen during training, meaning its weights are not updated. Training is performed on the newly added classification head.

**Training configuration:**

- Input size: `224 × 224 × 3`
- Pre-trained weights: ImageNet
- Optimizer: Adam
- Loss function: Binary cross-entropy
- Batch size: 32
- Training duration: 5 epochs
- Base model: Frozen

This approach allows the classifier to reuse previously learned visual representations rather than learning every feature from the beginning.

---

## Model Performance Comparison

The following results are taken from the training-history comparison documented in the project report.

| Metric | Custom CNN | MobileNetV2 |
|---|---:|---:|
| Validation accuracy | 78.16% | 98.36% |
| Validation loss | 0.4616 | 0.0429 |
| Training epochs | 10 | 5 |
| Input resolution | 128 × 128 | 224 × 224 |
| Batch size | 512 | 32 |

### Key Findings

- **Higher validation accuracy:** MobileNetV2 achieved a reported validation accuracy 20.20 percentage points higher than the custom CNN.
- **Lower validation loss:** MobileNetV2 achieved a substantially lower reported validation loss.
- **Fewer training epochs:** The transfer learning model was trained for five epochs, compared with ten epochs for the custom CNN.
- **Benefits of pre-trained features:** The results demonstrate the potential of transfer learning to improve classification performance on this dataset.

The reported metrics should be interpreted alongside the evaluation limitations described below.

---

## Model Evaluation and Visualization

The notebook includes several techniques for analyzing the models beyond accuracy alone.

### Learning Curves

Training and validation accuracy and loss are plotted across epochs to examine model convergence and training behavior.

### Confusion Matrices

Confusion matrices are generated to inspect correct predictions and misclassifications for the cat and dog classes.

### Classification Report

The notebook uses Scikit-learn to calculate classification metrics, including:

- Precision
- Recall
- F1-score
- Class support

These metrics provide a more detailed view of performance for each category.

### ROC Curve and AUC

Receiver Operating Characteristic (ROC) curves are plotted for both models using predicted probabilities.

The Area Under the Curve (AUC) is calculated to compare their ability to distinguish between the two classes across different classification thresholds.

### Single-Image Prediction

The custom CNN is also used to make a prediction for an individual image from the dataset, demonstrating the basic inference workflow.

---

## Limitations and Evaluation Considerations

Several limitations should be considered when interpreting the reported results.

- **No independent test set:** The project uses a training-validation split, so the reported performance does not represent evaluation on a separate held-out test set.
- **Custom CNN evaluation discrepancy:** The custom CNN confusion-matrix implementation predicts on a validation iterator that uses the default shuffled order. This can misalign predictions and labels, making the resulting classification metrics unreliable.
- **Metric consistency:** The report identifies a discrepancy between the custom CNN's reported validation accuracy of 78.16% and the accuracy derived from its confusion-matrix snapshot. These metrics should not be treated as directly comparable until the evaluation pipeline is corrected.
- **Frozen MobileNetV2 backbone:** The pre-trained convolutional layers are not fine-tuned for the specific dataset.
- **Limited hyperparameter search:** The models are trained with selected configurations rather than a systematic hyperparameter optimization process.
- **Binary classification only:** The current system is designed to classify cats and dogs, not multiple animal species or breeds.

For reliable evaluation, the validation generators should use `shuffle=False` when predictions are compared with ground-truth labels. The models should then be re-evaluated using aligned predictions, confusion matrices, classification reports, and ROC-AUC calculations.

---

## Future Improvements

Potential next steps include:

- Correcting the validation pipeline and recalculating all evaluation metrics.
- Introducing a separate held-out test set.
- Fine-tuning selected MobileNetV2 layers with a low learning rate.
- Experimenting with alternative architectures such as EfficientNetB0 and ResNet50V2.
- Applying learning-rate scheduling and systematic hyperparameter optimization.
- Using Grad-CAM to visualize the image regions influencing predictions.
- Extending the task to multi-class animal or breed classification.
- Exploring model quantization and pruning for resource-efficient deployment.

---

## Running the Project

### Prerequisites

Install Python and Jupyter Notebook, then install the libraries used in the notebook.

```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn pillow tqdm requests jupyter
```

### Steps

1. Clone the repository.

```bash
git clone https://github.com/kerimovaayse80-ui/dogs-vs-cats-image-classification.git
```

2. Navigate to the project directory.

```bash
cd dogs-vs-cats-image-classification
```

3. Launch Jupyter Notebook.

```bash
jupyter notebook
```

4. Open `Dogs_And_Cats_Identification.ipynb` and execute the cells in order.

5. Ensure the dataset has been downloaded and extracted successfully before running the data preparation and training steps.

**Note:** Training time depends on your hardware and available computational resources. The notebook downloads the dataset from Microsoft's archive, so an
