# Road Sign Detection Project
![image](https://github.com/user-attachments/assets/cd6f3387-44d6-435c-a355-694b1e451203)

This project focuses on developing a machine learning model capable of detecting and classifying various road signs from images. Accurate road sign recognition is crucial for applications such as autonomous driving and advanced driver-assistance systems (ADAS). By leveraging a labeled dataset of road sign images, we aim to train a model that can identify and categorize different types of road signs in real-world scenarios.

---

## Index

1. [Introduction](#introduction)
2. [Dataset Selection](#dataset-selection)
3. [Challenges and Considerations](#challenges-and-considerations)
4. [Adjustments Based on Feedback](#adjustments-based-on-feedback)
5. [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
6. [Methodology](#methodology)
7. [Models Implemented](#models-implemented)
8. [Results](#results)
9. [Future Enhancements](#future-enhancements)
10. [How to Run the Project](#how-to-run-the-project)
11. [Contributing](#contributing)
12. [Acknowledgments](#acknowledgments)

---

## Introduction

The objective of this project is to build a robust machine learning model that can accurately detect and classify road signs from images. This capability is essential for enhancing the safety and efficiency of autonomous vehicles and driver-assistance systems. By training the model on a comprehensive dataset of road sign images with corresponding annotations, we aim to achieve high accuracy in real-time road sign recognition.

## Dataset Selection

We utilized the [Road Sign Detection](https://www.kaggle.com/datasets/andrewmvd/road-sign-detection) dataset from Kaggle, which comprises 877 images of road signs categorized into four distinct classes. Each image is accompanied by bounding box annotations in PASCAL VOC format, providing precise localization of the road signs within the images.

## Challenges and Considerations

Throughout the project, we encountered several challenges:

- **Data Imbalance:** Some classes of road signs were underrepresented in the dataset, potentially leading to biased model predictions.
- **Variations in Lighting and Weather Conditions:** The images exhibited a range of lighting and weather conditions, affecting the visibility and appearance of the road signs.
- **Occlusions and Background Clutter:** Some road signs were partially obscured or blended into complex backgrounds, complicating the detection process.

## Adjustments Based on Feedback

Based on peer reviews and expert feedback, we implemented the following adjustments:

- **Data Augmentation:** Applied techniques such as rotation, scaling, and brightness adjustments to increase the diversity of the training data and mitigate class imbalance.
- **Hyperparameter Tuning:** Conducted extensive tuning of model hyperparameters to enhance performance and prevent overfitting.
- **Advanced Preprocessing:** Implemented image normalization and contrast enhancement to improve the model's robustness to varying lighting conditions.

## Data Cleaning and Preprocessing

To prepare the dataset for model training, we performed the following steps:

1. **Annotation Parsing:** Extracted class labels and bounding box coordinates from the XML annotation files using Python’s `xml.etree.ElementTree` module.
2. **Image Loading and Conversion:** Loaded images using OpenCV, converted them from BGR to RGB format, and resized them to a consistent dimension suitable for the model.
3. **Normalization:** Scaled pixel values to the [0, 1] range to facilitate faster convergence during training.
4. **Data Augmentation:** Applied random transformations to augment the dataset and improve the model’s generalization capabilities.

## Methodology

Our approach involved the following stages:

1. **Exploratory Data Analysis (EDA):** Conducted EDA to understand the distribution of classes, identify potential issues, and gain insights into the dataset.
2. **Model Selection:** Chose a Convolutional Neural Network (CNN) architecture suitable for object detection tasks.
3. **Training and Validation:** Split the dataset into training and validation sets, trained the model using the training set, and evaluated its performance on the validation set.
4. **Evaluation Metrics:** Used metrics such as precision, recall, F1-score, and mean Average Precision (mAP) to assess the model's performance.

## Models Implemented

We implemented the following models:

- **Baseline CNN:** A simple CNN architecture to establish a performance baseline.

## Results

Our final model achieved the following performance metrics:

- **Precision:** 0.83
- **Recall:** 0.83
- **F1-Score:** 0.82
- **mAP:** 0.83

These results indicate that the model performs well in detecting and classifying road signs, with a good balance between precision and recall.

## Future Enhancements

To further improve the project, we plan to:

- **Expand the Dataset:** Incorporate additional images to cover a wider variety of road signs and environmental conditions.
- **Implement Real-Time Detection:** Optimize the model for deployment in real-time systems with constraints on processing power and latency.
- **Explore Advanced Architectures:** Investigate the use of more recent object detection architectures, such as EfficientDet or YOLOv5, to enhance performance.

## How to Run the Project

To run the project locally:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/road-sign-detection.git
   cd AI_BOOTCAMP_Final_Project_Street_Sign_Recognition
   ```

2. **Install Dependencies:**  
   Ensure you have Python 3.10 or later. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Acuire the Data:**  
    ```bash
   jupyter execute Data_Mining.ipynb
   ```

4. **Preprocess the Data:**  
   Run the preprocessing script to prepare the data for training:

   ```bash
   jupyter execute Data_Preprocessing.ipynb
   ```

5. **Train the Model:**  
   Execute the training script:

   ```bash
      jupyter execute Best_Model.ipynb
   ```

6. **Run the Gradio App:**  
   After training, evaluate the model’s performance:

   ```bash
      jupyter execute gradio_app.ipynb
   ```

## Contributing

We welcome contributions to enhance the project. To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix:

   ```bash
   git checkout -b feature-name
   ```

3. Commit your changes:

   ```bash
   git commit -m "Description of feature"
   ```

4. Push to your branch:

   ```bash
   git push origin feature-name
   ```

5. Open a Pull Request detailing your changes.

### Contributors:
- Tom Haight
- Austin French
- Jaylen Guevara-Kirksey Bey
- Asif Mahmud

## Acknowledgments

We extend our gratitude to the contributors of the Road Sign Detection dataset on Kaggle (https://www.kaggle.com/datasets/andrewmvd/road-sign-detection) for providing the foundational data for this project. Additionally, we thank the open-source community for developing the tools and frameworks that made this project possible.
