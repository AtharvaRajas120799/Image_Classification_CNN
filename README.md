# Image Classification using Convolutional Neural Networks

A complete CNN pipeline built from scratch in TensorFlow/Keras — covering data preprocessing, augmentation, model architecture design, training, and evaluation with quantitative metrics.

## What This Project Does

- Builds a custom CNN architecture (not a pre-trained model) for binary image classification
- Implements data augmentation (rotation, flip, zoom) to improve generalization on small datasets
- Trains with systematic hyperparameter selection: learning rate, batch size, epochs, dropout
- Evaluates model performance using confusion matrix, classification report (precision, recall, F1), and training/validation accuracy curves

## Technical Stack

Python, TensorFlow/Keras, NumPy, OpenCV, Matplotlib, Seaborn, Scikit-Learn

## Project Structure
cnn_project/
├── src/model_code.py      # Training pipeline
├── models/                # Saved model weights
└── README.md              # Detailed documentation

## How to Run

```bash
git clone https://github.com/AtharvaRajas120799/Image_Classification_CNN.git
cd Image_Classification_CNN/cnn_project/src
pip install tensorflow numpy matplotlib seaborn scikit-learn opencv-python
python model_code.py
```

## What I Learned

This was my first end-to-end ML project. I built it to understand the full pipeline — not just calling model.fit(), but understanding why specific architectures work, how augmentation prevents overfitting, and how to evaluate a model beyond just accuracy. The skills I developed here (data pipelines, model evaluation, systematic experimentation) directly informed my later work on 3D perception at TU Munich.

## Author

**Atharva Rajas** — M.Eng. Mechatronics & Robotics | [GitHub](https://github.com/AtharvaRajas120799)
