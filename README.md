# ✍️ Handwritten Text Generation

## 🎯 Objective
The goal of this project is to generate **realistic handwritten-style text** using **deep learning models**.  
We synthetically create handwriting stroke data, train a Bidirectional LSTM model, and generate new handwriting sequences.

---

## 🛠️ Tech Stack
- Python 3
- TensorFlow
- NumPy
- Matplotlib
- TQDM

---

## 🛆 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ParavaVarshitaReddy/Handwritten-Text-Generation.git
   cd handwritten-text-generation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run

Run the `main.py` script:

```bash
python main.py
```

This will:
- Create synthetic handwriting data.
- Normalize and batch the data.
- Train the model.
- Generate handwriting samples with different temperature settings.
- Save generated images and training plots in the `output/` directory.

---

## 📈 Results

- **Model Summary**:  
  - 2-layer Bidirectional LSTM
  - Total Parameters: **2,108,931** (~8.04 MB)

- **Training Performance**:
  - Final Training Loss: **0.0057**
  - Final Validation Loss: **0.0033**

- **Model Saved**:  
  - `handwriting_model.h5`

- **Outputs**:
  - Handwriting samples generated at temperatures **0.1**, **0.3**, **0.5**, **0.7**, and **1.0**.
  - Training loss curve plotted and saved.

---

## 🖋️ Sample Generated Handwriting

| Temperature 0.5 |
|:---:|
| ![Handwriting Sample](output/sample_temp_0.5.png) |

---

## 📊 Training History Plot

| Training Loss vs Validation Loss |
|:---:|
| ![Training History](output/training_history.png) |

---

## 📝 Folder Structure

```
handwritten-text-generation/
├── data/
│   └── processed/
├── output/
│   ├── sample_temp_0.1.png
│   ├── sample_temp_0.3.png
│   ├── sample_temp_0.5.png
│   ├── sample_temp_0.7.png
│   ├── sample_temp_1.0.png
│   └── training_history.png
├── src/
│   ├── data_utils.py
│   ├── model.py
│   ├── generate.py
│   └── train.py
├── main.py
├── requirements.txt
└── README.md
```

---

## ✨ Outcome
A generative deep learning model capable of producing **handwritten-style strokes** that mimic human writing patterns, showcasing skills in **sequence modeling**, **synthetic data generation**, and **deep learning creativity**.

---

# 🔥 Thank you for visiting this project! 🔥

##⚡ Important Notes
The data/processed/ and output/ folders are intentionally kept empty in the repository.

These folders will be automatically populated when you run main.py:

data/processed/ will contain the generated synthetic stroke datasets (train_strokes.pkl and val_strokes.pkl).

output/ will contain the generated handwriting samples and training history plot.

This approach keeps the repository clean and lightweight, as per best practices.
