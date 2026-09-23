# 🧬 Hepatitis C Classification using Artificial Neural Network (ANN)

> **IBM Data Analytics Internship Project**  
> Classifying Hepatitis C patient stages from clinical blood test data using a deep ANN model.

---

## 📌 Project Overview

Hepatitis C is a viral infection that causes liver inflammation and can progress through several stages — from a healthy blood donor all the way to Cirrhosis. Early and accurate classification of a patient's condition is critical for timely clinical intervention.

This project builds an end-to-end **multi-class classification pipeline** using a deep **Artificial Neural Network (ANN)** trained on real clinical lab data. The pipeline covers:

- Loading and cleaning the dataset (missing value imputation)
- Encoding categorical features
- Handling severe class imbalance using **SMOTE**
- Normalising features with **MinMaxScaler**
- Designing, training, and evaluating a **6-hidden-layer ANN** with Dropout regularisation
- Visualising results via Confusion Matrix, Predicted vs. True plots, and Training History

---

## 📂 Repository Structure

```
IBM-Data-Analytics-project/
│
├── VandanaKumari_HepatitisC_ANN.ipynb   ← Main Jupyter Notebook (full pipeline)
├── hcvdat0.csv                           ← Hepatitis C dataset (615 records)
├── requirements.txt                      ← Python dependencies
├── VandanaKumari_ProjectReport.docx      ← Full project report (Word)
└── README.md                             ← This file
```

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Name** | HCV Data — Hepatitis C Virus |
| **Source** | UCI Machine Learning Repository via Kaggle |
| **Link** | https://www.kaggle.com/datasets/fedesoriano/hepatitis-c-dataset |
| **File** | `hcvdat0.csv` |
| **Records** | 615 patients |
| **Features** | 12 clinical blood test markers + 1 target |
| **Task** | Multi-class classification (5 classes) |

### Feature Columns

| Feature | Description |
|---------|-------------|
| `Age` | Patient age in years |
| `Sex` | Patient sex (m / f) |
| `ALB` | Albumin |
| `ALP` | Alkaline phosphatase |
| `ALT` | Alanine transaminase |
| `AST` | Aspartate transaminase |
| `BIL` | Bilirubin |
| `CHE` | Cholinesterase |
| `CHOL` | Cholesterol |
| `CREA` | Creatinine |
| `GGT` | Gamma-glutamyl transferase |
| `PROT` | Total protein |

### Target Classes

| Class Label | Description |
|-------------|-------------|
| `0` | Blood Donor (healthy) |
| `1` | Suspect Blood Donor |
| `2` | Hepatitis |
| `3` | Fibrosis |
| `4` | Cirrhosis |

---

## 🏗️ Model Architecture

The ANN is a fully connected **Sequential** model built with **TensorFlow / Keras**:

```
Input (12 features)
        │
  Dense(512, ReLU) → Dropout(0.5)
        │
  Dense(256, ReLU) → Dropout(0.5)
        │
  Dense(128, ReLU) → Dropout(0.5)
        │
  Dense(64, ReLU)  → Dropout(0.5)
        │
  Dense(32, ReLU)  → Dropout(0.5)
        │
  Dense(16, ReLU)  → Dropout(0.5)
        │
  Dense(5, Softmax)   ← Output: 5 class probabilities
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss | Categorical Cross-Entropy |
| Batch Size | 64 |
| Max Epochs | 200 |
| Validation Split | 20% |
| Early Stopping | patience=100, monitor=`val_loss`, `restore_best_weights=True` |

---

## 🛠️ Technologies Used

| Category | Library / Tool |
|----------|----------------|
| Language | Python 3.9+ |
| Notebook | Jupyter Notebook |
| Data Processing | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn` |
| ML / Preprocessing | `scikit-learn` |
| Class Balancing | `imbalanced-learn` (SMOTE) |
| Deep Learning | `TensorFlow 2.x` / `Keras` |
| Clustering | `minisom`, `KMeans` |
| Image Processing | `OpenCV` |

---

## ⚙️ Setup & Run Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/vandana-kumari25/IBM-Data-Analytics-project.git
cd IBM-Data-Analytics-project
```

### 2. Create a Virtual Environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Fix the Dataset Path

Open `VandanaKumari_HepatitisC_ANN.ipynb` and update **Cell 5** to point to the local CSV:

```python
# Replace:
df = pd.read_csv("/kaggle/input/hepatitiscdata/hcvdat0.csv")

# With:
df = pd.read_csv("hcvdat0.csv")
```

### 5. Launch Jupyter and Run

```bash
jupyter notebook VandanaKumari_HepatitisC_ANN.ipynb
```

Then select **Kernel → Restart & Run All** to execute the full pipeline end-to-end.

---

## 📈 Evaluation Outputs

The notebook generates the following evaluation artefacts automatically:

- **Accuracy Score** — overall test-set classification accuracy
- **Classification Report** — per-class Precision, Recall, F1-Score
- **Confusion Matrix** — heatmap of true vs. predicted labels across all 5 classes
- **Predicted vs. True Plot** — scatter overlay of predicted (red) and actual (white) labels
- **Training History** — dual-panel plots of Accuracy and Loss curves over epochs

---

## 👩‍💻 Author

**Vandana Kumari**  
IBM Data Analytics Internship  
GitHub: [@vandana-kumari25](https://github.com/vandana-kumari25)
