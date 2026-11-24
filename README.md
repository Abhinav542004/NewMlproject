# 📊 Vendor Invoice Intelligence Portal

An internal analytics portal powered by machine learning to predict freight costs and flag risky vendor invoices for manual review.

---

## 🚀 Features

| Module | Algorithm | Task |
|---|---|---|
| **Freight Cost Prediction** | Linear / Decision Tree / Random Forest Regression | Predict freight cost from invoice amount |
| **Invoice Risk Flagging** | Random Forest Classifier (GridSearchCV tuned) | Flag invoices that require manual approval |

---

## 🗂️ Project Structure

```
NewMlproject/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
│
├── data/
│   └── inventory.db                # SQLite database (vendor_invoice + purchases tables)
│
├── Freight cost prediction/
│   ├── data_preprocessing.py       # Load & prepare freight data
│   ├── modeling_evaluation.py      # Train & evaluate regression models
│   ├── train.py                    # Training pipeline entry point
│   └── models/
│       └── predict_freight_model.pkl
│
├── invoice_flagging/
│   ├── data_preprocessing.py       # Load, label & scale invoice data
│   ├── modelling_evaluation.py     # Train & evaluate classifier
│   ├── train.py                    # Training pipeline entry point
│   └── models/
│       ├── invoice_risk_model.pkl
│       └── scaler.pkl
│
├── inference/
│   ├── predict_freight.py          # Freight cost inference module
│   └── predict_invoice_flag.py     # Invoice risk inference module
│
└── notebooks/
    ├── Predecting Freight cost.ipynb
    └── Invoice Flaging.ipynb
```

---

## ⚙️ Setup & Installation

**1. Clone the repository**
```bash
git clone <your-repo-url>
cd NewMlproject
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Ensure the database is in place**

Place `inventory.db` in the `data/` folder. The database must contain:
- `vendor_invoice` table — with columns: `PONumber`, `Quantity`, `Dollars`, `Freight`, `InvoiceDate`, `PODate`, `PayDate`
- `purchases` table — with columns: `PONumber`, `Brand`, `Quantity`, `Dollars`, `ReceivingDate`, `PODate`

---

## 🏋️ Retraining the Models

**Freight Cost model** (run from inside the `Freight cost prediction/` folder):
```bash
cd "Freight cost prediction"
python train.py
```

**Invoice Risk model** (run from inside the `invoice_flagging/` folder):
```bash
cd invoice_flagging
python train.py
```

Pre-trained `.pkl` files are already included. Only retrain if the underlying data changes.

---

## ▶️ Running the App

From the project root:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🧠 Modelling Details

### Freight Cost Prediction
- **Features:** Invoice Dollars (`Dollars`)
- **Models evaluated:** Linear Regression, Decision Tree Regressor, Random Forest Regressor
- **Selection criterion:** Lowest Mean Absolute Error (MAE)
- **Metrics reported:** MAE, RMSE, R²

### Invoice Risk Flagging
- **Features:** `invoice_quantity`, `invoice_dollars`, `Freight`, `total_item_quantity`, `total_item_dollars`
- **Model:** Random Forest Classifier with `GridSearchCV` (5-fold CV, F1 scoring)
- **Class balancing:** `class_weight="balanced"`
- **Labelling logic:** Invoices are flagged if invoice amount differs from PO amount by > \$5, or if the average receiving delay exceeds 10 days
- **Metrics reported:** Accuracy, F1-score, Classification Report, Confusion Matrix

---

## 📌 Notes

- Models are cached at app startup using `@st.cache_resource` — no reload on every prediction.
- The scaler (`scaler.pkl`) is saved during training and must be present alongside `invoice_risk_model.pkl` for inference to work.
- Python 3.10+ recommended (project was developed on CPython 3.14).
