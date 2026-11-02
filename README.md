<<<<<<< HEAD
# Heart Disease Risk Assessment

A Streamlit application that demonstrates a machine-learning classification model using patient health measurements from the supplied cleaned heart disease dataset.

## Project contents

- `ui.py` — English-language Streamlit user interface
- `heart_model.pkl` — trained Logistic Regression pipeline
- `heart_disease_ml_ready.csv` — cleaned dataset (916 records, 19 input features and target)
- `heart_disease_model.ipynb` — model training and evaluation notebook
- `requirements.txt` — Python dependencies

## Run locally

Python 3.10–3.12 is recommended because the supplied model was saved with scikit-learn 1.4.0.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run ui.py
```

### Windows

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run ui.py
```

Open the local URL printed in the terminal (usually `http://localhost:8501`).

## Model output

The app displays the probability returned by the supplied Logistic Regression pipeline and groups it into a simple lower, moderate, or higher model-estimated likelihood band. These bands are presentation labels, not medical thresholds.

## Important limitations and safety

- This is an educational machine-learning project, not a medical device.
- The model was trained on a relatively small dataset of 916 cleaned patient records.
- The source dataset is intended for heart disease classification/diagnosis and does not establish a person's future 10-year risk.
- Model probabilities and results may not generalize to other populations or clinical settings.
- Do not use this application to diagnose, treat, or rule out a condition, or to delay medical care. A qualified healthcare professional must interpret medical information.

## Troubleshooting

- **Model file not found:** keep `heart_model.pkl` in the same directory as `ui.py`.
- **ModuleNotFoundError:** activate the virtual environment and run `pip install -r requirements.txt`.
- **scikit-learn version warning/error:** use Python 3.10–3.12 and the pinned `scikit-learn==1.4.0` dependency in this project.
- **Port already in use:** run `streamlit run ui.py --server.port 8502`.

## Dataset reference

The original project describes the UCI Heart Disease dataset, collected from four hospitals. See the [UCI dataset page](https://archive.ics.uci.edu/dataset/45/heart+disease).
=======
# Heart-disease-risk-predictor
>>>>>>> e1c6fe3ca0d480c2b2aff7f0bd3468672d14dc03
