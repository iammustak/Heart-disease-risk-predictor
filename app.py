from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "heart_model.pkl"
TEST_SCORES = {"ROC-AUC": "0.903", "Accuracy": "0.81", "Recall": "0.87", "Precision": "0.80"}

# Dropdown choices: model value -> plain-English text
OPTIONS = {
    "chest_pain_type": ("Chest pain type", {
        "typical_angina": "Typical angina",
        "atypical_angina": "Atypical angina",
        "non_anginal": "Non-anginal pain",
        "asymptomatic": "Asymptomatic",
    }),
    "st_slope": ("Exercise ECG pattern", {
        "upsloping": "Upward (usually normal)",
        "flat": "Flat (possible concern)",
        "downsloping": "Downward (most concerning)",
        "missing": "Not available",
    }),
    "thalassemia": ("Heart blood-flow scan result", {
        "normal": "Normal blood flow",
        "fixed_defect": "Permanent defect (old damage or scar)",
        "reversible_defect": "Temporary defect (artery narrowing under stress)",
        "missing": "Not available",
    }),
}
# Plain-English names for the numeric model columns (used in the factors chart)
NAMES = {
    "age": "Age", "is_male": "Male sex", "cholesterol": "Cholesterol level",
    "max_heart_rate": "Highest heart rate in exercise",
    "exercise_angina": "Chest pain during exercise",
    "st_depression": "Heart strain on exercise ECG",
    "num_major_vessels": "Blocked heart arteries",
}
# Probability limit -> label and the Streamlit message type used to show it
LEVELS = [(0.30, "Lower", "success"), (0.60, "Moderate", "warning"), (1.01, "Higher", "error")]


def feature_name(col):
    """Plain-English name for a model column."""
    if col in NAMES:
        return NAMES[col]
    for key, (title, choices) in OPTIONS.items():
        if col.startswith(key + "_"):
            return f"{title}: {choices[col[len(key) + 1:]].split(' (')[0]}"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Heart Disease Risk Assessment", page_icon="❤️", layout="wide")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("About this project")
    st.write("A machine-learning demo that uses a trained Logistic Regression model "
             "on the UCI Heart Disease dataset (916 patients).")
    st.subheader("Model scores")
    st.caption("Measured on a test set of 184 patients.")
    st.markdown("\n".join(f"- **{name}:** {score}" for name, score in TEST_SCORES.items()))
    st.warning("Educational use only. This tool cannot diagnose heart disease or replace "
               "a qualified healthcare professional.")

# ---------------- Page header ----------------
st.title("❤️ Heart Disease Risk Assessment")
st.caption("Fill in the details below and click **Predict risk**. Hover over the ❔ icon "
           "next to each field to see what it means and what a normal value looks like.")

try:
    model = load_model()
except FileNotFoundError:
    st.error("The model file `heart_model.pkl` was not found. Keep it next to `ui.py`.")
    st.stop()

# ---------------- Input form ----------------
with st.form("patient_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Basic information")
        age = st.number_input("**Age (years)**", 18, 100, 50,
                              help="Heart disease risk rises with age. Risk is higher for men "
                                   "over 45 and women over 55.")
        sex = st.radio("**Sex**", ["Male", "Female"], horizontal=True,
                       help="Men tend to develop heart disease earlier. In women the risk "
                            "rises after menopause.")
        chol = st.number_input("**Cholesterol level (mg/dl)**", 100, 600, 220,
                               help="Total cholesterol in the blood. Below 200 is desirable, "
                                    "200 to 239 is borderline high, and 240 or more is high.")
    with col2:
        st.subheader("Symptoms")
        cp = st.selectbox("**Chest pain type**", list(OPTIONS["chest_pain_type"][1]),
                          format_func=OPTIONS["chest_pain_type"][1].get,
                          help="Typical angina: pressure or tightness when exerting that eases "
                               "with rest. Atypical angina: only some of those features. "
                               "Non-anginal pain: chest pain not caused by the heart. "
                               "Asymptomatic: no chest pain.")
        exang = st.radio("**Chest pain during exercise**", ["No", "Yes"], horizontal=True,
                         help="Does exercise bring on chest pain? 'No' is normal. 'Yes' can "
                              "mean the heart is not getting enough blood during effort.")
        max_hr = st.number_input("**Highest heart rate during exercise (bpm)**", 60, 220, 150,
                                 help="The peak heart rate reached in an exercise test. A rough "
                                      "maximum is 220 minus age, and reaching about 85% of it "
                                      "is a good result. A much lower peak can point to a "
                                      "heart problem.")
    with col3:
        st.subheader("Test results")
        st_dep = st.number_input("**Heart strain on exercise ECG (mm)**", 0.0, 6.0, 1.0, 0.1,
                                 help="Also called ST depression. It shows how much the ECG "
                                      "line drops during exercise compared with rest. 0 is "
                                      "normal, 1 mm or more is abnormal, and 2 mm or more is "
                                      "more severe.")
        slope = st.selectbox("**Exercise ECG pattern**", list(OPTIONS["st_slope"][1]),
                             format_func=OPTIONS["st_slope"][1].get,
                             help="The shape of the ECG line at peak exercise (ST slope). "
                                  "Upward is usually normal, flat suggests reduced blood "
                                  "flow, and downward is the most concerning.")
        vessels = st.selectbox("**Blocked heart arteries (0 to 3)**", [0, 1, 2, 3],
                               help="How many of the 3 main heart arteries show narrowing "
                                    "on an X-ray scan. 0 is normal. The higher the number, "
                                    "the higher the risk.")
        thal = st.selectbox("**Heart blood-flow scan result**", list(OPTIONS["thalassemia"][1]),
                            format_func=OPTIONS["thalassemia"][1].get,
                            help="Result of a thallium stress scan that checks blood supply "
                                 "to the heart. 'Permanent defect' means old damage. "
                                 "'Temporary defect' means an artery is narrowed and the "
                                 "heart is short of blood under stress.")
    predict = st.form_submit_button("🔍 Predict risk", type="primary")

# ---------------- Result (shown only after the button is clicked) ----------------
if predict:
    # Build one row with the exact columns the model was trained on
    cols = list(model.feature_names_in_)
    row = dict.fromkeys(cols, 0)
    row.update({"age": age, "is_male": int(sex == "Male"), "cholesterol": chol,
                "max_heart_rate": max_hr, "exercise_angina": int(exang == "Yes"),
                "st_depression": st_dep, "num_major_vessels": vessels})
    for key, value in {"chest_pain_type": cp, "st_slope": slope, "thalassemia": thal}.items():
        row[f"{key}_{value}"] = 1  # dropdown choice -> one-hot column
    X = pd.DataFrame([row], columns=cols)

    prob = float(model.predict_proba(X)[0, 1])
    level, status = next((n, s) for limit, n, s in LEVELS if prob < limit)

    # How much each input moved the result (scaled value x model coefficient)
    impact = pd.Series(model[0].transform(X)[0] * model[-1].coef_[0], index=cols)
    top = impact.reindex(impact.abs().sort_values(ascending=False).index)[:6]
    top.index = [feature_name(c) for c in top.index]

    st.divider()
    st.subheader("Result")
    left, right = st.columns([1, 1.5])
    with left:
        st.metric("Estimated likelihood of heart disease", f"{prob:.0%}")
        st.progress(prob)
        getattr(st, status)(f"{level} estimated likelihood")
        st.caption("This is a model estimate, not a diagnosis.")
    with right:
        st.markdown("**What affected this result the most**")
        st.bar_chart(top)
        st.caption("A positive bar raises the estimate. A negative bar lowers it.")
    with st.expander("See the exact values sent to the model"):
        st.dataframe(X.T.rename(columns={0: "value"}))
    st.info("If you have chest pain, shortness of breath, fainting or sudden severe "
            "discomfort, seek urgent medical care. Do not use this app to decide whether "
            "care is needed.")

st.divider()
st.caption("Heart Disease Risk Assessment • Machine-learning learning project • Not for clinical use")
