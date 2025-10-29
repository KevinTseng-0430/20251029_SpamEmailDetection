import joblib
from pathlib import Path
import json
import random
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split

BASE = Path(__file__).parent.resolve()
MODEL_PATH = BASE / "model.joblib"
VECT_PATH = BASE / "vectorizer.joblib"
METRICS_PATH = BASE / "metrics.json"
EXAMPLES_PATH = BASE / "examples.json"


def load_artifacts():
    if not MODEL_PATH.exists() or not VECT_PATH.exists():
        return None, None, None, None
    clf = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECT_PATH)
    metrics = None
    examples = None
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r") as f:
                metrics = json.load(f)
        except Exception:
            metrics = None
    if EXAMPLES_PATH.exists():
        try:
            with open(EXAMPLES_PATH, "r") as f:
                examples = json.load(f)
        except Exception:
            examples = None
    return clf, vectorizer, metrics, examples


DATA_URL = (
    "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
)


st.set_page_config(page_title="Spam classifier — demo", layout="wide")
st.markdown("# 📬 Spam Classifier — Baseline (Logistic Regression)")

clf, vectorizer, metrics, examples = load_artifacts()

if clf is None:
    st.warning("Model artifacts not found. Run `python train.py` to train the baseline model.")
else:
    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        threshold = st.slider("Spam confidence threshold", 0.0, 1.0, 0.5, 0.01)
        st.write("Model artifacts:")
        st.write(f"- model: {MODEL_PATH.name}")
        st.write(f"- vectorizer: {VECT_PATH.name}")
        if metrics:
            st.write(f"- trained on {metrics.get('n_train', '?')} samples")
            if st.button("Show model metrics"):
                st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
                st.write(f"Precision: {metrics.get('precision', 0):.4f}")
                st.write(f"Recall: {metrics.get('recall', 0):.4f}")
                st.write(f"F1: {metrics.get('f1', 0):.4f}")

    # Main UI
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Classify a message")

        # Class distribution histogram (try dataset first, fallback to examples)
        try:
            df_counts = None
            df = pd.read_csv(DATA_URL, header=None, names=["label", "message"])  # may be cached by pandas
            counts = df["label"].value_counts()
            df_counts = pd.Series({k: int(v) for k, v in counts.items()})
            st.bar_chart(df_counts)
        except Exception:
            # fallback: use examples if available
            if examples:
                from collections import Counter

                ctr = Counter([ex.get("true", 0) for ex in examples])
                # map 0->ham,1->spam
                counts = {"ham": ctr.get(0, 0), "spam": ctr.get(1, 0)}
                st.bar_chart(pd.Series(counts))

        # Performance charts (attempt to compute using test split)
        try:
            # load dataset and recreate a test split consistent with training
            df = pd.read_csv(DATA_URL, header=None, names=["label", "message"])  # re-use same source
            df = df.dropna()
            X_all = df["message"].astype(str).values
            y_all = df["label"].map({"ham": 0, "spam": 1}).values
            X_train_ds, X_test_ds, y_train_ds, y_test_ds = train_test_split(
                X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
            )
            X_test_t_ds = vectorizer.transform(X_test_ds)
            y_score = clf.predict_proba(X_test_t_ds)[:, 1]
            y_pred_ds = clf.predict(X_test_t_ds)

            # Confusion matrix
            cm = confusion_matrix(y_test_ds, y_pred_ds)
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ham", "spam"])
            disp.plot(ax=ax_cm, cmap=plt.cm.Blues, colorbar=False)
            ax_cm.set_title("Confusion Matrix")
            st.pyplot(fig_cm)

            # ROC curve
            fpr, tpr, _ = roc_curve(y_test_ds, y_score)
            roc_auc = auc(fpr, tpr)
            fig_roc, ax_roc = plt.subplots(figsize=(4, 3))
            ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
            ax_roc.plot([0, 1], [0, 1], "--", color="gray")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title("ROC Curve")
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)

            # Precision-Recall
            precision, recall, _ = precision_recall_curve(y_test_ds, y_score)
            fig_pr, ax_pr = plt.subplots(figsize=(4, 3))
            ax_pr.plot(recall, precision)
            ax_pr.set_xlabel("Recall")
            ax_pr.set_ylabel("Precision")
            ax_pr.set_title("Precision-Recall")
            st.pyplot(fig_pr)

            # Top tokens by coefficient
            try:
                feat_names = vectorizer.get_feature_names_out()
                coefs = clf.coef_[0]
                top_pos = sorted(zip(coefs, feat_names), reverse=True)[:10]
                top_neg = sorted(zip(coefs, feat_names))[:10]
                df_pos = pd.DataFrame(top_pos, columns=["coef", "token"]).assign(sign="spam")
                df_neg = pd.DataFrame(top_neg, columns=["coef", "token"]).assign(sign="ham")
                st.subheader("Top tokens")
                cols_tt = st.columns(2)
                cols_tt[0].write("Top spam tokens")
                cols_tt[0].table(df_pos)
                cols_tt[1].write("Top ham tokens")
                cols_tt[1].table(df_neg)
            except Exception:
                pass
        except Exception:
            # If computing charts failed, quietly continue
            pass

        # ensure session state key exists so example buttons can set it
        if "message" not in st.session_state:
            st.session_state["message"] = ""

        # text area bound to session state so updates persist across reruns
        st.text_area("Enter email / SMS text to classify", key="message", height=220)

        # helper to set message from examples
        def _set_message(txt: str):
            st.session_state["message"] = txt

        # Example generators: one ham and one spam sample button
        st.markdown("**Generate example message**")
        gen_cols = st.columns(2)
        ham_examples = [ex.get("text", "") for ex in (examples or []) if ex.get("true", 0) == 0]
        spam_examples = [ex.get("text", "") for ex in (examples or []) if ex.get("true", 0) == 1]

        def _set_random_from_list(lst):
            if not lst:
                return
            _set_message(random.choice(lst))

        gen_cols[0].button("Generate ham message", on_click=_set_random_from_list, args=(ham_examples,))
        gen_cols[1].button("Generate spam message", on_click=_set_random_from_list, args=(spam_examples,))

        # read current message from session state
        message = st.session_state.get("message", "")

        if st.button("Predict"):
            if not message.strip():
                st.info("Type or paste a message to classify")
            else:
                X = vectorizer.transform([message])
                prob = clf.predict_proba(X)[0]
                pred = clf.predict(X)[0]
                spam_prob = float(prob[1])
                label = "spam" if pred == 1 else "ham"

                # display result with colored confidence bar
                st.markdown("---")
                st.markdown(f"### Result: **{label.upper()}**")
                # choose color: red for spam, green for ham
                color = "#e02424" if label == "spam" else "#2ecc71"
                pct = int(spam_prob * 100)
                bar_html = f"""
<div style='background-color:#eee; width:100%; height:22px; border-radius:6px;'>
  <div style='width:{pct}%; height:100%; background-color:{color}; border-radius:6px;'></div>
</div>
<div style='margin-top:6px; font-size:14px;'>Confidence (spam): <strong>{spam_prob:.3f}</strong></div>
"""
                components.html(bar_html, height=60)
                if spam_prob >= threshold:
                    st.success("Above threshold — treat as spam")
                else:
                    st.info("Below threshold — treat as ham")

    with col2:
        st.subheader("Model summary")
        if metrics:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
            st.write(f"Precision: {metrics.get('precision', 0):.4f}")
            st.write(f"Recall: {metrics.get('recall', 0):.4f}")
            st.write(f"F1: {metrics.get('f1', 0):.4f}")
        else:
            st.write("No metrics available. Re-run `python train.py` to generate metrics.json.")

        st.markdown("---")
        st.subheader("Quick tips")
        st.write("- Paste an SMS or email body into the text box and click Predict.")
        st.write("- Use the example buttons to try sample messages.")
        st.write("- Adjust the spam confidence threshold in the sidebar.")
