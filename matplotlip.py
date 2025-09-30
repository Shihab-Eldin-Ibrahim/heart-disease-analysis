import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Optional: Streamlit and Ngrok
try:
    import streamlit as st
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        RUNNING_IN_STREAMLIT = get_script_run_ctx() is not None
    except Exception:
        RUNNING_IN_STREAMLIT = False
except ImportError:
    st = None
    RUNNING_IN_STREAMLIT = False

try:
    from pyngrok import ngrok
except ImportError:
    ngrok = None

def train_and_plot(n_samples=500, n_features=10, test_size=0.2, C=1.0, random_state=42):
    # 1) Data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=min(5, n_features),
        n_redundant=min(2, max(0, n_features - 2)),
        n_classes=2,
        random_state=random_state
    )

    # 2) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 3) Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4) Model
    model = LogisticRegression(max_iter=1000, C=C, random_state=random_state)
    model.fit(X_train_scaled, y_train)

    # 5) Evaluate
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    # 6) Confusion matrix figure
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap='Blues')

    ax.set_title('Confusion Matrix - Logistic Regression')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black')

    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    return acc, report, fig

def run_streamlit_ui():
    st.title("Beginner ML Demo: Logistic Regression")
    st.write("Simple synthetic dataset + scaling + training + evaluation.")

    with st.sidebar:
        st.header("Settings")
        n_samples = st.slider("n_samples", 100, 2000, 500, 50)
        n_features = st.slider("n_features", 2, 30, 10, 1)
        test_size = st.slider("test_size", 0.1, 0.5, 0.2, 0.05)
        C = st.slider("LogReg C (regularization)", 0.01, 5.0, 1.0, 0.01)
        random_state = st.number_input("random_state", 0, 9999, 42, 1)

        st.markdown("---")
        st.subheader("Ngrok (Bonus)")
        start_ngrok = st.button("Start Ngrok Tunnel")
        if start_ngrok:
            if ngrok is None:
                st.warning("Install pyngrok: pip install pyngrok")
            else:
                token = os.environ.get("NGROK_AUTH_TOKEN")
                if token:
                    ngrok.set_auth_token(token)
                try:
                    tunnel = ngrok.connect(8501)
                    st.success(f"Public URL: {tunnel.public_url}")
                    st.caption("Share this URL to access your Streamlit app.")
                except Exception as e:
                    st.error(f"Ngrok error: {e}")

    if st.button("Train"):
        acc, report, fig = train_and_plot(
            n_samples=n_samples,
            n_features=n_features,
            test_size=test_size,
            C=C,
            random_state=int(random_state)
        )
        st.write(f"Accuracy: {acc:.4f}")
        st.text("Classification Report:")
        st.text(report)
        st.pyplot(fig)

def run_cli():
    print("Running in CLI mode. For UI: streamlit run matplotlip.py")
    acc, report, fig = train_and_plot()
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(report)
    plt.show()

if RUNNING_IN_STREAMLIT and st is not None:
    run_streamlit_ui()
else:
    if __name__ == "__main__":
        run_cli()