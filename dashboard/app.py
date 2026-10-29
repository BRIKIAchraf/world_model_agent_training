import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


METRICS_DIR = Path("outputs/metrics")

st.set_page_config(
    page_title="World Model Agent",
    layout="wide",
)

st.title("World Model Agent - suivi des expériences")
st.caption(
    "Les graphiques utilisent les métriques sauvegardées pendant les entraînements."
)


def load_metrics(path):
    return json.loads(path.read_text(encoding="utf-8"))


if not METRICS_DIR.exists():
    st.info("Aucune métrique disponible. Lancez un entraînement instrumenté.")
    st.stop()

metric_files = sorted(METRICS_DIR.glob("*.json"))

if not metric_files:
    st.info("Aucun fichier de métriques trouvé.")
    st.stop()

selected_file = st.selectbox(
    "Expérience",
    metric_files,
    format_func=lambda path: path.stem,
)

metrics = load_metrics(selected_file)
history = metrics.get("history", {})
epochs = history.get("epoch", [])

if epochs:
    st.subheader("Évolution de l'entraînement")
    columns = st.columns(2)

    if "loss" in history:
        with columns[0]:
            loss_data = pd.DataFrame(
                {
                    "epoch": epochs,
                    "Loss": history["loss"],
                }
            )
            st.line_chart(
                loss_data,
                x="epoch",
                y="Loss",
            )

    if "accuracy" in history:
        with columns[1]:
            accuracy_data = pd.DataFrame(
                {
                    "epoch": epochs,
                    "Accuracy": history["accuracy"],
                }
            )
            st.line_chart(
                accuracy_data,
                x="epoch",
                y="Accuracy",
            )

evaluation = metrics.get("evaluation", {})

if evaluation:
    st.subheader("Résultats d'évaluation")
    st.metric(
        "Accuracy globale",
        f"{evaluation.get('accuracy', 0.0):.3f}",
    )

    if "balanced_accuracy" in evaluation:
        st.metric(
            "Balanced accuracy",
            f"{evaluation['balanced_accuracy']:.3f}",
        )

    confusion = evaluation.get("confusion")

    if confusion:
        st.subheader("Matrice de confusion")

        figure, axis = plt.subplots()
        matrix = [
            [confusion["true_0_pred_0"], confusion["true_0_pred_1"]],
            [confusion["true_1_pred_0"], confusion["true_1_pred_1"]],
        ]

        axis.imshow(matrix, cmap="Blues")
        axis.set_xlabel("Classe prédite")
        axis.set_ylabel("Vraie classe")
        axis.set_xticks([0, 1], ["Normale", "Défectueuse"])
        axis.set_yticks([0, 1], ["Normale", "Défectueuse"])

        for row in range(2):
            for column in range(2):
                axis.text(
                    column,
                    row,
                    matrix[row][column],
                    ha="center",
                    va="center",
                )

        st.pyplot(figure)

st.subheader("Données brutes")
st.json(metrics)

