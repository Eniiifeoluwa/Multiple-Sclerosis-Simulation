# utils.py
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter
import numpy as np

def plot_lesion_matrix(days, lesion_matrix, max_patients=10):
    """Return a Matplotlib figure of lesion trajectories for Streamlit."""
    df = pd.DataFrame(lesion_matrix[:max_patients].T, index=days)
    fig, ax = plt.subplots(figsize=(10,6))
    for col in df.columns:
        ax.plot(df.index, df[col], alpha=0.7)
    ax.set_xlabel("Observation Days")
    ax.set_ylabel("Lesion Count")
    ax.set_title("MS Lesion Progression (simulated)")
    return fig

def plot_kaplan_meier(lesion_matrix, threshold=5):
    """Return a Matplotlib figure of synthetic Kaplan-Meier curve for Streamlit."""
    kmf = KaplanMeierFitter()
    n_patients, observation_days = lesion_matrix.shape
    durations = []
    events = []
    for i in range(n_patients):
        lesion = lesion_matrix[i]
        event_time = np.argmax(lesion >= threshold) + 1 if np.any(lesion >= threshold) else observation_days
        event_occurred = int(lesion[event_time-1] >= threshold)
        durations.append(event_time)
        events.append(event_occurred)
    kmf.fit(durations, event_observed=events)
    fig, ax = plt.subplots(figsize=(8,5))
    kmf.plot_survival_function(ax=ax)
    ax.set_title("Synthetic Kaplan-Meier: Lesion-free Survival")
    ax.set_xlabel("Observation Days")
    ax.set_ylabel("Probability lesion-free")
    return fig
