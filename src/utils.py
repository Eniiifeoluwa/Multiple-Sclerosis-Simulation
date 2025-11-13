import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter

def plot_lesion_matrix(days, lesion_matrix, max_patients=10):
    df = pd.DataFrame(lesion_matrix[:max_patients].T, index=days)
    fig, ax = plt.subplots(figsize=(10,6))
    for col in df.columns:
        ax.plot(df.index, df[col], alpha=0.7)
    ax.set_xlabel("Observation Days")
    ax.set_ylabel("Lesion Count")
    ax.set_title("MS Lesion Progression (simulated)")
    return fig

def plot_kaplan_meier(lesion_matrix, threshold=2.0):
    """
    Create realistic Kaplan-Meier curve:
    - Event occurs when lesion count crosses threshold.
    """
    kmf = KaplanMeierFitter()
    n_patients, observation_days = lesion_matrix.shape
    durations = []
    events = []

    for i in range(n_patients):
        lesion = lesion_matrix[i]
        crossed = np.where(lesion >= threshold)[0]
        if len(crossed) > 0:
            event_time = crossed[0] + 1
            event_occurred = 1
        else:
            event_time = observation_days
            event_occurred = 0
        durations.append(event_time)
        events.append(event_occurred)

    kmf.fit(durations, event_observed=events)
    fig, ax = plt.subplots(figsize=(8,5))
    kmf.plot_survival_function(ax=ax, ci_show=True)
    ax.set_title("Synthetic Kaplan-Meier: Lesion-free Survival")
    ax.set_xlabel("Observation Days")
    ax.set_ylabel("Probability lesion-free")
    return fig
