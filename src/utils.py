import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter
import numpy as np

def plot_lesion_matrix(times, lesion_matrix, max_patients=10):
    """Plot lesion trajectories for first `max_patients` patients."""
    df = pd.DataFrame(lesion_matrix[:max_patients].T, index=times)
    plt.figure(figsize=(10,6))
    for col in df.columns:
        plt.plot(df.index, df[col], alpha=0.7)
    plt.xlabel("Time steps")
    plt.ylabel("Lesion Count")
    plt.title("MS Lesion Progression (simulated)")
    plt.show()

def plot_kaplan_meier(lesion_matrix, threshold=5):
    """Plot synthetic Kaplan-Meier curve for lesion-free survival."""
    kmf = KaplanMeierFitter()
    n_patients, timesteps = lesion_matrix.shape
    durations = []
    events = []
    for i in range(n_patients):
        lesion = lesion_matrix[i]
        event_time = np.argmax(lesion >= threshold) + 1 if np.any(lesion >= threshold) else timesteps
        event_occurred = int(lesion[event_time-1] >= threshold)
        durations.append(event_time)
        events.append(event_occurred)
    kmf.fit(durations, event_observed=events)
    kmf.plot_survival_function()
    plt.title("Synthetic Kaplan-Meier: Lesion-free Survival")
    plt.xlabel("Time steps")
    plt.ylabel("Probability lesion-free")
    plt.show()
