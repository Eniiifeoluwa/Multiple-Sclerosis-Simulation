"""
Visualization and Analysis Utilities
Functions for plotting lesion trajectories and survival curves
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter


def plot_lesion_matrix(days, lesion_matrix, max_patients=10):
    """
    Plot lesion progression trajectories for multiple patients.
    
    Args:
        days: Array of observation days
        lesion_matrix: Matrix of lesion counts (patients × days)
        max_patients: Maximum number of patient trajectories to display
        
    Returns:
        matplotlib.figure.Figure: The generated plot
    """
    # Limit to max_patients
    display_matrix = lesion_matrix[:max_patients]
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame(
        display_matrix.T,
        index=days,
        columns=[f"Patient {i+1}" for i in range(len(display_matrix))]
    )
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for column in df.columns:
        ax.plot(df.index, df[column], alpha=0.7, linewidth=2, label=column)
    
    # Formatting
    ax.set_xlabel("Observation Days", fontsize=12)
    ax.set_ylabel("Lesion Count", fontsize=12)
    ax.set_title("MS Lesion Progression (Simulated Cohort)", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legend only if few patients
    if len(display_matrix) <= 5:
        ax.legend(loc='best', framealpha=0.9)
    
    plt.tight_layout()
    return fig


def calculate_event_times(lesion_matrix, threshold=2.0):
    """
    Calculate time to event (lesion threshold crossing) for each patient.
    
    Args:
        lesion_matrix: Matrix of lesion counts (patients × days)
        threshold: Lesion count threshold defining an "event"
        
    Returns:
        tuple: (durations, events)
            - durations: Time to event or censoring
            - events: 1 if event occurred, 0 if censored
    """
    n_patients, observation_days = lesion_matrix.shape
    durations = []
    events = []
    
    for patient_idx in range(n_patients):
        patient_lesions = lesion_matrix[patient_idx]
        
        # Find first day when threshold is crossed
        threshold_days = np.where(patient_lesions >= threshold)[0]
        
        if len(threshold_days) > 0:
            # Event occurred
            event_time = threshold_days[0] + 1  # Convert to 1-indexed
            event_occurred = 1
        else:
            # Patient remained below threshold (censored)
            event_time = observation_days
            event_occurred = 0
        
        durations.append(event_time)
        events.append(event_occurred)
    
    return durations, events


def plot_kaplan_meier(lesion_matrix, threshold=2.0):
    """
    Create Kaplan-Meier survival curve for lesion-free survival.
    
    Event is defined as lesion count crossing the specified threshold.
    
    Args:
        lesion_matrix: Matrix of lesion counts (patients × days)
        threshold: Lesion count threshold for defining an event
        
    Returns:
        matplotlib.figure.Figure: The generated survival curve
    """
    # Calculate event times
    durations, events = calculate_event_times(lesion_matrix, threshold)
    
    # Fit Kaplan-Meier model
    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed=events, label='Lesion-Free Survival')
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 5))
    kmf.plot_survival_function(ax=ax, ci_show=True)
    
    # Formatting
    ax.set_title(
        f"Kaplan-Meier: Lesion-Free Survival\n(Threshold: {threshold} lesions)",
        fontsize=14,
        fontweight='bold'
    )
    ax.set_xlabel("Observation Days", fontsize=12)
    ax.set_ylabel("Probability of Remaining Lesion-Free", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Add summary statistics
    median_survival = kmf.median_survival_time_
    if not np.isnan(median_survival):
        ax.axvline(
            median_survival,
            color='red',
            linestyle='--',
            alpha=0.5,
            label=f'Median: {median_survival:.1f} days'
        )
        ax.legend()
    
    plt.tight_layout()
    return fig