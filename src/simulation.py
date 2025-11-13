"""
MS Lesion Simulation Module
Generates realistic Multiple Sclerosis lesion trajectories for patient cohorts
"""

import numpy as np


def apply_gene_effects(base_value, gene_factors, effect_type):
    """
    Apply gene-based modifications to attack or recovery rates.
    
    Args:
        base_value: Initial value before gene effects
        gene_factors: Dictionary of gene effects
        effect_type: Either 'immune' or 'neuron'
        
    Returns:
        float: Modified value after gene effects
    """
    if not gene_factors:
        return base_value
    
    modified_value = base_value
    for gene, effects in gene_factors.items():
        effect = effects.get(effect_type, 0)
        modified_value *= (1 + effect)
    
    return modified_value


def simulate_patient_trajectory(
    observation_days,
    drug_effectiveness,
    gene_factors=None,
    random_seed=None
):
    """
    Simulate lesion trajectory for a single patient.
    
    Args:
        observation_days: Number of days to simulate
        drug_effectiveness: Drug effect (0.0 to 1.0)
        gene_factors: Dictionary of gene effects
        random_seed: Optional seed for reproducibility
        
    Returns:
        np.array: Lesion counts over time
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Initialize patient-specific factors
    immune_factor = np.random.uniform(0.8, 1.2)
    neuron_resilience = np.random.uniform(0.8, 1.2)
    
    # Track lesion progression
    lesion_trajectory = np.zeros(observation_days)
    current_lesion = 0.0
    
    for day in range(observation_days):
        # Stochastic flare-ups (Poisson-distributed)
        flare = np.random.poisson(0.3)
        
        # Calculate attack intensity
        base_attack = np.random.uniform(0.3, 1.0) * immune_factor * (1 - drug_effectiveness)
        attack = base_attack + flare
        
        # Calculate recovery rate
        base_recovery = neuron_resilience * np.random.uniform(0.2, 0.7)
        recovery = base_recovery
        
        # Apply gene effects
        attack = apply_gene_effects(attack, gene_factors, "immune")
        recovery = apply_gene_effects(recovery, gene_factors, "neuron")
        
        # Update lesion count (cannot be negative)
        current_lesion = max(0, current_lesion + attack - recovery)
        lesion_trajectory[day] = current_lesion
    
    return lesion_trajectory


def simulate_cohort(
    num_patients=20,
    observation_days=20,
    drug_effectiveness=0.5,
    gene_factors=None
):
    """
    Simulate MS lesion trajectories for a cohort of patients.
    
    Args:
        num_patients: Number of patients to simulate
        observation_days: Number of days to track each patient
        drug_effectiveness: Treatment effectiveness (0.0 = no effect, 1.0 = full effect)
        gene_factors: Dictionary mapping genes to their effects on immune/neuron systems
        
    Returns:
        tuple: (days array, lesion_matrix)
            - days: Array of day numbers [1, 2, ..., observation_days]
            - lesion_matrix: Shape (num_patients, observation_days) with lesion counts
    """
    # Validate inputs
    if num_patients <= 0:
        raise ValueError("num_patients must be positive")
    if observation_days <= 0:
        raise ValueError("observation_days must be positive")
    if not 0 <= drug_effectiveness <= 1:
        raise ValueError("drug_effectiveness must be between 0 and 1")
    
    # Initialize output arrays
    days = np.arange(1, observation_days + 1)
    lesion_matrix = np.zeros((num_patients, observation_days))
    
    # Simulate each patient
    for patient_idx in range(num_patients):
        lesion_matrix[patient_idx] = simulate_patient_trajectory(
            observation_days=observation_days,
            drug_effectiveness=drug_effectiveness,
            gene_factors=gene_factors
        )
    
    return days, lesion_matrix