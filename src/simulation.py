import numpy as np

def simulate_cohort(num_patients=20, observation_days=20, drug_effectiveness=0.5, gene_factors=None):
    """
    Simulate lesion trajectories for a cohort of patients.

    Parameters:
    - num_patients: int, number of patients
    - observation_days: int, number of days to simulate
    - drug_effectiveness: float [0,1], overall drug effect
    - gene_factors: dict, optional {"GeneA": {"immune": x, "neuron": y}, ...}

    Returns:
    - days: np.array of day numbers
    - lesion_matrix: np.array of shape (num_patients, observation_days)
    """
    days = np.arange(1, observation_days + 1)
    lesion_matrix = np.zeros((num_patients, observation_days))

    for i in range(num_patients):
        lesion = 0
        immune_factor = np.random.uniform(0.8, 1.2)
        neuron_resilience = np.random.uniform(0.8, 1.2)

        for t in range(observation_days):
            # Base lesion dynamics
            attack = np.random.uniform(0.5, 1.5) * immune_factor * (1 - drug_effectiveness)
            recovery = neuron_resilience * np.random.uniform(0.3, 0.8)

            # Gene effects
            if gene_factors:
                for gene, effect in gene_factors.items():
                    attack *= 1 + effect.get("immune", 0)
                    recovery *= 1 + effect.get("neuron", 0)

            lesion = max(0, lesion + attack - recovery)
            lesion_matrix[i, t] = lesion

    return days, lesion_matrix
