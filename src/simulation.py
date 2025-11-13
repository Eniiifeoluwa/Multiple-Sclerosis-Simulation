import numpy as np

def simulate_cohort(num_patients=20, observation_days=20, drug_effectiveness=0.5, gene_factors=None):
    """
    Simulate realistic MS lesion trajectories for a cohort of patients.
    """
    days = np.arange(1, observation_days + 1)
    lesion_matrix = np.zeros((num_patients, observation_days))

    for i in range(num_patients):
        lesion = 0
        immune_factor = np.random.uniform(0.8, 1.2)
        neuron_resilience = np.random.uniform(0.8, 1.2)

        for t in range(observation_days):
            # Stochastic flare-ups
            flare = np.random.poisson(0.3)
            
            # Base lesion dynamics
            attack = np.random.uniform(0.3, 1.0) * immune_factor * (1 - drug_effectiveness) + flare
            recovery = neuron_resilience * np.random.uniform(0.2, 0.7)

            # Gene effects
            if gene_factors:
                for gene, effect in gene_factors.items():
                    attack *= 1 + effect.get("immune", 0)
                    recovery *= 1 + effect.get("neuron", 0)

            lesion = max(0, lesion + attack - recovery)
            lesion_matrix[i, t] = lesion

    return days, lesion_matrix
