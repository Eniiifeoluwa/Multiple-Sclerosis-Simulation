import numpy as np

class Patient:
    def __init__(self, id, neuron_resilience=1.0, immune_activity=1.0):
        self.id = id
        self.neuron_resilience = neuron_resilience
        self.immune_activity = immune_activity
        self.lesion_count = 0

def simulate_cohort(num_patients=20, timesteps=50, drug_effectiveness=0.2,
                    gene_factors=None, remission_chance=0.05, immune_variability=0.1):
    patients = [Patient(
        i,
        neuron_resilience=np.random.uniform(0.8, 1.2),
        immune_activity=np.random.uniform(0.8, 1.2)
    ) for i in range(num_patients)]

    lesion_matrix = np.zeros((num_patients, timesteps))

    for t in range(timesteps):
        for i, p in enumerate(patients):
            attack = p.immune_activity * np.random.uniform(1-immune_variability, 1+immune_variability)
            recovery = p.neuron_resilience * np.random.uniform(0.3, 0.8)

            if gene_factors:
                for gene, effect in gene_factors.items():
                    attack *= 1 + effect.get("immune",0)
                    recovery *= 1 + effect.get("neuron",0)

            delta = attack - recovery
            p.lesion_count += max(delta * (1 - drug_effectiveness),0)

            # Random remission
            if np.random.rand() < remission_chance:
                p.lesion_count = max(p.lesion_count - np.random.uniform(0.5, 1.0), 0)

            lesion_matrix[i, t] = p.lesion_count

    return np.arange(timesteps), lesion_matrix
