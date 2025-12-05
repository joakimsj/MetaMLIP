from ase.io import read, write
import numpy as np
from mace.calculators import MACECalculator
from tqdm import tqdm
import matplotlib.pyplot as plt

# Load all structures
all_structures = read('frames_for_DFT_eval.xyz', ':')

device = 'cuda'  # or 'cuda:0'

calculator = MACECalculator(
    model_paths='/scratch/project_462000838/active_learning_nextflow/input/MACE_models/mace-mpa-0-medium.model',
    device=device
)

SIMILARITY_THRESHOLD = 1.0

filtered_structures = []
filtered_descriptors = []

# Distance matrix will only track filtered structures
n_structures = len(all_structures)
max_possible = n_structures
distance_matrix = np.zeros((max_possible, max_possible))
kept_indices = []

for idx, atoms in enumerate(tqdm(all_structures, desc="Filtering structures")):
    try:
        desc = calculator.get_descriptors(atoms, invariants_only=False)
    except Exception as e:
        print(f"⚠️ Failed to get descriptor for structure {idx}, skipping: {e}")
        continue

    is_similar = False
    for j, existing_desc in enumerate(filtered_descriptors):
        distance = np.linalg.norm(desc - existing_desc)
        distance_matrix[len(filtered_descriptors)][j] = distance
        distance_matrix[j][len(filtered_descriptors)] = distance
        if distance < SIMILARITY_THRESHOLD:
            is_similar = True
            break

    if not is_similar:
        filtered_structures.append(atoms)
        filtered_descriptors.append(desc)
        kept_indices.append(idx)

# Crop matrix to size of filtered structures
n_filtered = len(filtered_structures)
cropped_matrix = distance_matrix[:n_filtered, :n_filtered]

# Plot heatmap
plt.figure(figsize=(8, 6))
plt.imshow(cropped_matrix, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Descriptor Distance')
plt.title('MACE Descriptor Distance Heatmap (Filtered Structures)')
plt.xlabel('Structure Index')
plt.ylabel('Structure Index')
plt.tight_layout()
plt.savefig('descriptor_heatmap.png', dpi=300)
#plt.show()

# Save filtered structures
if filtered_structures:
    write('frames_for_DFT_eval_filtered.xyz', filtered_structures, format='extxyz')
    print(f"✅ Filtered dataset written with {n_filtered} structures.")
else:
    print("❌ No unique structures kept after filtering.")
