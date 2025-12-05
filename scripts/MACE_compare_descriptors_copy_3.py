from ase.io import read, write
import numpy as np
from mace.calculators import MACECalculator
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser()
parser.add_argument("--new", default="frames_for_DFT_eval.xyz", help="Path to new candidate structures")
parser.add_argument("--reference", default="growing_dataset.xyz", help="Path to existing dataset")
parser.add_argument("--output", default="frames_for_DFT_eval_filtered.xyz", help="Output path for filtered structures")
parser.add_argument("--threshold", type=float, default=1.0, help="Descriptor distance threshold for similarity")
parser.add_argument("--model", default="/scratch/project_462000838/active_learning_nextflow/input/MACE_models/mace-mpa-0-medium.model", help="Path to MACE model")
args = parser.parse_args()

device = 'cuda'

# Load structures
new_structures = read(args.new, ":")

# Load reference structures if file exists
if os.path.exists(args.reference):
    reference_structures = read(args.reference, ":")
else:
    print(f"⚠️ Reference dataset '{args.reference}' not found. Proceeding without reference comparison.")
    reference_structures = []

calculator = MACECalculator(
    model_paths=args.model,
    device=device
)

# Compute descriptors for reference set (growing dataset)
print("📦 Computing descriptors for reference dataset...")
ref_descriptors = []
for idx, atoms in enumerate(tqdm(reference_structures, desc="Reference descriptors")):
    try:
        desc = calculator.get_descriptors(atoms, invariants_only=False)
        ref_descriptors.append(desc)
    except Exception as e:
        print(f"⚠️ Skipping reference structure {idx} due to error: {e}")

# Now process new structures
print("🔍 Filtering new structures against reference and each other...")
filtered_structures = []
filtered_descriptors = []

distance_matrix = np.zeros((len(new_structures), len(new_structures)))  # Optional for plotting

for i, atoms in enumerate(tqdm(new_structures, desc="Filtering new structures")):
    try:
        desc = calculator.get_descriptors(atoms, invariants_only=False)
    except Exception as e:
        print(f"⚠️ Skipping structure {i} due to descriptor error: {e}")
        continue

    is_similar = False

    # Compare with reference dataset
    for ref_desc in ref_descriptors:
        if np.linalg.norm(desc - ref_desc) < args.threshold:
            is_similar = True
            break

    # Compare with already accepted new structures
    if not is_similar:
        for j, existing_desc in enumerate(filtered_descriptors):
            dist = np.linalg.norm(desc - existing_desc)
            distance_matrix[len(filtered_descriptors)][j] = dist
            distance_matrix[j][len(filtered_descriptors)] = dist
            if dist < args.threshold:
                is_similar = True
                break

    if not is_similar:
        filtered_structures.append(atoms)
        filtered_descriptors.append(desc)

# Optional: Save distance heatmap
if len(filtered_descriptors) > 1:
    n = len(filtered_descriptors)
    cropped = distance_matrix[:n, :n]
    plt.figure(figsize=(8, 6))
    plt.imshow(cropped, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Descriptor Distance')
    plt.title('Filtered New Structures Descriptor Distances')
    plt.xlabel('Structure Index')
    plt.ylabel('Structure Index')
    plt.tight_layout()
    plt.savefig('descriptor_heatmap.png', dpi=300)

# Save output
if filtered_structures:
    write(args.output, filtered_structures, format='extxyz')
    print(f"✅ Saved {len(filtered_structures)} filtered new structures to {args.output}")
else:
    print("❌ No new unique structures found.")
