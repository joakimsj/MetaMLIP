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
parser.add_argument("--max_structures", type=int, default=None, help="Maximum number of structures to save for reference calcs")
args = parser.parse_args()

device = 'cuda'

# Load structures
new_structures = read(args.new, ":")

# Define the calculator and models
calculator = MACECalculator(
    model_paths=args.model,
    device=device
)

# Load reference structures if file exists and is not empty
reference_structures = []
if os.path.exists(args.reference) and os.path.getsize(args.reference) > 0:
    try:
        reference_structures = read(args.reference, ":")
        if len(reference_structures) == 0:
            print(f"Reference dataset '{args.reference}' is empty after reading. Proceeding without reference comparison.")
    except Exception as e:
        print(f"Could not read reference dataset '{args.reference}': {e}")
        print("Proceeding without reference comparison.")
else:
    print(f"Reference dataset '{args.reference}' not found or empty. Proceeding without reference comparison.")

# Compute descriptors for reference set (only if non-empty)
ref_descriptors = []
if len(reference_structures) == 0:
    print("No reference structures loaded — skipping reference descriptor computation.")
else:
    print("Computing descriptors for reference dataset...")
    for idx, atoms in enumerate(tqdm(reference_structures, desc='Reference descriptors')):
        try:
            desc = calculator.get_descriptors(atoms, invariants_only=False)
            ref_descriptors.append(desc)
        except Exception as e:
            print(f"Skipping reference structure {idx} due to error: {e}")


# Check the total number of structures (ensuring enough for finetune)

num_new = len(new_structures)
num_ref = len(reference_structures)
total_structures = num_new + num_ref

print(f"Total structures available (reference + new) = {total_structures}")

# If we have fewer than 20 total structures, request another MTD sampling round
if total_structures < 20:
    print("Fewer than 20 total structures. Requesting more MTD sampling...")
    exit(10)  # <--- return a non-zero exit code to trigger re-run


# Now process new structures
print("Filtering new structures against reference and each other...")
filtered_structures = []
filtered_descriptors = []

distance_matrix = np.zeros((len(new_structures), len(new_structures)))  # Optional for plotting

for i, atoms in enumerate(tqdm(new_structures, desc="Filtering new structures")):
    if args.max_structures is not None and len(filtered_structures) >= args.max_structures:
        print(f"Reached maximum number of {args.max_structures} structures. Stopping filtering.")
        break

    try:
        desc = calculator.get_descriptors(atoms, invariants_only=False)
    except Exception as e:
        print(f"Skipping structure {i} due to descriptor error: {e}")
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


# Count total structures including reference to ensure enough for training
num_new_filtered = len(filtered_structures)
total_structures_filtered = num_new_filtered + num_ref

print(f"Total structures available, including filtered (reference + new_filtered) = {total_structures_filtered}")

# If we have fewer than 20 total structures, request another MTD sampling round
if total_structures_filtered < 20:
    print("Fewer than 20 total structures. Requesting more MTD sampling...")
    exit(10)  # <--- return a non-zero exit code to trigger re-run


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
    print(f"Saved {len(filtered_structures)} filtered new structures to {args.output}")
else:
    print(" No new unique structures found.")
