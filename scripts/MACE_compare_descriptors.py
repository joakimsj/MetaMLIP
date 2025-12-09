from ase.io import read, write
from collections import Counter
import numpy as np
from mace.calculators import MACECalculator
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import os

# -----------------------
# Helper: Structure Signature
# -----------------------
def structure_signature(atoms):
    """Return a canonical chemical signature based on atom types."""
    counts = Counter(atoms.get_chemical_symbols())
    return tuple(sorted(counts.items()))  # e.g. (('C',7),('H',10),('O',2))


# -----------------------
# Arguments
# -----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--new", default="frames_for_DFT_eval.xyz",
                    help="Path to new candidate structures")
parser.add_argument("--reference", default="growing_dataset.xyz",
                    help="Path to existing dataset")
parser.add_argument("--output", default="frames_for_DFT_eval_filtered.xyz",
                    help="Filtered output structure file")
parser.add_argument("--threshold", type=float, default=1.0,
                    help="Descriptor distance threshold")
parser.add_argument("--model",
                    default="/scratch/project_462000838/active_learning_nextflow/input/MACE_models/mace-mpa-0-medium.model",
                    help="Path to MACE model")
parser.add_argument("--max_structures", type=int, default=None,
                    help="Maximum number of structures to keep")
parser.add_argument("--min_new_structures", type=int, default=20,
                    help="If fewer than this number of *new* structures are present or pass filtering, "
                         "exit(10) to request new metadynamics sampling.")
args = parser.parse_args()

device = "cuda"

# -----------------------
# Load new candidate structures
# -----------------------
new_structures = read(args.new, ":")
print(f"Loaded {len(new_structures)} new structures.")

# --- NEW REQUIREMENT: Request new MTD runs if too few new structures ---
if len(new_structures) < args.min_new_structures:
    print(f"Only {len(new_structures)} new structures. "
          f"Need at least {args.min_new_structures}. Requesting more MTD sampling...")
    exit(10)

# -----------------------
# Load MACE calculator
# -----------------------
calculator = MACECalculator(
    model_paths=args.model,
    device=device
)

# -----------------------
# Load reference dataset
# -----------------------
reference_structures = []
if os.path.exists(args.reference) and os.path.getsize(args.reference) > 0:
    try:
        reference_structures = read(args.reference, ":")
        print(f"Loaded {len(reference_structures)} reference structures.")
    except Exception as e:
        print(f"Warning: failed to read reference dataset: {e}")
else:
    print("Reference dataset missing or empty.")

# -----------------------
# Precompute reference descriptors by chemical signature
# -----------------------
signature_to_ref_desc = {}

if len(reference_structures) > 0:
    print("Computing descriptors for reference dataset...")
    for atoms in tqdm(reference_structures, desc="Reference descriptors"):
        sig = structure_signature(atoms)
        try:
            desc = calculator.get_descriptors(atoms, invariants_only=False)
        except Exception as e:
            print(f"Skipping reference structure due to descriptor error: {e}")
            continue
        signature_to_ref_desc.setdefault(sig, []).append(desc)


# -----------------------
# FILTER NEW STRUCTURES
# -----------------------
print("Filtering new structures (signature-aware)...")

filtered_structures = []
filtered_descriptors = []
filtered_signatures = []

distance_matrix = np.zeros((len(new_structures), len(new_structures)))

for i, atoms in enumerate(tqdm(new_structures, desc="Filtering new structures")):

    if args.max_structures is not None and len(filtered_structures) >= args.max_structures:
        print(f"Reached maximum {args.max_structures} filtered structures. Stopping.")
        break

    sig = structure_signature(atoms)

    try:
        desc = calculator.get_descriptors(atoms, invariants_only=False)
    except Exception as e:
        print(f"Skipping structure {i} due to descriptor error: {e}")
        continue

    is_similar = False

    # ---- Compare with reference structures of same signature ----
    if sig in signature_to_ref_desc:
        for ref_desc in signature_to_ref_desc[sig]:
            if np.linalg.norm(desc - ref_desc) < args.threshold:
                is_similar = True
                break

    # ---- Compare with previously filtered new structures ----
    if not is_similar:
        for j, (sig_existing, existing_desc) in enumerate(zip(filtered_signatures, filtered_descriptors)):
            if sig_existing != sig:
                continue  # skip different compositions

            dist = np.linalg.norm(desc - existing_desc)
            distance_matrix[len(filtered_descriptors)][j] = dist
            distance_matrix[j][len(filtered_descriptors)] = dist

            if dist < args.threshold:
                is_similar = True
                break

    if not is_similar:
        filtered_structures.append(atoms)
        filtered_descriptors.append(desc)
        filtered_signatures.append(sig)


# -----------------------
# REQUIREMENT: Filter count check based on NEW structures only
# -----------------------
if len(filtered_structures) < args.min_new_structures:
    print(f"Only {len(filtered_structures)} filtered new structures "
          f"(required {args.min_new_structures}). Requesting more MTD sampling...")
    exit(10)


# -----------------------
# Save heatmap (optional)
# -----------------------
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

# -----------------------
# Save output
# -----------------------
if filtered_structures:
    write(args.output, filtered_structures, format='extxyz')
    print(f"Saved {len(filtered_structures)} filtered structures to {args.output}")
else:
    print("No new unique structures found.")

