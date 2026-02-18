import argparse
import os
import numpy as np
from ase.io import read, write
from ase.constraints import FixAtoms
from ase.neighborlist import NeighborList
from ase.mep import NEB
from ase.optimize.fire import FIRE as QuasiNewton
from mace.calculators import MACECalculator

# === Argument parser ===
parser = argparse.ArgumentParser(description="Run NEB with MACE committee and evaluate path uncertainty")

parser.add_argument("--initial_file", type=str, required=True)
parser.add_argument("--final_file", type=str, required=True)
parser.add_argument("--model_paths", type=str, nargs='+', required=True)
parser.add_argument("--z_threshold", type=float, default=2.0)
parser.add_argument("--n_images", type=int, default=8)
parser.add_argument("--n_images_eval", type=int, default=10)

args = parser.parse_args()

initial_name = os.path.splitext(os.path.basename(args.initial_file))[0]

# === Read structures ===
initial = read(args.initial_file)
final = read(args.final_file)

# === MACE Committee ===
mace_committee = MACECalculator(
    model_paths=args.model_paths,
    device='cuda',
    default_dtype='float64',
    head='default'
)

# === Fix slab atoms ===
def fix_slab(atoms):
    fixed_indices = [i for i, atom in enumerate(atoms)
                     if atom.position[2] < args.z_threshold]
    atoms.set_constraint(FixAtoms(indices=fixed_indices))

fix_slab(initial)
fix_slab(final)

# === Relax endpoints ===
initial.calc = mace_committee
final.calc = mace_committee

QuasiNewton(initial).run(fmax=0.05)
QuasiNewton(final).run(fmax=0.05)

# === Build NEB images ===
def make_calc():
    return MACECalculator(
        model_paths=args.model_paths,
        device='cuda',
        default_dtype='float64',
        head='default'
    )

images = [initial]
for _ in range(args.n_images):
    images.append(initial.copy())
images.append(final)

for img in images:
    fix_slab(img)
    img.calc = make_calc()

# === NEB ===
neb = NEB(images)
neb.interpolate('idpp')

trajfile = f"{initial_name}_neb_committee.traj"
logfile = f"{initial_name}_neb_committee.log"

qn = QuasiNewton(neb, trajectory=trajfile, logfile=logfile)
qn.run(fmax=0.05)

# === Uncertainty computation ===
def compute_force_uncertainty(atoms, mace_committee, cutoff=5.0):
    """
    Returns:
        force_sigma : (n_atoms,)   magnitude-based atomic uncertainty
        s_local     : (n_atoms,)   locally averaged uncertainty
    """

    atoms.calc = mace_committee
    _ = atoms.get_forces()  # ensure results are populated

    # --- per-model forces ---
    forces_comm = atoms.calc.results["forces_comm"]
    # shape: (n_models, n_atoms, 3)

    # --- per-atom, per-component std ---
    force_std = forces_comm.std(axis=0)     # (n_atoms, 3)

    # --- atomic uncertainty measures ---
    s_atom = force_std.mean(axis=1)          # component-averaged
    force_sigma = np.linalg.norm(force_std, axis=1)  # magnitude-based

    # --- neighbor list for local aggregation ---
    cutoffs = [cutoff/2] * len(atoms)
    nl = NeighborList(cutoffs, self_interaction=True, bothways=True)
    nl.update(atoms)

    s_local = np.zeros(len(atoms))
    for i in range(len(atoms)):
        indices, _ = nl.get_neighbors(i)
        s_local[i] = s_atom[indices].mean()

    return force_sigma, s_local

# === Evaluate U_path ===
images = read(trajfile, f"-{n_images_eval}:")

images_out = []
U_path = []

for img in images:
    force_sigma, s_local = compute_force_uncertainty(img, mace_committee)

    # store per-atom fields for OVITO
    img.arrays["per_atom_force_uncertainty"] = force_sigma
    img.arrays["local_force_uncertainty"] = s_local

    # store per-image diagnostics
    img.info["mean_local_uncertainty"] = float(s_local.mean())
    img.info["max_local_uncertainty"] = float(s_local.max())

    U_path.append(s_local.mean())
    images_out.append(img)

U_path = np.array(U_path)

print("\n===== NEB Path Uncertainty =====")
print("U_path =", U_path)
print("Integrated path uncertainty =", U_path.mean())
print("Max path uncertainty =", U_path.max())

# Save uncertainty profile
np.savetxt(f"{initial_name}_U_path.dat", U_path)

# Save annotated trajectory for OVITO
write(f"{initial_name}_neb_committee_uncertainty.xyz",
      images_out,
      format="extxyz",
      write_results=False)

# === Harvest high-uncertainty frames for DFT ===

# Sort images by max local uncertainty (descending)
harvest = sorted(
    images_out,
    key=lambda img: img.info["max_local_uncertainty"],
    reverse=True
)

# Choose top-k frames
top_k = min(5, len(harvest))   # <-- tune this
harvest = harvest[:top_k]

# Save DFT training structures
dft_xyz = f"{initial_name}_neb_dft_harvest.xyz"

write(
    dft_xyz,
    harvest,
    format="extxyz",
    write_results=False
)

print(f"\nSaved {len(harvest)} high-uncertainty NEB frames for DFT → {dft_xyz}")
print("Max uncertainties:", [img.info["max_local_uncertainty"] for img in harvest])

      
