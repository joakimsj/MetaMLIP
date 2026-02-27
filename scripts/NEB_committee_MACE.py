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
parser.add_argument("--final_file", type=str)
parser.add_argument("--model_paths", type=str, nargs='+', required=True)
parser.add_argument("--z_threshold", type=float, default=2.0)
parser.add_argument("--n_images", type=int, default=8)
parser.add_argument("--n_images_eval", type=int, default=10)
parser.add_argument("--max_steps", type=int, default=1500)  

args = parser.parse_args()

initial_name = os.path.splitext(os.path.basename(args.initial_file))[0]

# === Read structures ===
if args.final_file is not None:
    initial = read(args.initial_file)
    final = read(args.final_file)
else:
    initial = read(args.initial_file, '0')
    final = read(args.initial_file, '-1')

# === MACE Committee ===
def make_calc():
    return MACECalculator(
        model_paths=args.model_paths,
        device='cuda',
        default_dtype='float64',
        head='default'
    )

mace_committee = make_calc()

# === Fix slab atoms ===
def fix_slab(atoms):
    fixed_indices = [i for i, atom in enumerate(atoms)
                     if atom.position[2] < args.z_threshold]
    atoms.set_constraint(FixAtoms(indices=fixed_indices))

fix_slab(initial)
fix_slab(final)

# === Build NEB images ===
trajfile = f"{initial_name}_neb_committee.traj"
logfile  = f"{initial_name}_neb_committee.log"

restart = os.path.exists(trajfile)

# === Relax endpoints only if fresh run ===
if not restart:
    print("No existing NEB trajectory → relaxing endpoints")
    initial.calc = mace_committee
    final.calc   = mace_committee
    QuasiNewton(initial).run(fmax=0.05)
    QuasiNewton(final).run(fmax=0.05)
else:
    print("Restart detected → skipping endpoint relaxation")

# === Build or load NEB images ===
if restart:
    print(f"Loading previous NEB state from {trajfile}")
    try:
        images = read(trajfile, f"-{args.n_images_eval}:")
        for img in images:
            fix_slab(img)
            img.calc = make_calc()
    except Exception:
        print("Failed to load trajectory — rebuilding NEB band")
        restart = False

if not restart:
    images = [initial]
    for _ in range(args.n_images):
        images.append(initial.copy())
    images.append(final)

    for img in images:
        fix_slab(img)
        img.calc = make_calc()

    neb = NEB(images, climb=True)
    neb.interpolate('idpp')
else:
    neb = NEB(images, climb=True)

# === Run NEB safely ===
neb_failed = False

try:
    opt = QuasiNewton(neb, trajectory=trajfile, logfile=logfile)
    opt.run(fmax=0.05, steps=args.max_steps)
except Exception as e:
    print("\n!!! NEB optimization failed !!!")
    print(str(e))
    neb_failed = True

# === Load trajectory safely ===
if os.path.exists(trajfile):
    images_eval = read(trajfile, f"-{args.n_images_eval}:")
else:
    print("No trajectory found — using interpolated band")
    images_eval = neb.images

# === Uncertainty computation ===
def compute_force_uncertainty(atoms, mace_committee, cutoff=5.0):
    atoms.calc = mace_committee
    _ = atoms.get_forces()

    forces_comm = atoms.calc.results["forces_comm"]
    force_std = forces_comm.std(axis=0)

    s_atom = force_std.mean(axis=1)
    force_sigma = np.linalg.norm(force_std, axis=1)

    cutoffs = [cutoff / 2] * len(atoms)
    nl = NeighborList(cutoffs, self_interaction=True, bothways=True)
    nl.update(atoms)

    s_local = np.zeros(len(atoms))
    for i in range(len(atoms)):
        indices, _ = nl.get_neighbors(i)
        s_local[i] = s_atom[indices].mean()

    return force_sigma, s_local

# === Evaluate uncertainty ===
images_out = []
U_path = []

for img in images_eval:
    try:
        force_sigma, s_local = compute_force_uncertainty(img, mace_committee)
    except Exception:
        continue

    img.arrays["per_atom_force_uncertainty"] = force_sigma
    img.arrays["local_force_uncertainty"] = s_local
    img.info["mean_local_uncertainty"] = float(s_local.mean())
    img.info["max_local_uncertainty"] = float(s_local.max())

    U_path.append(s_local.mean())
    images_out.append(img)

if len(images_out) == 0:
    print("WARNING: No valid frames — falling back to endpoints")
    images_out = [initial, final]

U_path = np.array(U_path)

np.savetxt(f"{initial_name}_U_path.dat", U_path)

write(f"{initial_name}_neb_committee_uncertainty.xyz",
      images_out,
      format="extxyz",
      write_results=False)

# === Harvest frames ===
harvest = sorted(
    images_out,
    key=lambda img: img.info.get("max_local_uncertainty", 0.0),
    reverse=True
)

top_k = min(5, len(harvest))
harvest = harvest[:top_k]

dft_xyz = f"{initial_name}_neb_dft_harvest.xyz"

if len(harvest) == 0:
    print("WARNING: Empty harvest — dumping fallback frames")
    harvest = images_out[:min(3, len(images_out))]

write(dft_xyz, harvest, format="extxyz", write_results=False)

print(f"\nSaved {len(harvest)} frames → {dft_xyz}")
print("Max uncertainties:", [img.info.get("max_local_uncertainty", 0.0) for img in harvest])
