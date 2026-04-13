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
parser.add_argument("--max_steps", type=int, default=1500)

args = parser.parse_args()

use_committee = len(args.model_paths) > 1

if use_committee:
    print(f"Using committee of {len(args.model_paths)} models")
else:
    print("Single model NEB detected → skipping uncertainty estimation")

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

# === Fix slab atoms ===
def fix_slab(atoms):
    fixed_indices = [i for i, atom in enumerate(atoms)
                     if atom.position[2] < args.z_threshold]
    atoms.set_constraint(FixAtoms(indices=fixed_indices))

fix_slab(initial)
fix_slab(final)

# === Detect NEB band size ===
def detect_n_images(images, tol=1e-3):
    ref = images[0].get_positions()
    for i in range(1, len(images)):
        if np.allclose(images[i].get_positions(), ref, atol=tol):
            return i
    raise RuntimeError("Could not detect NEB band size")

# === Build NEB images ===
trajfile = f"{initial_name}_neb_committee.traj"
logfile  = f"{initial_name}_neb_committee.log"

restart = os.path.exists(trajfile)

# === Relax endpoints ===
if not restart:
    print("No existing NEB trajectory → relaxing endpoints")
    initial.calc = make_calc()
    final.calc   = make_calc()
    QuasiNewton(initial).run(fmax=0.05)
    QuasiNewton(final).run(fmax=0.05)
else:
    print("Restart detected → skipping endpoint relaxation")

# === Build or load images ===
if restart:
    print(f"Loading previous NEB state from {trajfile}")
    try:
        all_images = read(trajfile, ":")

        n_images_detected = detect_n_images(all_images)
        print(f"Detected {n_images_detected} images per NEB iteration")

        images = all_images[-n_images_detected:]

        for img in images:
            fix_slab(img)
            img.calc = make_calc()
            img.get_potential_energy()  # ensure energy exists

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

    # FORCE energy evaluation 
    print("Evaluating initial energies for all images...")
    for i, img in enumerate(images):
        e = img.get_potential_energy()
        print(f"Image {i}: {e:.6f} eV")

    neb = NEB(images, climb=True)
    neb.interpolate('idpp')

else:
    neb = NEB(images, climb=True)

# === Run NEB ===
neb_failed = False

try:
    opt = QuasiNewton(neb, trajectory=trajfile, logfile=logfile)
    opt.run(fmax=0.05, steps=args.max_steps)
except Exception as e:
    print("\n!!! NEB optimization failed !!!")
    print(str(e))
    neb_failed = True

# === Extract FINAL converged band ===
if os.path.exists(trajfile):
    all_images = read(trajfile, ":")
    n_images_detected = detect_n_images(all_images)
    print(f"Extracting final NEB band ({n_images_detected} images)")
    images_eval = all_images[-n_images_detected:]
else:
    print("No trajectory found — using current NEB images")
    images_eval = neb.images

# === Uncertainty computation ===
def compute_force_uncertainty(atoms, cutoff=5.0):
    atoms.calc = make_calc()
    _ = atoms.get_forces()

    if "forces_comm" not in atoms.calc.results:
        raise RuntimeError("No committee forces available")

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


# === Evaluate uncertainty OR fallback ===
images_out = []
U_path = []

if use_committee:
    print("Evaluating uncertainty with committee...")

    for img in images_eval:
        try:
            force_sigma, s_local = compute_force_uncertainty(img)
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

else:
    print("Skipping uncertainty evaluation")

    # Just pass through NEB images
    images_out = images_eval

    # Optional: still write a dummy file for compatibility
    np.savetxt(f"{initial_name}_U_path.dat",
               np.zeros(len(images_out)),
               header="dummy uncertainty (single model)")

write(f"{initial_name}_neb_committee_uncertainty.xyz",
      images_out,
      format="extxyz",
      write_results=False)

# === Harvest frames ===
if use_committee:
    harvest = sorted(
        images_out,
        key=lambda img: img.info.get("max_local_uncertainty", 0.0),
        reverse=True
    )
else:
    # No uncertainty → just take central images (most interesting region)
    mid = len(images_out) // 2
    harvest = images_out[max(0, mid-2):mid+3]

harvest = harvest[:min(5, len(harvest))]

if len(harvest) == 0:
    print("WARNING: Empty harvest — dumping fallback frames")
    harvest = images_out[:min(3, len(images_out))]


write(f"{initial_name}_neb_dft_harvest.xyz", harvest,
      format="extxyz", write_results=False)

print(f"\nSaved {len(harvest)} frames")
print("Max uncertainties:", [img.info.get("max_local_uncertainty", 0.0) for img in harvest])
