import numpy as np
from ase.io import read, write
import argparse

# === CLI ===
parser = argparse.ArgumentParser(description="Compare DFT and MLIP forces from extxyz files")

parser.add_argument("dft_file", help="DFT reference extxyz")
parser.add_argument("mlip_file", help="MLIP extxyz")

args = parser.parse_args()

dft_images  = read(args.dft_file, ":")
mlip_images = read(args.mlip_file, ":")

def read_forces_from_extxyz(filename):
    forces_all = []

    with open(filename, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        n_atoms = int(lines[i].strip())
        header = lines[i+1]

        # Parse Properties line
        prop_str = header.split("Properties=")[1].split()[0]
        props = prop_str.split(":")

        # Build column map
        columns = []
        col_idx = 0
        j = 0
        while j < len(props):
            name = props[j]
            count = int(props[j+2])
            columns.append((name, col_idx, col_idx + count))
            col_idx += count
            j += 3

        # Find force columns
        force_col = None
        for name, start, end in columns:
            if name.lower() == "forces":
                force_col = (start, end)

        if force_col is None:
            raise RuntimeError("No forces found in file")

        start, end = force_col

        # Read atoms
        frame_forces = []
        for k in range(n_atoms):
            parts = lines[i+2+k].split()
            vals = list(map(float, parts[1:]))  # skip species
            frame_forces.append(vals[start:end])

        forces_all.append(np.array(frame_forces))

        i += n_atoms + 2

    return forces_all

# === Read trajectories ===
dft_forces  = read_forces_from_extxyz(args.dft_file)
mlip_forces = read_forces_from_extxyz(args.mlip_file)

if len(dft_forces) != len(mlip_forces):
    raise ValueError("Number of frames does not match")

all_errors = []

for i, (F_dft, F_mlip) in enumerate(zip(dft_forces, mlip_forces)):

    diff = F_mlip - F_dft
    err = np.linalg.norm(diff, axis=1)

    rmse = np.sqrt((err**2).mean())
    mae  = err.mean()
    maxe = err.max()

    all_errors.append(err)
    mlip_images[i].arrays["force_error"] = err
    mlip_images[i].info["force_rmse"] = float(rmse)
    mlip_images[i].info["force_mae"]  = float(mae)
    mlip_images[i].info["force_max"]  = float(maxe)

    print(f"Frame {i:3d} | RMSE: {rmse:.6f} | MAE: {mae:.6f} | Max: {maxe:.6f}")

# === Global ===
all_errors = np.concatenate(all_errors)  # ✅ FIX

print("\n===== GLOBAL ERROR METRICS =====")
print(f"RMSE: {np.sqrt((all_errors**2).mean()):.6f} eV/Å")
print(f"MAE : {all_errors.mean():.6f} eV/Å")
print(f"Max : {all_errors.max():.6f} eV/Å")

write("mlip_with_force_error.extxyz", mlip_images, format="extxyz")
print("Saved mlip_with_force_error.extxyz")
