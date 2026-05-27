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
    frames = read(filename, index=":")  
    forces_all = [atoms.get_forces() for atoms in frames]
    return forces_all

# === Read trajectories ===
dft_forces  = read_forces_from_extxyz(args.dft_file)
mlip_forces = read_forces_from_extxyz(args.mlip_file)

if len(dft_forces) != len(mlip_forces):
    raise ValueError("Number of frames does not match")

all_errors = []

# === DEBUG print statements ===
for i, (F_dft, F_mlip) in enumerate(zip(dft_forces, mlip_forces)):

    diff = F_mlip - F_dft
    err = np.linalg.norm(diff, axis=1)

    print("F_dft[216] =", F_dft[216])
    print("F_mlip[216] =", F_mlip[216])
    print("diff[216] =", diff[216])
    print("err[216] =", err[216])

    break
# ==============================

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
all_errors = np.concatenate(all_errors)  

print("\n===== GLOBAL ERROR METRICS =====")
print(f"RMSE: {np.sqrt((all_errors**2).mean()):.6f} eV/Å")
print(f"MAE : {all_errors.mean():.6f} eV/Å")
print(f"Max : {all_errors.max():.6f} eV/Å")

write("mlip_with_force_error.extxyz", mlip_images, format="extxyz")
print("Saved mlip_with_force_error.extxyz")
