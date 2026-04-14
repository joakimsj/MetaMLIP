import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("error_file")
parser.add_argument("unc_file")
args = parser.parse_args()

# === Read trajectories ===
err_images = read(args.error_file, ":")
unc_images = read(args.unc_file, ":")

if len(err_images) != len(unc_images):
    raise ValueError("Mismatch in number of frames")

all_err = []
all_unc = []
all_loc = []

# === Collect data ===
for i, (img_err, img_unc) in enumerate(zip(err_images, unc_images)):

    if "force_error" not in img_err.arrays:
        continue

    if "per_atom_force_uncertainty" not in img_unc.arrays:
        continue

    err = img_err.arrays["force_error"]
    unc = img_unc.arrays["per_atom_force_uncertainty"]
    loc = img_unc.arrays["local_force_uncertainty"]

    if len(err) != len(unc):
        raise ValueError(f"Atom mismatch in frame {i}")

    all_err.append(err)
    all_unc.append(unc)
    all_loc.append(loc)

# Flatten everything
all_err = np.concatenate(all_err)
all_unc = np.concatenate(all_unc)
all_loc = np.concatenate(all_loc)

# === Correlations ===
def corr(x, y):
    return np.corrcoef(x, y)[0, 1]

print("\n=== Correlations ===")
print(f"Error vs per-atom uncertainty : {corr(all_err, all_unc):.3f}")
print(f"Error vs local uncertainty    : {corr(all_err, all_loc):.3f}")

# === Plotting ===
plt.figure()
plt.scatter(all_unc, all_err, s=5, alpha=0.3)
plt.xlabel("Per-atom uncertainty")
plt.ylabel("Force error (eV/Å)")
plt.title("Error vs Per-atom Uncertainty")

plt.figure()
plt.scatter(all_loc, all_err, s=5, alpha=0.3)
plt.xlabel("Local uncertainty")
plt.ylabel("Force error (eV/Å)")
plt.title("Error vs Local Uncertainty")

plt.figure()
plt.scatter(all_unc, all_loc, s=5, alpha=0.3)
plt.xlabel("Per-atom uncertainty")
plt.ylabel("Local uncertainty")
plt.title("Uncertainty vs Local Uncertainty")

plt.show()

# === Per-structure correlation ===

struct_unc = []
struct_err = []

for img_err, img_unc in zip(err_images, unc_images):

    # Extract from info dict
    if "force_rmse" not in img_err.info:
        continue
    if "mean_local_uncertainty" not in img_unc.info:
        continue

    struct_err.append(img_err.info["force_rmse"])
    struct_unc.append(img_unc.info["mean_local_uncertainty"])

struct_err = np.array(struct_err)
struct_unc = np.array(struct_unc)

# === Correlation ===
corr_struct = np.corrcoef(struct_unc, struct_err)[0, 1]

print("\n=== Structure-level correlation ===")
print(f"RMSE vs mean local uncertainty: {corr_struct:.3f}")

# === Plot ===
plt.figure()
plt.scatter(struct_unc, struct_err, s=40)

for i, (u, e) in enumerate(zip(struct_unc, struct_err)):
    plt.text(u, e, str(i), fontsize=8)  # label frames (optional)

plt.xlabel("Mean local uncertainty")
plt.ylabel("Force RMSE (eV/Å)")
plt.title("Structure-level correlation")

plt.show()
