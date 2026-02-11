import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from sys import argv

# Files
initial_traj = "ini_pbe_rel_relax_cp2k.traj"
final_traj   = "final_relax_cp2k.traj"
neb_traj     = str(argv[1])

# ---- Read endpoint energies ----

initial = read(initial_traj, index=-1)
final   = read(final_traj, index=-1)

Ei = initial.calc.results['energy']
Ef = final.calc.results['energy']

print(f"Initial energy: {Ei:.8f} eV")
print(f"Final   energy: {Ef:.8f} eV")

# ---- Read NEB intermediate energies ----

neb_images = read(neb_traj, index=":")

def get_energy(img):
    if img.calc is not None and 'energy' in img.calc.results:
        return img.calc.results['energy']
    if 'energy' in img.info:
        return img.info['energy']
    raise RuntimeError("Missing energy")

# Skip endpoints from NEB trajectory
neb_mid = neb_images[-9:-1]

Emid = [get_energy(img) for img in neb_mid]

# ---- Assemble full profile ----

energies = np.array([Ei] + Emid + [Ef])

# Relative energies
energies -= energies[0]

# Reaction coordinate
s = np.linspace(0, 1, len(energies))

np.savetxt("neb_profile.dat", np.c_[s, energies],
           header="reaction_coordinate  energy(eV)")


print("\n===== FINAL NEB PROFILE =====\n")
for i, e in enumerate(energies):
    print(f"Image {i:2d}: {e:12.6f} eV")

print("\nSaved: neb_profile.dat")

def plot_neb_profile(s, energies, outfile="neb_profile.png", title="NEB Energy Profile"):

    plt.figure(figsize=(6,4))

    plt.plot(s, energies, "-o", lw=2.2, ms=6, color="black")

    plt.xlabel("Reaction coordinate", fontsize=12)
    plt.ylabel("Energy (eV)", fontsize=12)
    plt.title(title, fontsize=13)

    plt.tick_params(axis='both', labelsize=11)

    plt.grid(alpha=0.25, linestyle="--")

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()

    print(f"Saved NEB profile plot → {outfile}")

plot_neb_profile(s, energies)
