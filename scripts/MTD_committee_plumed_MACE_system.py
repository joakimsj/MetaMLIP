import argparse
from ase.neighborlist import NeighborList
from ase.calculators.plumed import Plumed
from ase import units
from ase.io import read, write
from ase.constraints import FixAtoms
from mace.calculators import MACECalculator
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import matplotlib.pyplot as plt
import numpy as np
import os

# === Custom exception to stop MD cleanly ===
class StopMD(Exception):
    pass

# === Argument parser ===
parser = argparse.ArgumentParser(description="Run MTD with MACE committee and PLUMED")

parser.add_argument("--input_file", type=str, help="Initial structure file (.traj)")
parser.add_argument("--model_paths", type=str, nargs='+', required=True, help="List of trained MACE model paths")
parser.add_argument("--timestep", type=float, default=1.0, help="MD timestep in fs")
parser.add_argument("--z_threshold", type=float, default=2.0, help="z-threshold in Å for fixing slab atoms")
parser.add_argument("--nsteps", type=int, default=2500, help="Number of MD steps")
parser.add_argument("--temperature", type=float, default=400, help="Temperature in Kelvin")
parser.add_argument("--pace", type=int, default=400, help="METAD PACE")
parser.add_argument("--height", type=float, default=4.0, help="METAD height")
parser.add_argument("--sigma1", type=float, default=0.1, help="METAD sigma1")
parser.add_argument("--sigma2", type=float, default=0.2, help="METAD sigma2")
parser.add_argument("--biasfactor", type=float, default=5, help="METAD bias factor")
parser.add_argument("--stride", type=int, default=10, help="PLUMED print stride")
parser.add_argument("--interval", type=int, default=5, help="ASE attach interval")
parser.add_argument("--variance_limit", type=float, default=0.500, help="Variance threshold")
parser.add_argument("--force_variance_limit", type=float, default=0.0200, help="Per-atom force variance threshold")
parser.add_argument("--c1_threshold", type=float, default=2.0, help="Threshold for CV c1")
parser.add_argument("--c2_threshold", type=float, default=2.5, help="Threshold for CV c2")

args = parser.parse_args()

# === Derived ===
kT = args.temperature * units.kB
atoms = read(args.input_file)

# === MACE Committee ===
mace_committee = MACECalculator(model_paths=args.model_paths, device='cuda', default_dtype='float64', head='default')

# === PLUMED input string ===
plumed_input = [
    f"UNITS LENGTH=A TIME={1/(1000*units.fs)} ENERGY={units.mol/units.kJ}",
    "c1: COORDINATION GROUPA=217 GROUPB=219-221 R_0=2.2",
    "c2: COORDINATION GROUPA=217 GROUPB=39,56,57,58,59,60,61,62,63,64,79,80,81,85,87,89,90,92 R_0=2.0",
    "LOWER_WALLS ARG=c2 AT=0.3 KAPPA=100 LABEL=d1",
    f"metad: METAD ARG=c1,c2 HEIGHT={args.height} PACE={args.pace} " +
    f"SIGMA={args.sigma1},{args.sigma2} GRID_MIN=0.0,0.0 GRID_MAX=5.0,5.0 " +
    f"BIASFACTOR={args.biasfactor} TEMP={args.temperature} FILE=HILLS",
    f"PRINT ARG=c1,c2,metad.bias STRIDE={args.stride} FILE=COLVAR",
    f"FLUSH STRIDE=1"
]

# === Setup calc ===
atoms.calc = Plumed(calc=mace_committee, input=plumed_input, timestep=args.timestep, atoms=atoms, kT=kT)
z_threshold = args.z_threshold
print(z_threshold)
fixed_indices = [i for i, atom in enumerate(atoms) if atom.position[2] < z_threshold]
print(fixed_indices)
fix_constraint = FixAtoms(indices=fixed_indices)
atoms.set_constraint(fix_constraint)
MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)
dyn = VelocityVerlet(atoms, timestep=args.timestep * units.fs)

# === Monitoring and output ===
time_fs = []
temperatures = []
energies_all = [[] for _ in range(len(args.model_paths))]
variances = []
committee_energies = []
frames_with_variance = []

fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex='all', gridspec_kw={'hspace': 0})

def read_last_colvar(filename="COLVAR"):
    """Read the last line of COLVAR to get c1 and c2."""
    with open(filename, "r") as f:
        lines = [l for l in f if not l.startswith("#")]
        if not lines:
            return None, None
        last_line = lines[-1].split()
        c1 = float(last_line[1])  # adjust indices if needed
        c2 = float(last_line[2])
        return c1, c2

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


def write_frame():
    atoms_copy = atoms.copy()
    atoms_copy.calc = mace_committee

    # === Read CVs from COLVAR ===
    c1, c2 = read_last_colvar()
    if (c1 is not None and c1 < args.c1_threshold) or (c2 is not None and c2 > args.c2_threshold):
        print(f"Stopping simulation: c1={c1}, c2={c2}")
        raise StopMD

    # === Logging and saving ===
    dyn.atoms.write('MACE_MTD_committee_system.xyz', append=True, write_results=False)

    t_fs = dyn.get_time() / units.fs
    time_fs.append(t_fs)
    temperatures.append(dyn.atoms.get_temperature())

    for i, e in enumerate(atoms_copy.calc.results['energies']):
        energies_all[i].append(e / len(dyn.atoms))
    committee_energies.append(atoms_copy.calc.results['energy'] / len(dyn.atoms))

    variance = atoms_copy.calc.results['energy_var']
    variances.append(variance)

    # === Per-atom force uncertainty ===
    force_sigma, s_local = compute_force_uncertainty(atoms_copy, mace_committee)

    # Store per-atom property for OVITO
    atoms_copy.arrays["per_atom_force_uncertainty"] = force_sigma
    atoms_copy.arrays["local_force_uncertainty"] = s_local
    #print(force_sigma)

    # Reduce to structure-level diagnostics
    local_mean = s_local.mean()
    #local_p95 = np.percentile(s_local, 95)
    local_max = s_local.max()
    max_force_sigma = np.max(force_sigma)
    mean_force_sigma = np.mean(force_sigma)

    # Store metadata
    atoms_copy.info["energy_variance"] = variance
    atoms_copy.info["max_force_uncertainty"] = max_force_sigma
    atoms_copy.info["mean_force_uncertainty"] = mean_force_sigma
    atoms_copy.info["local_max_force_uncertainty"] = float(local_max)
    atoms_copy.info["local_mean_force_uncertainty"] = float(local_mean)

    # === Selection logic ===
    select = False

    if variance is not None and variance >= args.variance_limit:
        select = True

    if local_max >= args.force_variance_limit:
        select = True

    if select:
        frames_with_variance.append((local_max, atoms_copy))

    # === Save uncertainty and CVs in a separate log file ===
    with open("cv_uncertainty_topo.dat", "a") as f:
        f.write(f"{t_fs:.4f} {c1:.6f} {c2:.6f} "
                f"{variance:.6e} {max_force_sigma:.6e} {mean_force_sigma:.6e} {local_max:.6e}\n")

dyn.attach(write_frame, interval=args.interval)

# Add header to uncertainty-CV log file
with open("cv_uncertainty_topo.dat", "w") as f:
    f.write("# time_fs c1 c2 energy_var max_force_unc mean_force_unc local_max_force_unc\n")

# === Run dynamics with clean stopping ===
try:
    dyn.run(args.nsteps)
except StopMD:
    print("Simulation stopped early by CV or variance threshold.")

# === Plot the data ===
ax[0].axhline(y=args.variance_limit, color='r', linestyle=':')
ax[0].plot(time_fs, variances, color="y")
ax[0].set_ylabel("Variance")
ax[0].legend(["Threshold", "Estimated Variance"])

ax[1].plot(time_fs, temperatures, color="r")
ax[1].set_ylabel("T (K)")

for i, e_list in enumerate(energies_all):
    ax[2].plot(time_fs, e_list, label=f"E mace{i+1}")
ax[2].plot(time_fs, committee_energies, color="black", label="E committee")
ax[2].set_ylabel("E (eV/atom)")
ax[2].set_xlabel("Time (fs)")
ax[2].legend(loc='upper left')

# === Ensure at least the last frame is saved ===
if not frames_with_variance:
    last_frame = atoms.copy()
    last_frame.calc = mace_committee

    # --- compute uncertainties ---
    force_sigma, s_local = compute_force_uncertainty(last_frame, mace_committee)

    variance = last_frame.calc.results.get("energy_var", None)

    # --- store per-atom ---
    last_frame.arrays["per_atom_force_uncertainty"] = force_sigma
    last_frame.arrays["local_force_uncertainty"] = s_local

    # --- store per-structure ---
    last_frame.info["energy_variance"] = variance
    last_frame.info["max_force_uncertainty"] = float(np.max(force_sigma))
    last_frame.info["mean_force_uncertainty"] = float(np.mean(force_sigma))

    local_max = float(np.max(s_local))
    last_frame.info["local_max_force_uncertainty"] = local_max

    frames_with_variance.append(
        (local_max, last_frame)
    )

# === Output filtered frames ===
sorted_frames = [atoms for _, atoms in sorted(frames_with_variance, key=lambda x: (x[0] if x[0] is not None else -1), reverse=True)]
write('frames_for_DFT_eval.xyz', sorted_frames, format='extxyz', write_results=False, append=True)

plt.tight_layout()
plt.savefig('mace_mtd_committee_analysis.png', dpi=300)
