import argparse
from ase.calculators.plumed import Plumed
from ase import units
from ase.io import read, write
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
parser.add_argument("--nsteps", type=int, default=2500, help="Number of MD steps")
parser.add_argument("--temperature", type=float, default=400, help="Temperature in Kelvin")
parser.add_argument("--pace", type=int, default=400, help="METAD PACE")
parser.add_argument("--height", type=float, default=4.0, help="METAD height")
parser.add_argument("--sigma1", type=float, default=0.1, help="METAD sigma1")
parser.add_argument("--sigma2", type=float, default=0.2, help="METAD sigma2")
parser.add_argument("--biasfactor", type=float, default=5, help="METAD bias factor")
parser.add_argument("--stride", type=int, default=10, help="PLUMED print stride")
parser.add_argument("--interval", type=int, default=5, help="ASE attach interval")
parser.add_argument("--variance_limit", type=float, default=0.0015, help="Variance threshold")
parser.add_argument("--c1_threshold", type=float, default=1.5, help="Threshold for CV c1")
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

    if variance is not None and variance >= args.variance_limit:
        atoms_copy.info['variance'] = variance
        frames_with_variance.append((variance, atoms_copy))

    # === Plot updates ===
    ax[0].axhline(y=args.variance_limit, color='r', linestyle=':')
    ax[0].plot(time_fs, variances, color="y")
    ax[0].set_ylabel("Variance")
    ax[0].legend(["Threshold", "Estimated Variance"])

    ax[1].plot(time_fs, temperatures, color="r")
    ax[1].set_ylabel("T (K)")

    for i, e_list in enumerate(energies_all):
    #ax[2].plot(time_fs, e_list, label=f"E mace{i+1}")
    ax[2].plot(time_fs, committee_energies, color="black", label="E committee")
    ax[2].set_ylabel("E (eV/atom)")
    ax[2].set_xlabel("Time (fs)")
    ax[2].legend(loc='upper left')

dyn.attach(write_frame, interval=args.interval)

# === Run dynamics with clean stopping ===
try:
    dyn.run(args.nsteps)
except StopMD:
    print("Simulation stopped early by CV or variance threshold.")

# === Ensure at least the last frame is saved ===
if not frames_with_variance:
    last_frame = atoms.copy()
    last_frame.calc = mace_committee
    last_frame.info['variance'] = None  # no variance exceeded
    frames_with_variance.append((None, last_frame))

# === Output filtered frames ===
sorted_frames = [atoms for _, atoms in sorted(frames_with_variance, key=lambda x: (x[0] if x[0] is not None else -1), reverse=True)]
write('frames_for_DFT_eval.xyz', sorted_frames, format='extxyz', write_results=False)

plt.tight_layout()
plt.savefig('mace_mtd_committee_analysis.png', dpi=300)
