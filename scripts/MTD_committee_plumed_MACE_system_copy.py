from ase.calculators.plumed import Plumed
from ase import units
from ase import Atoms
from ase.io import read, write
from mace.calculators import mace_mp, MACECalculator
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary, ZeroRotation)
from ase.visualize import view
from sys import argv
import numpy as np
import os
import matplotlib.pyplot as plt

# === PARAMETERS ===
timestep = 0.001
ps = 1000 * units.fs
height = 4.0
pace = 400
sigma1 = 0.1
sigma2 = 0.2
stride = 10
interval = 5
biasfactor = 5
nsteps = 2500
variance_limit = 0.0015
temperature = 400
kT = temperature * units.kB

# === Lists to store results ===
time_fs = []
temperature = []
energies_1 = []
energies_2 = []
energies_3 = []
variances = []
committee_energies = []
frames_with_variance = []

# === INPUT FILE ===
input_file = argv[1]
atoms = read(input_file)
save_file = os.path.splitext(argv[1])[0]

# === Create PLUMED input string ===
plumed_input = [f"UNITS LENGTH=A TIME={1/ps} ENERGY={units.mol/units.kJ}",
"c1: COORDINATION GROUPA=217 GROUPB=219-221 R_0=2.2",
"c2: COORDINATION GROUPA=217 GROUPB=39,56,57,58,59,60,61,62,63,64,79,80,81,85,87,89,90,92 R_0=2.0",
"LOWER_WALLS ARG=c2 AT=0.3 KAPPA=100. LABEL=d1",
f"metad: METAD ARG=c1,c2 HEIGHT={height} PACE={pace} " +
f"SIGMA={sigma1},{sigma2} GRID_MIN=0.0,0.0 GRID_MAX=5.0,5.0 " +
f"BIASFACTOR={biasfactor} TEMP=400 FILE=HILLS",
f"PRINT ARG=c1,c2,metad.bias STRIDE={stride} FILE=COLVAR"]

# === Model paths ====
model_paths = ['/scratch/project_462000838/mace/metadynamics/TDMAS/MACE_models/MACE_model_03_finetuned.model',
               '/scratch/project_462000838/mace/metadynamics/TDMAS/MACE_models/MACE_model_04_finetuned.model',
               '/scratch/project_462000838/mace/metadynamics/TDMAS/MACE_models/MACE_model_05_finetuned.model']

#mace_single = MACECalculator(model_paths=model_paths[0], device='cuda', default_dtype='float64')
mace_committee = MACECalculator(model_paths=model_paths, device='cuda', default_dtype='float64')

# === Initialize PLUMED ===
atoms.calc = Plumed(calc=mace_committee,
                    input=plumed_input,
                    timestep=timestep,
                    atoms=atoms,
                    kT=kT)

# Set the momenta corresponding to T=400K, (0.03 kBT)
MaxwellBoltzmannDistribution(atoms, temperature_K=400) #0.03/units.kB

# Set up the Molecular dynamics run (VV for NVE ensemble)
dyn = VelocityVerlet(atoms, timestep=1.0 * units.fs)


fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex='all', gridspec_kw={'hspace': 0, 'wspace': 0})

def write_frame():
        atoms_copy = atoms.copy()
        atoms_copy.calc = mace_committee
        energy = atoms_copy.get_potential_energy()
        dyn.atoms.write('MACE_MTD_committee_system.xyz', append=True, write_results=False)
        timestep_fs = dyn.get_time()/units.fs
        atoms_copy.info['timestep'] = timestep_fs
        time_fs.append(timestep_fs)
        temperature.append(dyn.atoms.get_temperature())
        energies_1.append(atoms_copy.calc.results['energies'][0]/len(dyn.atoms))
        energies_2.append(atoms_copy.calc.results['energies'][1]/len(dyn.atoms))
        energies_3.append(atoms_copy.calc.results['energies'][2]/len(dyn.atoms))
        variance = atoms_copy.calc.results['energy_var']
        variances.append(variance)
        committee_energies.append(atoms_copy.calc.results['energy']/len(dyn.atoms))
        if variance is not None and variance >= variance_limit:
            #atoms_copy = atoms.copy()
            atoms_copy.info['variance'] = variance
            #atoms_copy.info['energy'] = energy
            #print(atoms_copy.info)
            frames_with_variance.append((variance, atoms_copy))

        # plot variance threshold for DFT recalculation
        ax[0].axhline(y=variance_limit, color='r', linestyle=':')

        # plot committee variance
        ax[0].plot(np.array(time_fs), np.array(variances), color="y")
        ax[0].set_ylabel(r'committee variance')
        ax[0].legend(['Error threshold for recalculation', 'Estimated Error (committee variances)'], loc='upper left')

        # plot the temperature of the system as subplots
        ax[1].plot(np.array(time_fs), temperature, color="r", label='Temperature')
        ax[1].set_ylabel("T (K)")

        ax[2].plot(np.array(time_fs), energies_1, color="g")
        ax[2].plot(np.array(time_fs), energies_2, color="y")
        ax[2].plot(np.array(time_fs), energies_3, color="olive")
        ax[2].plot(np.array(time_fs), committee_energies, color="black")
        ax[2].set_ylabel("E (eV/atom)")
        ax[2].set_xlabel('Time (fs)')
        ax[2].legend(['E mace1', 'E mace2', 'E mace3', 'E committee_avg'], loc='upper left')

dyn.attach(write_frame, interval=interval)
dyn.run(nsteps)

# Sort frames by decresing variance and save for reference calculations
sorted_frames = [atoms for variance, atoms in sorted(frames_with_variance, key=lambda x: x[0], reverse=True)]
#print(sorted_frames)
write('frames_for_DFT_eval.xyz', sorted_frames, format='extxyz', write_results=False)

plt.tight_layout()
plt.savefig('mace_mtd_committee_analysis.png', dpi=300)
