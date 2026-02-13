import os
import numpy as np
from ase.io import read, write
from ase import units
from ase.constraints import FixAtoms
from mace.calculators import mace_mp, MACECalculator
from ase.mep import NEB
from ase.optimize.fire import FIRE as QuasiNewton # consider BGFS
from ase.vibrations import Vibrations
from sys import argv

# Read initial and final geometries
initial = read(argv[1])
final = read(argv[2])
z_threshold = 2.0
initial_name = os.path.splitext(argv[1])[0]
final_name = os.path.splitext(argv[2])[0]

#models='./MACE_models/MACE_model_seed_123.model'
#models='./MACE_models/mace-mpa-0-medium.model'
models='./MACE_models/MACE_model_seed_123_force_var.model'

# Define calculators for initial/final states
initial_calc = MACECalculator(model_paths=models, device='cuda', default_dtype='float64') #  enablecueq=True on NVIDIA GPUs

# Attach calculator and compute forces for initial state
initial.calc = initial_calc

fixed_indices = [i for i, atom in enumerate(initial) if atom.position[2] < z_threshold]
fix_constraint = FixAtoms(indices=fixed_indices)
initial.set_constraint(fix_constraint)

relax = QuasiNewton(initial)
relax.run(fmax=0.05)
#initial_energy = initial.get_potential_energy()
#initial_forces = initial.get_forces()

final_calc = MACECalculator(model_paths=models, device='cuda', default_dtype='float64') #  enablecueq=True

# Attach calculator and compute forces for final state
final.calc = final_calc

fixed_indices = [i for i, atom in enumerate(final) if atom.position[2] < z_threshold]
fix_constraint = FixAtoms(indices=fixed_indices)
final.set_constraint(fix_constraint)

relax = QuasiNewton(final)
relax.run(fmax=0.05)

#final_energy = final.get_potential_energy()
#final_forces = final.get_forces()

# Create intermediate images (including initial and final)
images = [initial]

for i in range(8):
    images.append(initial.copy())

# Define calculator for all images
for image in images[1:]:
    image.calc = MACECalculator(model_paths=models, device='cuda', default_dtype='float64') #  enablecueq=True

# Append final image
images.append(final)

# Constrain substrate atoms of intermediate images
fixed_indices = [i for i, atom in enumerate(image) if atom.position[2] < z_threshold]
fix_constraint = FixAtoms(indices=fixed_indices)
for image in images:
    image.set_constraint(fix_constraint)

# Run IDPP interpolation
neb = NEB(images)
neb.interpolate('idpp')

# Run NEB optimization
qn = QuasiNewton(neb, trajectory=f'{initial_name}_idpp_NEB_f_var.traj', logfile=f'{initial_name}_idpp_NEB_f_var.log')
qn.run(fmax=0.05)

# Run vibrational analysis on the NEB-determined TS geometry
neb_traj = f"{initial_name}_idpp_NEB_f_var.traj"

images = read(neb_traj, ":")
neb_mid = images[-9:-1]

energies = []
for img in neb_mid:
    if "energy" in img.calc.results:
        energies.append(img.calc.results["energy"])
    else:
        raise RuntimeError("Energy missing in trajectory.")

imax = np.argmax(energies)
ts = neb_mid[imax]
ts.calc = MACECalculator(model_paths=models, device='cuda', default_dtype='float64') #  enablecueq=True

print(f"TS image index: {imax}, energy = {energies[imax]:.6f}")
write("TS_guess.traj", ts)

reactive_indices = list(range(len(ts)))  # consider subset in the region where reaction happens

vib = Vibrations(ts, indices=reactive_indices)
vib.run()
vib.summary(log='vibrations.txt')
vib.write_mode(0)

#energies = vib.get_energies()
#print("\nVibrational energies (eV):")
#print(energies)

print("\nImaginary frequencies present:",
      np.any(np.imag(energies) > 0))



