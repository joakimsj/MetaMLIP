import os
from ase.io import read
from ase import units
from ase.constraints import FixAtoms
from mace.calculators import mace_mp, MACECalculator
from ase.mep import NEB
from ase.optimize.fire import FIRE as QuasiNewton # consider BGFS
from sys import argv

# Read initial and final geometries
initial = read(argv[1])
final = read(argv[2])
z_threshold = 2.0
initial_name = os.path.splitext(argv[1])[0]
final_name = os.path.splitext(argv[2])[0]

models='/scratch/project_2012660/MACE_active_learning/AL_TDMAS_one_mol/results/reTrainMACE/seed_123/MACE_model_seed_123.model'

# Define calculators for initial/final states
initial_calc = MACECalculator(model_paths=models, device='cuda', enablecueq=True, default_dtype='float64')

# Attach calculator and compute forces for initial state
initial.calc = initial_calc

fixed_indices = [i for i, atom in enumerate(initial) if atom.position[2] < z_threshold]
fix_constraint = FixAtoms(indices=fixed_indices)
initial.set_constraint(fix_constraint)

relax = QuasiNewton(initial)
relax.run(fmax=0.05)
#initial_energy = initial.get_potential_energy()
#initial_forces = initial.get_forces()

final_calc = MACECalculator(model_paths=models, device='cuda', enablecueq=True, default_dtype='float64')

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
    image.calc = MACECalculator(model_paths=models, device='cuda', enablecueq=True, default_dtype='float64')

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
qn = QuasiNewton(neb, trajectory=f'{initial_name}_NEB_idpp.traj', logfile=f'{initial_name}_NEB_idpp.log')
qn.run(fmax=0.05)

