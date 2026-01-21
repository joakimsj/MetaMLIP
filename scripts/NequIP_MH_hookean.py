from ase import Atoms
from ase.io import read
import numpy as np
from ase.io.aims import read_aims, write_aims
from nequip.ase import NequIPCalculator
from ase.optimize import BFGS
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary, ZeroRotation)
from ase.units import fs
from ase.visualize import view
from ase.constraints import Hookean, FixAtoms
from ase.optimize.minimahopping import MHPlot, MinimaHopping
from sys import argv
import os 

## Command line argument 1: input geometry

# Create an ASE Atoms object with the initial geometry
atoms = read_aims(argv[1])
save_file = os.path.splitext(argv[1])[0] 
#view(atoms)

metal_surface_size = 192  # Specify the number of atoms in the metal surface
molecule_size = 35  # Specify the number of atoms in each adsorbate

# Initialize an empty list to store constraints
constraints = []

    
def get_adsorbate_neighbors(atoms, metal_surface_size, molecule_size, threshold=2.0):
    #Separate metal surface and adsorbates
    metal_surface = atoms[:metal_surface_size]
    adsorbates = atoms[metal_surface_size:]

    # Initialize an empty list to store closest bonded atom pairs for each adsorbate
    closest_bond_indices_per_adsorbate = []

    # Calculate the global index count from the metal atoms
    global_index_offset = metal_surface_size

    # Iterate over each adsorbate
    for i in range(0, len(adsorbates), molecule_size):
        current_adsorbate = adsorbates[i:i+molecule_size]

        # Get the connectivity matrix for the current adsorbate
        connectivity_matrix = current_adsorbate.get_all_distances(mic=True)  # mic=True for minimum image convention

        # Initialize an empty list for the current adsorbate
        closest_bond_indices = []

        # Iterate over each atom in the current adsorbate
        for atom_idx in range(len(current_adsorbate)):
            global_atom_idx = atom_idx + i + global_index_offset  # Calculate the global atom index

            # Find neighbors within the specified threshold
            neighbors_mask = np.where(connectivity_matrix[atom_idx] < threshold)[0]

            # Exclude the atom itself
            neighbors_mask = neighbors_mask[neighbors_mask != atom_idx]

            # Iterate over the neighbors
            for neighbor_idx in neighbors_mask:
                global_neighbor_idx = neighbor_idx + i + global_index_offset  # Calculate the global neighbor index
                distance = connectivity_matrix[atom_idx, neighbor_idx]
                atom_type = current_adsorbate[atom_idx].symbol
                neighbor_type = current_adsorbate[neighbor_idx].symbol

                # Ensure that the pair is not a duplicate
                pair = tuple(sorted([global_atom_idx, global_neighbor_idx]))
                closest_bond_indices.append((pair, atom_type, neighbor_type, distance))

        # Append the closest bond indices for the current adsorbate
        closest_bond_indices_per_adsorbate.append(closest_bond_indices)

    return closest_bond_indices_per_adsorbate

# Get closest bond indices for each adsorbate
closest_bond_indices_per_adsorbate = get_adsorbate_neighbors(atoms, metal_surface_size, molecule_size, threshold=2.0)

# Initialize sets for each bond type to track unique pairs
c_c_bond_indices = set()
c_h_bond_indices = set()
c_o_bond_indices = set()
o_h_bond_indices = set()
c_n_bond_indices = set()
o_n_bond_indices = set()

# Iterate over each adsorbate
for i, closest_bond_indices in enumerate(closest_bond_indices_per_adsorbate):
    # Iterate over each closest bond
    for bond in closest_bond_indices:
        pair = bond[0]
        atom_type, neighbor_type = bond[1], bond[2]

        # Determine bond type and add to the corresponding set
        if atom_type == 'C' and neighbor_type == 'C':
            c_c_bond_indices.add(pair)
        elif atom_type == 'C' and neighbor_type == 'H' or atom_type == 'H' and neighbor_type == 'C':
            c_h_bond_indices.add(pair)
        elif atom_type == 'C' and neighbor_type == 'O' or atom_type == 'O' and neighbor_type == 'C':
            c_o_bond_indices.add(pair)
        elif atom_type == 'O' and neighbor_type == 'H' or atom_type == 'H' and neighbor_type == 'O':
            o_h_bond_indices.add(pair)
        elif atom_type == 'C' and neighbor_type == 'N' or atom_type == 'N' and neighbor_type == 'C':
            c_n_bond_indices.add(pair)
        elif atom_type == 'O' and neighbor_type == 'N' or atom_type == 'N' and neighbor_type == 'O':
            o_n_bond_indices.add(pair)

# Convert sets to lists for further use or printing
c_c_bond_indices = list(c_c_bond_indices)
c_h_bond_indices = list(c_h_bond_indices)
c_o_bond_indices = list(c_o_bond_indices)
o_h_bond_indices = list(o_h_bond_indices)
c_n_bond_indices = list(c_n_bond_indices)
o_n_bond_indices = list(o_n_bond_indices)

# Hookean for C-C bonds
spring_constant_cc = 9.0  # Adjust as needed
for pair in c_c_bond_indices:
    idx1, idx2 = pair
    length = atoms.get_distance(idx1, idx2) + 0.5
    c = Hookean(a1=int(idx1), a2=int(idx2), rt=float(length), k=spring_constant_cc)
    constraints.append(c)

# Hookean for C-H bonds
spring_constant_ch = 7.0  # Adjust as needed
for pair in c_h_bond_indices:
    idx1, idx2 = pair
    length = atoms.get_distance(idx1, idx2) + 0.5
    c = Hookean(a1=int(idx1), a2=int(idx2), rt=float(length), k=spring_constant_cc)
    constraints.append(c)

# Hookean for C-O bonds
spring_constant_co = 6.0  # Adjust as needed
for pair in c_o_bond_indices:
    idx1, idx2 = pair
    length = atoms.get_distance(idx1, idx2) + 0.5
    c = Hookean(a1=int(idx1), a2=int(idx2), rt=float(length), k=spring_constant_cc)
    constraints.append(c)
    
# Hookean for O-H bonds
spring_constant_oh = 5.0  # Adjust as needed
for pair in o_h_bond_indices:
    idx1, idx2 = pair
    length = atoms.get_distance(idx1, idx2) + 0.5
    c = Hookean(a1=int(idx1), a2=int(idx2), rt=float(length), k=spring_constant_cc)
    constraints.append(c)
    
# Hookean for C-N bonds
spring_constant_cn = 7.0  # Adjust as needed
for pair in c_n_bond_indices:
    idx1, idx2 = pair
    length = atoms.get_distance(idx1, idx2) + 0.5
    c = Hookean(a1=int(idx1), a2=int(idx2), rt=float(length), k=spring_constant_cc)
    constraints.append(c)
    d = Hookean(a1=int(idx2), a2=(0., 0., 1., -25.), k=15.) # apply a downward force on each adsorbate via the N atom if z is above 25 Å
    constraints.append(d)
    
# Hookean for O-N bonds
spring_constant_on = 7.0  # Adjust as needed
for pair in o_n_bond_indices:
    idx1, idx2 = pair
    length = atoms.get_distance(idx1, idx2) + 0.5
    c = Hookean(a1=int(idx1), a2=int(idx2), rt=float(length), k=spring_constant_cc)
    constraints.append(c)         

# fix surface
c = FixAtoms(indices=[atom.index for atom in atoms if
                                 atom.symbol == 'Au'])
constraints.append(c)

# set all constraints
atoms.set_constraint(constraints)

# Define your NequIP-based calculator
nequip_calculator = NequIPCalculator.from_deployed_model(argv[2])

# Attach the calculator to the ASE Atoms object
atoms.calc = nequip_calculator

# Instantiate and run the minima hopping algorithm.
hop = MinimaHopping(atoms,
                    Ediff0=2.5,
                    T0=4000.,
                    minima_traj="/scratch/work/jestilj1/nequip_minima_hopping/4NPaG_monolayer_alpha/minima_commmon.traj")
hop(totalsteps=100)

mhplot = MHPlot()
mhplot.save_figure(f'NequIP_{save_file}_MH_summary.png')
