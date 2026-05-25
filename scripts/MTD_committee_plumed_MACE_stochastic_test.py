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


import numpy as np
import os

from ase import units
from ase.io import read, write
from ase.constraints import FixAtoms
from ase.calculators.plumed import Plumed
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.neighborlist import NeighborList, natural_cutoffs

from mace.calculators import MACECalculator


# ============================================================
# USER SETTINGS
# ============================================================

MODEL_PATHS = ["MACE_model_seed_141575.model"]

SLAB_FILE = "SiO2_start.traj"
PRECURSOR_FILE = "TDMAS_mol.traj"

TEMPERATURE = 500
TIMESTEP_FS = 1.0

MAX_DEPOSITION_CYCLES = 3
MAX_STEPS_PER_CYCLE = 3000
CHECK_INTERVAL = 20

# deposition criteria
SURFACE_CN_THRESHOLD = 1.5
LIGAND_CN_THRESHOLD = 2.5

# precursor insertion
INSERTION_HEIGHT = 6.0

# ============================================================
# LOAD SYSTEM
# ============================================================

slab = read(SLAB_FILE)

# Fix bottom atoms
z_fix = np.percentile(slab.positions[:, 2], 20)

fixed = [i for i, a in enumerate(slab)
         if a.position[2] < z_fix]

slab.set_constraint(FixAtoms(indices=fixed))

# ============================================================
# MACE
# ============================================================

mace_calc = MACECalculator(
    model_paths=MODEL_PATHS,
    device="cuda",
    default_dtype="float64"
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def build_neighbor_list(atoms):

    cutoffs = natural_cutoffs(atoms, mult=1.1)

    nl = NeighborList(
        cutoffs,
        self_interaction=False,
        bothways=True
    )

    nl.update(atoms)

    return nl


def insert_precursor(atoms, precursor_file):

    precursor = read(precursor_file)

    slab_top = atoms.positions[:, 2].max()

    com = precursor.get_center_of_mass()

    shift = np.array([
        np.random.uniform(-2, 2),
        np.random.uniform(-2, 2),
        slab_top + INSERTION_HEIGHT - com[2]
    ])

    precursor.positions += shift

    atoms.extend(precursor)

    return atoms

def detect_tdmas(atoms):

    """
    Detect:
    - precursor Si center
    - precursor ligands
    - surface OH oxygens
    """

    nl = build_neighbor_list(atoms)

    precursor_data = []

    # --------------------------------------------------------
    # GLOBAL SURFACE OXYGEN DETECTION
    # --------------------------------------------------------

    z_surface = np.percentile(atoms.positions[:, 2], 70)

    surface_oxygens = []

    for i, atom in enumerate(atoms):

        if atom.symbol != "O":
            continue

        # top part of slab
        if atom.position[2] < z_surface:
            continue

        neigh, offsets = nl.get_neighbors(i)

        has_hydrogen = any(
            atoms[j].symbol == "H"
            for j in neigh
        )

        # OH oxygen
        if has_hydrogen:
            surface_oxygens.append(i)

    print("Detected OH oxygens:", surface_oxygens)

    # --------------------------------------------------------
    # PRECURSOR DETECTION
    # --------------------------------------------------------

    for i, atom in enumerate(atoms):

        if atom.symbol != "Si":
            continue

        neigh, offsets = nl.get_neighbors(i)

        ligands = []

        for j in neigh:

            if atoms[j].symbol in ["N", "C", "H"]:
                ligands.append(j)

        # precursor Si should have ligands
        if len(ligands) > 0:

            precursor_data.append({
                "center": i,
                "ligands": sorted(list(set(ligands))),
                "surface_oxygens": surface_oxygens
            })

    return precursor_data


def generate_plumed_input(center,
                          ligands,
                          surface_oxygens):

    ligand_str = ",".join(map(str, ligands))
    oxygen_str = ",".join(map(str, surface_oxygens))

    plumed_lines = [

        f"UNITS LENGTH=A "
        f"TIME={1/(1000*units.fs)} "
        f"ENERGY={units.mol/units.kJ}",

        # precursor-surface coordination
        f"c_surface: COORDINATION "
        f"GROUPA={center} "
        f"GROUPB={oxygen_str} "
        f"R_0=2.2",

        # precursor-ligand coordination
        f"c_ligand: COORDINATION "
        f"GROUPA={center} "
        f"GROUPB={ligand_str} "
        f"R_0=2.0",

        # metadynamics
        f"METAD ARG=c_surface,c_ligand "
        f"PACE=300 "
        f"HEIGHT=2.0 "
        f"SIGMA=0.15,0.15 "
        f"BIASFACTOR=8 "
        f"TEMP={TEMPERATURE} "
        f"FILE=HILLS",

        "PRINT ARG=c_surface,c_ligand "
        "STRIDE=10 FILE=COLVAR",

        "FLUSH STRIDE=1"
    ]

    return plumed_lines


def read_last_colvar():

    if not os.path.exists("COLVAR"):
        return None, None

    with open("COLVAR") as f:

        lines = [l for l in f
                 if not l.startswith("#")]

    if len(lines) == 0:
        return None, None

    vals = lines[-1].split()

    c_surface = float(vals[1])
    c_ligand = float(vals[2])

    return c_surface, c_ligand


def deposition_successful():

    c_surface, c_ligand = read_last_colvar()

    if c_surface is None:
        return False

    print(
        f"CVs -> "
        f"surface={c_surface:.3f} "
        f"ligand={c_ligand:.3f}"
    )

    return (
        c_surface > SURFACE_CN_THRESHOLD and
        c_ligand < LIGAND_CN_THRESHOLD
    )


# ============================================================
# MAIN WORKFLOW
# ============================================================

atoms = slab.copy()

MaxwellBoltzmannDistribution(
    atoms,
    temperature_K=TEMPERATURE
)

for cycle in range(MAX_DEPOSITION_CYCLES):

    print("\n")
    print("=" * 60)
    print(f"DEPOSITION CYCLE {cycle}")
    print("=" * 60)

    # --------------------------------------------------------
    # INSERT NEW PRECURSOR
    # --------------------------------------------------------

    atoms = insert_precursor(atoms, PRECURSOR_FILE)

    # --------------------------------------------------------
    # DETECT PRECURSOR
    # --------------------------------------------------------

    precursor_data = detect_tdmas(atoms)

    if len(precursor_data) == 0:
        raise RuntimeError("No precursor detected")

    active = precursor_data[-1]

    center = active["center"]
    ligands = active["ligands"]
    surface_oxygens = active["surface_oxygens"]

    print("Detected precursor:")
    print("center =", center)
    print("ligands =", ligands)
    print("surface oxygens =", surface_oxygens)

    # --------------------------------------------------------
    # RESET PLUMED
    # --------------------------------------------------------

    if os.path.exists("HILLS"):
        os.remove("HILLS")

    if os.path.exists("COLVAR"):
        os.remove("COLVAR")

    if atoms.calc is not None:
        del atoms.calc

    plumed_input = generate_plumed_input(
        center,
        ligands,
        surface_oxygens
    )

    atoms.calc = Plumed(
        calc=mace_calc,
        input=plumed_input,
        timestep=TIMESTEP_FS,
        atoms=atoms,
        kT=TEMPERATURE * units.kB
    )

    dyn = VelocityVerlet(
        atoms,
        timestep=TIMESTEP_FS * units.fs
    )

    # --------------------------------------------------------
    # RUN SHORT CHUNKS
    # --------------------------------------------------------

    deposited = False

    for step in range(0,
                      MAX_STEPS_PER_CYCLE,
                      CHECK_INTERVAL):

        dyn.run(CHECK_INTERVAL)

        write(
            "trajectory.xyz",
            atoms,
            append=True
        )

        if deposition_successful():

            print("Deposition detected.")
            deposited = True
            break

    # --------------------------------------------------------
    # FAILSAFE
    # --------------------------------------------------------

    if not deposited:

        print("Deposition timeout.")
        print("Trying next precursor.")

        continue

    print("Proceeding to next precursor pulse.")

# ============================================================
# SAVE FINAL STRUCTURE
# ============================================================

write("final_structure.xyz", atoms)

print("Done.")
