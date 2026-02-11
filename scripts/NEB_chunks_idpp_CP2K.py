import os
import sys
import numpy as np
from ase.io import read, write
from ase.constraints import FixAtoms
from ase.mep import NEB
from ase.calculators.cp2k import CP2K
from ase.optimize.fire import FIRE
import time

# -------------------------
# User parameters
# -------------------------
n_images = 8
z_threshold = 2.0
fmax_target = 0.05
steps_per_chunk = 2
max_chunks = 300
restart_file = "neb_restart.traj"
energy_file = "neb_energies.txt"

# -------------------------
# Read input
# -------------------------
initial = read(sys.argv[1])
final = read(sys.argv[2])
initial_name = os.path.splitext(sys.argv[1])[0]

# -------------------------
# CP2K input block
# -------------------------
inp = """
&FORCE_EVAL
  METHOD Quickstep
  &DFT
    BASIS_SET_FILE_NAME BASIS_MOLOPT_UZH
    POTENTIAL_FILE_NAME POTENTIAL_UZH
    CHARGE 0
    MULTIPLICITY 1
    &MGRID
       CUTOFF [Ry] 400
    &END
    &XC
      &XC_FUNCTIONAL
         &PBE
         &END
      &END XC_FUNCTIONAL
      ! adding Grimme's D3 correction (by default without C9 terms)
      &VDW_POTENTIAL
         POTENTIAL_TYPE PAIR_POTENTIAL
         &PAIR_POTENTIAL
            PARAMETER_FILE_NAME dftd3.dat
            TYPE DFTD3
            REFERENCE_FUNCTIONAL PBE
            R_CUTOFF [angstrom] 16
         &END
      &END VDW_POTENTIAL
    &END XC
    &QS
       METHOD GPW
       EPS_DEFAULT 1.0E-10
       EXTRAPOLATION ASPC
    &END
    &POISSON
       PERIODIC XYZ ! the default, gas phase systems should have 'NONE' and a wavelet solver
       POISSON_SOLVER PERIODIC
    &END
    ! use the OT METHOD for robust and efficient SCF, suitable for all non-metallic systems.
    &SCF
      SCF_GUESS ATOMIC ! can be used to RESTART an interrupted calculation
      MAX_SCF 300
      EPS_SCF 1.0E-6 ! accuracy of the SCF procedure typically 1.0E-6 - 1.0E-7
      &OT
        ! an accurate preconditioner suitable also for larger systems
        PRECONDITIONER FULL_ALL
        ENERGY_GAP 0.001
        MINIMIZER DIIS
      &END OT
      &OUTER_SCF ! repeat the inner SCF cycle 10 times
        MAX_SCF 5
        EPS_SCF 1.0E-6 ! must match the above
      &END
      &PRINT
        &RESTART OFF
        &END
      &END PRINT
    IGNORE_CONVERGENCE_FAILURE TRUE
    &END SCF
  &END DFT
  &SUBSYS
    &KIND O
      BASIS_SET ORB DZVP-MOLOPT-PBE-GTH-q6
      POTENTIAL GTH-PBE-q6
    &END KIND
    &KIND C
      BASIS_SET ORB DZVP-MOLOPT-PBE-GTH-q4
      POTENTIAL GTH-PBE-q4
    &END KIND
    &KIND Si
      BASIS_SET ORB DZVP-MOLOPT-PBE-GTH-q4
      POTENTIAL GTH-PBE-q4
    &END KIND
    &KIND H
      BASIS_SET ORB DZVP-MOLOPT-PBE-GTH-q1
      POTENTIAL GTH-PBE-q1
    &END KIND
    &KIND N
      BASIS_SET ORB DZVP-MOLOPT-PBE-GTH-q5
      POTENTIAL GTH-PBE-q5
    &END KIND
  &END SUBSYS
  &PRINT
    &FORCES ON
    &END FORCES
  &END PRINT
&END FORCE_EVAL
"""
# -------------------------
# Utility functions
# -------------------------
def apply_constraints(atoms):
    fixed_indices = [i for i, atom in enumerate(atoms)
                     if atom.position[2] < z_threshold]
    atoms.set_constraint(FixAtoms(indices=fixed_indices))

def relax_structure(atoms, label):
    """Relax structure using CP2K in a with-block to ensure proper cleanup."""
    with CP2K(
        command=os.environ['ASE_CP2K_COMMAND'],
        inp=inp,
        set_pos_file=True,
        label=label,
        basis_set=None,
        basis_set_file=None,
        max_scf=None,
        cutoff=None,
        force_eval_method=None,
        potential_file=None,
        poisson_solver=None,
        pseudo_potential=None,
        stress_tensor=False,
        xc=None
    ) as calc:
        atoms.calc = calc
        apply_constraints(atoms)
        opt = FIRE(atoms, trajectory=f"{label}.traj")
        opt.run(fmax=fmax_target)
        # Return energy after relaxation
        return atoms.calc.results.get('energy', None)

def save_energies(images, filename):
    """Save energies of all images to a file."""
    energies = []
    for i, img in enumerate(images):
        e = img.calc.results.get('energy', np.nan)
        energies.append(e)
    np.savetxt(filename, np.array(energies))
    print(f"Energies saved to {filename}")

def read_last_fmax(logfile):
    with open(logfile) as f:
        for line in f:
            pass  # iterate to the last line
    if "FIRE:" in line:
        parts = line.split()
        return float(parts[-1])  # fmax is last column
    return None

# -------------------------
# Relax endpoints if needed
# -------------------------
if not os.path.exists(f"{initial_name}_relax_cp2k.traj"):
    relax_structure(initial, f"{initial_name}_relax_cp2k")
else:
    initial = read(f"{initial_name}_relax_cp2k.traj", index=-1)

if not os.path.exists("final_relax_cp2k.traj"):
    relax_structure(final, "final_relax_cp2k")
else:
    final = read("final_relax_cp2k.traj", index=-1)

# -------------------------
# Robust restart detection
# -------------------------

def detect_last_chunk():
    logs = [f for f in os.listdir(".") if f.startswith("neb_chunk_") and f.endswith(".log")]
    if not logs:
        return -1
    chunks = [int(f.split("_")[-1].split(".")[0]) for f in logs]
    return max(chunks)

# -------------------------
# Energy logging (append mode)
# -------------------------

def append_energies(images, filename, chunk):
    with open(filename, "a") as f:
        f.write(f"# chunk {chunk}\n")
        for i, img in enumerate(images):
            e = img.calc.results.get("energy", np.nan)
            f.write(f"{chunk:5d} {i:3d} {e:20.10f}\n")
    print(f"Energies appended for chunk {chunk}")

# -------------------------
# Restart NEB geometries
# -------------------------

if os.path.exists(restart_file):
    print("Restarting NEB from", restart_file)
    images = read(restart_file, index=":")
else:
    images = [initial]
    for _ in range(n_images):
        images.append(initial.copy())
    images.append(final)

    for img in images:
        apply_constraints(img)

    neb = NEB(images, allow_shared_calculator=True)
    neb.interpolate("idpp")

# -------------------------
# Detect restart chunk index
# -------------------------

start_chunk = detect_last_chunk() + 1
print(f"Starting NEB from chunk {start_chunk}")

# -------------------------
# Chunked NEB loop
# -------------------------

for chunk in range(start_chunk, max_chunks):

    print(f"\n===== NEB chunk {chunk} =====")

    with CP2K(
        command=os.environ['ASE_CP2K_COMMAND'],
        inp=inp,
        set_pos_file=True,
        label=f"{initial_name}_neb_chunk_{chunk}",
        basis_set=None,
        basis_set_file=None,
        max_scf=None,
        cutoff=None,
        force_eval_method=None,
        potential_file=None,
        poisson_solver=None,
        pseudo_potential=None,
        stress_tensor=False,
        xc=None
    ) as calc:

        for img in images:
            img.calc = calc

        neb = NEB(images, allow_shared_calculator=True)

        opt = FIRE(
            neb,
            trajectory=f"neb_chunk_{chunk}.traj",
            logfile=f"neb_chunk_{chunk}.log"
        )

        opt.run(steps=steps_per_chunk)

        # Save energies after chunk
        append_energies(images, energy_file, chunk)

    # ---------- HARD cleanup (Slurm-safe) ----------

    write(restart_file, images)

    del opt, neb, calc
    for img in images:
        img.calc = None

    import gc
    gc.collect()
    time.sleep(15)  # important for Slurm step release

    # ---------- Convergence check from logfile ----------

    with open(f"neb_chunk_{chunk}.log") as f:
        for line in f:
            pass

    if "FIRE:" in line:
        fmax = float(line.split()[-1])
        print(f"Chunk {chunk}: fmax = {fmax:.6f}")
        if fmax < fmax_target:
            print("\nNEB converged successfully.")
            break

else:
    print("\nWARNING: NEB did not converge within max_chunks.")

# -------------------------
# Final writeout
# -------------------------

write("neb_final.traj", images)
print("Final NEB written to neb_final.traj")

