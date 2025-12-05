import argparse
import numpy as np
from ase.io import read, write
from ase.calculators.cp2k import CP2K
import os

# CP2K input string
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
      MAX_SCF 10
      EPS_SCF 1.0E-6 ! accuracy of the SCF procedure typically 1.0E-6 - 1.0E-7
      &OT
        ! an accurate preconditioner suitable also for larger systems
        PRECONDITIONER FULL_ALL
	ENERGY_GAP 0.001
        MINIMIZER DIIS
      &END OT
      &OUTER_SCF ! repeat the inner SCF cycle 10 times
        MAX_SCF 900
        EPS_SCF 1.0E-6 ! must match the above
      &END
      &PRINT
        &RESTART OFF
        &END
      &END PRINT
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
    &KIND Si
      BASIS_SET ORB DZVP-MOLOPT-PBE-GTH-q4
      POTENTIAL GTH-PBE-q4
    &END KIND
  &END SUBSYS
  &PRINT
    &FORCES ON
    &END FORCES
  &END PRINT
&END FORCE_EVAL
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Run CP2K single point + force calculations on selected frames.")
    parser.add_argument('--xyz', type=str, required=True, help='Input extxyz file (e.g., filtered.extxyz).')
    parser.add_argument('--indices', type=int, nargs='*', help='Explicit list of indices to include.')
    parser.add_argument('--start', type=int, help='Start index (inclusive) for a range.')
    parser.add_argument('--end', type=int, default=50 , help='End index (exclusive) for a range.')
    parser.add_argument('--cp2k-label', type=str, default='cp2k_calc', help='Label prefix for CP2K runs.')
    parser.add_argument('--output', type=str, default='cp2k_results.extxyz', help='Output extxyz file with energy and forces.')
    parser.add_argument('--checkpoint', type=str, default='cp2k_completed_indices.txt', help='Checkpoint file to store completed indices.')
    return parser.parse_args()

def select_indices(args, total):
    if args.indices:
        return args.indices
    elif args.start is not None or args.end is not None:
        start = args.start if args.start is not None else 0
        end = args.end if args.end is not None else total
        return list(range(start, end))
    else:
        raise ValueError("Please provide --indices, or --start/--end to define which frames to calculate.")

def load_checkpoint(path):
    if not os.path.isfile(path):
        return set()
    with open(path, 'r') as f:
        return set(int(line.strip()) for line in f if line.strip().isdigit())

def append_checkpoint(path, index):
    with open(path, 'a') as f:
        f.write(f"{index}\n")

def run_cp2k_calculations(atoms_list, indices, label_prefix, output_file, checkpoint_file):
    completed = load_checkpoint(checkpoint_file)

    # Truncate output file if starting from scratch
    if not os.path.exists(output_file):
        open(output_file, 'w').close()

    for i in indices:
        if i in completed:
            print(f"Skipping frame {i} (already completed).")
            continue

        atoms = atoms_list[i]
        calc = CP2K(
            basis_set=None,
            basis_set_file=None,
            max_scf=None,
            cutoff=None,
            force_eval_method=None,
            potential_file=None,
            poisson_solver=None,
            pseudo_potential=None,
            stress_tensor=False,
            xc=None,
            inp=inp,
            label=f"{label_prefix}_{i}"
        )
        atoms.calc = calc

        try:
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            atoms.info["REF_energy"] = energy
            atoms.arrays["REF_forces"] = forces

            write(output_file, atoms, append=True)
            append_checkpoint(checkpoint_file, i)

            print(f"Frame {i}: Energy = {energy:.6f} eV (saved)")

        except Exception as e:
            print(f"Frame {i}: CP2K failed with error:\n{e}\nSkipping.")
        finally:
            calc.close()

def main():
    args = parse_args()

    atoms_list = read(args.xyz, index=":")
    indices = select_indices(args, len(atoms_list))

    run_cp2k_calculations(
        atoms_list=atoms_list,
        indices=indices,
        label_prefix=args.cp2k_label,
        output_file=args.output,
        checkpoint_file=args.checkpoint
    )

    print(f"Done. Results saved in {args.output}")

if __name__ == "__main__":
    main()

