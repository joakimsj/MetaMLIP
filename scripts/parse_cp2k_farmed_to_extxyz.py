import os
import numpy as np
from ase.io import read, write
from ase import Atoms
from ase.units import Hartree, Bohr

def parse_cp2k_farming_output(filepath):
    energy = None
    forces = []
    in_force_block = False
    scf_converged = True

    with open(filepath, 'r') as f:
        for line in f:
            # Check SCF convergence more robustly
            if "SCF run NOT converged" in line:
                scf_converged = False

            # Parse energy
            if 'ENERGY|' in line and 'Total FORCE_EVAL ( QS ) energy [a.u.]:' in line:
                try:
                    energy = float(line.split()[-1]) * Hartree  # Convert to eV
                except ValueError:
                    continue

            # Detect start of force block
            elif 'ATOMIC FORCES in [a.u.]' in line:
                in_force_block = True
                skip_lines = 2  # Skip headers
                continue

            elif in_force_block:
                if skip_lines > 0:
                    skip_lines -= 1
                    continue
                if line.strip() == "":
                    break  # End of block
                try:
                    parts = line.split()
                    fx, fy, fz = [float(x) * Hartree / Bohr for x in parts[3:6]]
                    forces.append([fx, fy, fz])
                except (ValueError, IndexError):
                    continue

    if not scf_converged:
        print(f"⚠️ Skipping {filepath} — SCF did not converge.")
        return None

    if energy is None or not forces:
        print(f"⚠️ Skipping {filepath} — energy or forces not found.")
        return None

    return energy, forces


def collect_cp2k_results(run_prefix='run', structure_file='structure.xyz', farming_prefix='FARMING_OUT_', output='cp2k_farmed_dataset.xyz'):
    all_atoms = []

    for run_dir in sorted(os.listdir()):
        if not run_dir.startswith(run_prefix) or not os.path.isdir(run_dir):
            continue

        structure_path = os.path.join(run_dir, structure_file)
        farming_file = None
        for f in os.listdir(run_dir):
            if f.startswith(farming_prefix):
                farming_file = os.path.join(run_dir, f)
                break

        if not os.path.isfile(structure_path) or not farming_file:
            print(f"⚠️ Missing files in {run_dir}, skipping.")
            continue

        try:
            atoms = read(structure_path)
        except Exception as e:
            print(f"⚠️ Failed to read {structure_path}: {e}")
            continue

        parsed = parse_cp2k_farming_output(farming_file)
        if parsed is None:
            continue

        energy, forces = parsed
        atoms.info["REF_energy"] = energy
        atoms.set_array("REF_forces", np.array(forces))
        all_atoms.append(atoms)

    if all_atoms:
        write(output, all_atoms)
        print(f"✅ Wrote {len(all_atoms)} structures to {output}")
    else:
        print("❌ No structures parsed.")

if __name__ == "__main__":
    collect_cp2k_results()

