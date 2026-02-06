import argparse
import glob
import numpy as np
from ase.io import read, write

def parse_args():
    parser = argparse.ArgumentParser(description="Extract snapshots based on CV changes from metadynamics.")
    parser.add_argument('--colvar', type=str, default=None,
                        help='Path to COLVAR.metadynLog file (defaults to *COLVAR.metadynLog if found).')
    parser.add_argument('--xyz', type=str, required=True, help='Input .extxyz file.')
    parser.add_argument('--output', type=str, default='filtered.extxyz', help='Output .extxyz file.')
    parser.add_argument('--indices-output', type=str, default='selected_indices_from_trajectory.txt',
                        help='Filename to save selected frame indices.')
    parser.add_argument('--delta', type=float, required=True, help='Minimum change in collective variable.')
    parser.add_argument('--cv', type=int, required=True, help='Index (1-based) of collective variable to track.')
    return parser.parse_args()

def resolve_colvar_file(colvar_arg):
    if colvar_arg is not None:
        return colvar_arg
    matches = glob.glob("*COLVAR.metadynLog")
    if len(matches) == 1:
        print(f"Using default COLVAR file: {matches[0]}")
        return matches[0]
    elif len(matches) == 0:
        raise FileNotFoundError("No file ending in 'COLVAR.metadynLog' found in the current directory.")
    else:
        raise RuntimeError(f"Multiple files matching '*COLVAR.metadynLog' found: {matches}. Please specify with --colvar.")

def load_colvar_data(colvar_file, cv_index):
    data = []
    with open(colvar_file, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.split()
                timestep = int(float(parts[0]))
                cv_value = float(parts[cv_index])
                data.append((timestep, cv_value))
    return data

def select_frames_by_cv_change(data, delta):
    selected = [data[0][0]]  # Always select the first frame
    last_value = data[0][1]
    for timestep, cv_value in data[1:]:
        if abs(cv_value - last_value) >= delta:
            selected.append(timestep)
            last_value = cv_value
    return selected

def main():
    args = parse_args()
    colvar_file = resolve_colvar_file(args.colvar)

    # Load CV data
    cv_data = load_colvar_data(colvar_file, args.cv)

    # Determine which frames to keep
    selected_timesteps = select_frames_by_cv_change(cv_data, args.delta)

    # Save selected indices
    with open(args.indices_output, 'w') as f:
        for idx in selected_timesteps:
            f.write(f"{idx}\n")
    print(f"Saved selected indices to {args.indices_output}")

    # Read trajectory and filter
    all_atoms = read(args.xyz, index=':')
    selected_atoms = [atoms for i, atoms in enumerate(all_atoms) if i in selected_timesteps]

    # Write output
    write(args.output, selected_atoms)
    print(f"Written {len(selected_atoms)} frames to {args.output}")

if __name__ == '__main__':
    main()

