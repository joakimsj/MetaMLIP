#!/usr/bin/env python3
import argparse
import glob
import numpy as np
from ase.neighborlist import natural_cutoffs
from ase.io import read, write

# ========================== ARGUMENT PARSING ==========================

def parse_args():
    p = argparse.ArgumentParser(description="Reaction detection using CV window + changes in bond connectivity.")
    p.add_argument('--colvar', type=str, default=None)
    p.add_argument('--xyz', type=str, required=True)
    p.add_argument('--output', type=str, default='reaction_frames.extxyz')
    p.add_argument('--indices-output', type=str, default='reaction_indices.txt')
    p.add_argument('--delta', type=float, required=True, help='CV change threshold')
    p.add_argument('--cv', type=int, required=True, help='1-based CV index')
    p.add_argument('--padding', type=int, default=200, help='Frames around CV jump')
    p.add_argument('--persist', type=int, default=1, help='Number of frames connectivity change should persist for a reaction')    
    p.add_argument('--pre-frames', type=int, default=100, help='Frames before reaction for NEB window')
    p.add_argument('--post-frames', type=int, default=100, help='Frames after reaction for NEB window')
    p.add_argument('--form-scale', type=float, default=1.10, help='Bond formation cutoff scaling')
    p.add_argument('--break-scale', type=float, default=1.70, help='Bond breaking cutoff scaling')
    p.add_argument('--min_spacing', type=int, default=5, help='Minimum frames between reaction detection')
    p.add_argument('--zmin', type=float, default=10.0, help='Minimum z-position of atoms to be considered part of reaction ')
    p.add_argument('--debug', action='store_true', help='Verbose debug output')
    return p.parse_args()


# ========================== COLVAR HANDLING ==========================

def resolve_colvar_file(colvar_arg):
    if colvar_arg:
        return colvar_arg
    patterns = ["COLVAR", "*COLVAR*", "*colvar*"]
    matches = []
    for p in patterns:
        matches.extend(glob.glob(p))
    matches = sorted(set(matches))
    if len(matches) == 1:
        print(f"Auto-detected COLVAR file: {matches[0]}")
        return matches[0]
    elif len(matches) == 0:
        raise FileNotFoundError("No COLVAR-like file found.")
    else:
        raise RuntimeError("Multiple COLVAR-like files found:\n" + "\n".join(matches))


def load_colvar_data(colvar_file, cv_index):
    data = []
    with open(colvar_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            try:
                time = float(parts[0])
                cv_value = float(parts[cv_index])
                data.append((time, cv_value))
            except:
                continue
    if not data:
        raise RuntimeError(f"No usable CV data found in {colvar_file}")
    return np.array(data)


# ========================== TIME ALIGNMENT ==========================

def infer_traj_timestep(traj, cv_data=None):
    """
    Infer the timestep of the trajectory in the same units as the COLVAR (usually ps).
    If the trajectory frames have 'time' info, use that.
    Otherwise, use the total time from CV file and number of frames.
    
    Parameters:
        traj : ASE Atoms trajectory (list-like)
        cv_data : np.ndarray, shape (n,2), optional, first column = time
    
    Returns:
        dt_traj : float, timestep per frame
    """
    if 'time' in traj[0].info:
        times = [a.info['time'] for a in traj]
        dt_traj = np.mean(np.diff(times))
        return dt_traj
    elif cv_data is not None:
        total_time = int(cv_data[-1, 0]) - int(cv_data[0, 0])
        n_frames = len(traj)
        dt_traj = total_time / (n_frames - 1)
        print(f"[INFO] Inferring trajectory timestep from COLVAR: dt ≈ {dt_traj:.4f}")
        return dt_traj
    else:
        # fallback if nothing else
        print("[WARNING] No time info in traj or CV data; using default dt=10.0")
        return 10.0


def map_cv_to_trajectory(cv_data, traj, dt_traj):
    cv_times = cv_data[:, 0]
    cv_vals = cv_data[:, 1]
    traj_times = np.arange(len(traj)) * dt_traj
    traj_cv = np.zeros(len(traj))
    for i, t in enumerate(traj_times):
        idx = np.argmin(np.abs(cv_times - t))
        traj_cv[i] = cv_vals[idx]
    return traj_cv


# ========================== CV WINDOW DETECTION ==========================

def detect_cv_windows(traj_cv, delta, padding, debug=False):
    windows = []
    active = False
    start = None
    for i in range(1, len(traj_cv)):
        dcv = abs(traj_cv[i] - traj_cv[i-1])
        if dcv >= delta:
            if not active:
                start = i
                active = True
        else:
            if active:
                end = i
                windows.append((
                    max(0, start - padding),
                    min(len(traj_cv), end + padding)
                ))
                active = False
    if active:
        windows.append((max(0, start - padding), len(traj_cv)))
    merged = merge_overlapping_windows(windows)
    if debug:
        print("\n[CV] Final merged windows:")
        for w in merged:
            print(f"    {w[0]:6d} → {w[1]:6d}   Δt={(w[1]-w[0])} frames")
    return merged


def merge_overlapping_windows(windows):
    if not windows:
        return []
    windows.sort()
    merged = [windows[0]]
    for s, e in windows[1:]:
        ls, le = merged[-1]
        if s <= le:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


# ========================== HYSTERETIC CONNECTIVITY ==========================

def build_base_cutoffs(atoms):
    return np.array(natural_cutoffs(atoms))


def detect_bonds_hysteresis(atoms, prev_bonds, base_cutoffs,
                            form_scale, break_scale,
                            reactive_indices=None):
    pos = atoms.positions
    n = len(atoms)
    bonds = set()
    if reactive_indices is None:
        reactive_indices = set(range(n))
    for i in range(n):
        for j in range(i+1, n):
            if i not in reactive_indices and j not in reactive_indices:
                continue
            rij = np.linalg.norm(pos[i] - pos[j])
            r0 = base_cutoffs[i] + base_cutoffs[j]
            r_form = form_scale * r0
            r_break = break_scale * r0
            if (i, j) in prev_bonds:
                if rij < r_break:
                    bonds.add((i, j))
            else:
                if rij < r_form:
                    bonds.add((i, j))
    return bonds


def diff_bonds(prev, cur):
    formed = cur - prev
    broken = prev - cur
    return formed, broken

def detect_reactions_hysteresis(
        traj, traj_cv, cv_windows,
        form_scale,
        break_scale,
        persist,
        zmin,
        min_spacing,
        debug=False):

    detected = []
    reaction_log = []
    base_cutoffs = build_base_cutoffs(traj[0])

    for wstart, wend in cv_windows:

        if debug:
            print(f"\n[WINDOW] frames {wstart}:{wend}")

        # Define reactive region once per window
        if zmin is not None:
            reactive_indices = {
                i for i, pos in enumerate(traj[wstart].positions)
                if pos[2] > zmin
            }
            if debug:
                print(f"  Reactive atoms: {len(reactive_indices)}")
        else:
            reactive_indices = None

        prev_bonds = detect_bonds_hysteresis(
            traj[wstart], set(), base_cutoffs,
            form_scale, break_scale, reactive_indices
        )

        stable = 0
        candidate = None
        candidate_bonds = None
        last_reaction_frame = -min_spacing  # ensures first frame is considered

        for i in range(wstart+1, wend):

            # Skip frames too close to last detected reaction
            if i - last_reaction_frame < min_spacing:
                prev_bonds = detect_bonds_hysteresis(
                    traj[i], prev_bonds, base_cutoffs,
                    form_scale, break_scale, reactive_indices
                )
                continue

            bonds = detect_bonds_hysteresis(
                traj[i], prev_bonds, base_cutoffs,
                form_scale, break_scale, reactive_indices
            )

            if bonds != prev_bonds:
                stable += 1

                if stable == 1:
                    candidate = i
                    candidate_bonds = bonds

                if debug:
                    f, b = diff_bonds(prev_bonds, bonds)
                    print(f"[CANDIDATE] frame {i:6d}  CV={traj_cv[i]:8.4f}")
                    if f:
                        print("  Bonds formed:")
                        for a, b0 in sorted(f):
                            print(f"    {traj[i].symbols[a]}({a}) — {traj[i].symbols[b0]}({b0})")
                    if b:
                        print("  Bonds broken:")
                        for a, b0 in sorted(b):
                            print(f"    {traj[i].symbols[a]}({a}) — {traj[i].symbols[b0]}({b0})")

                if stable >= persist:
                    # Confirm reaction
                    if candidate not in detected:
                        detected.append(candidate)
                        f, b = diff_bonds(prev_bonds, candidate_bonds)
                        reaction_log.append((candidate, f, b))
                        last_reaction_frame = candidate

                        print(f"\n[REACTION CONFIRMED] frame {candidate}   CV={traj_cv[candidate]:.4f}")
                        print(f"  Persistent for {persist} frames")

                    # Reset stable to allow next reaction in window
                    stable = 0
                    candidate = None
                    candidate_bonds = None

            else:
                stable = 0
                candidate = None
                candidate_bonds = None

            prev_bonds = bonds

    return detected, reaction_log

# ========================== NEB WINDOW EXTRACTION ==========================

def extract_reaction_windows(indices, traj_len, pre, post):
    return [(i, max(0, i-pre), min(traj_len, i+post+1)) for i in indices]


# ========================== MAIN ==========================

def main():
    args = parse_args()

    colvar_file = resolve_colvar_file(args.colvar)
    cv_data = load_colvar_data(colvar_file, args.cv)

    traj = read(args.xyz, index=':')

    dt_traj = infer_traj_timestep(traj, cv_data)
    traj_cv = map_cv_to_trajectory(cv_data, traj, dt_traj)

    print(f"\nTrajectory frames: {len(traj)}")
    print(f"COLVAR entries:     {len(cv_data)}")
    print(f"Trajectory dt:      {dt_traj} fs")
    print("CV aligned → trajectory frames")

    cv_windows = detect_cv_windows(traj_cv, args.delta, args.padding, args.debug)
    print(f"\nDetected {len(cv_windows)} CV windows")

    reaction_indices, reaction_log = detect_reactions_hysteresis(
        traj, traj_cv, cv_windows,
        form_scale=args.form_scale,
        break_scale=args.break_scale,
        persist=args.persist,
        zmin=args.zmin,
        min_spacing=args.min_spacing,
        debug=args.debug
    )

    print(f"\nDetected {len(reaction_indices)} reactions")

    with open(args.indices_output, 'w') as f:
        for i in reaction_indices:
            f.write(f"{i}\n")

    write(args.output, [traj[i] for i in reaction_indices])
    print(f"Saved reaction frames → {args.output}")

    for rid, (i, start, end) in enumerate(
        extract_reaction_windows(reaction_indices, len(traj), args.pre_frames, args.post_frames), 1
    ):
        write(f"reaction_{rid:04d}_window.extxyz", traj[start:end])
        write(f"reaction_{rid:04d}_initial.xyz", traj[start])
        write(f"reaction_{rid:04d}_final.xyz", traj[end-1])

    if args.debug:
        print("\n========== SUMMARY ==========")
        for frame, formed, broken in reaction_log:
            print(f"Frame {frame:6d}   CV={traj_cv[frame]:.4f}")
            if formed:
                print("  Bonds formed:")
                for a, b in sorted(formed):
                    print(f"    {traj[frame].symbols[a]}({a}) — {traj[frame].symbols[b]}({b})")
            if broken:
                print("  Bonds broken:")
                for a, b in sorted(broken):
                    print(f"    {traj[frame].symbols[a]}({a}) — {traj[frame].symbols[b]}({b})")

    # Save reaction log
    with open("reaction_bonds.log", "w") as f:
        for frame, formed, broken in reaction_log:
            f.write(f"Frame {frame}\n")
            if formed:
                f.write("  Bonds formed:\n")
                for i, j in sorted(formed):
                    f.write(f"    {traj[frame].symbols[i]}({i}) — {traj[frame].symbols[j]}({j})\n")
            if broken:
                f.write("  Bonds broken:\n")
                for i, j in sorted(broken):
                    f.write(f"    {traj[frame].symbols[i]}({i}) — {traj[frame].symbols[j]}({j})\n")
            f.write("\n")

    print("\nDone.")


if __name__ == "__main__":
    main()
