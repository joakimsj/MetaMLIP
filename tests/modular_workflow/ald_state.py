"""Stable atom provenance and chemistry-state helpers for the ALD workflow.

Connectivity remains a geometric hypothesis; these helpers only provide
bookkeeping and apply the configured graph's edge classifications.
"""
from __future__ import annotations

import numpy as np

from ald_connectivity import (
    build_base_cutoffs,
    detect_bonds_hysteresis,
    find_connected_components,
)

ROLE_SUBSTRATE = 0
ROLE_PRECURSOR_A = 1
ROLE_WATER = 2

ATOM_ID = "ald_atom_id"
MOLECULE_ID = "ald_molecule_id"
ATOM_ROLE = "ald_atom_role"
IS_CENTER = "ald_is_center"
IS_DEPOSITED_CENTER = "ald_is_deposited_center"


def ensure_atom_provenance(atoms):
    """Initialize stable integer IDs and role arrays without changing geometry."""
    n_atoms = len(atoms)
    if ATOM_ID not in atoms.arrays:
        atoms.set_array(ATOM_ID, np.arange(n_atoms, dtype=np.int64))
    if MOLECULE_ID not in atoms.arrays:
        atoms.set_array(MOLECULE_ID, np.full(n_atoms, -1, dtype=np.int64))
    if ATOM_ROLE not in atoms.arrays:
        atoms.set_array(ATOM_ROLE, np.full(n_atoms, ROLE_SUBSTRATE, dtype=np.int8))
    if IS_CENTER not in atoms.arrays:
        atoms.set_array(IS_CENTER, np.zeros(n_atoms, dtype=np.int8))
    if IS_DEPOSITED_CENTER not in atoms.arrays:
        atoms.set_array(IS_DEPOSITED_CENTER, np.zeros(n_atoms, dtype=np.int8))

    ids = np.asarray(atoms.arrays[ATOM_ID], dtype=np.int64)
    if ids.shape != (n_atoms,) or len(np.unique(ids)) != n_atoms:
        raise ValueError("ald_atom_id must contain one unique stable ID per atom")
    atoms.info["_ald_next_atom_id"] = max(
        int(atoms.info.get("_ald_next_atom_id", 0)),
        int(ids.max()) + 1 if n_atoms else 0,
    )
    atoms.info["_ald_next_molecule_id"] = max(
        int(atoms.info.get("_ald_next_molecule_id", 0)),
        int(np.max(atoms.arrays[MOLECULE_ID])) + 1 if n_atoms else 0,
    )
    atoms.info.setdefault("_ald_substrate_atom_ids", [int(x) for x in ids])
    return atoms


def register_inserted_molecule(atoms, indices, role, center_local=0):
    """Assign stable atom/molecule IDs after appending a molecule to ``atoms``."""
    # ``insert_molecule`` initializes arrays before ASE extends the structure;
    # appended rows are placeholders until assigned below.
    if any(name not in atoms.arrays for name in (ATOM_ID, MOLECULE_ID, ATOM_ROLE, IS_CENTER, IS_DEPOSITED_CENTER)):
        ensure_atom_provenance(atoms)
    indices = [int(i) for i in indices]
    if not indices or min(indices) < 0 or max(indices) >= len(atoms):
        raise IndexError("Inserted molecule indices are empty or outside the structure")
    if center_local < 0 or center_local >= len(indices):
        raise IndexError("center_local is outside the inserted molecule")
    role_value = {"precursor_A": ROLE_PRECURSOR_A, "precursor_B": ROLE_WATER}.get(role)
    if role_value is None:
        raise ValueError(f"Unsupported inserted molecule role: {role}")

    first_id = int(atoms.info["_ald_next_atom_id"])
    molecule_id = int(atoms.info["_ald_next_molecule_id"])
    atoms.arrays[ATOM_ID][indices] = np.arange(first_id, first_id + len(indices))
    atoms.arrays[MOLECULE_ID][indices] = molecule_id
    atoms.arrays[ATOM_ROLE][indices] = role_value
    atoms.arrays[IS_CENTER][indices] = 0
    atoms.arrays[IS_CENTER][indices[center_local]] = 1
    atoms.arrays[IS_DEPOSITED_CENTER][indices] = 0
    atoms.info["_ald_next_atom_id"] = first_id + len(indices)
    atoms.info["_ald_next_molecule_id"] = molecule_id + 1
    if len(np.unique(atoms.arrays[ATOM_ID])) != len(atoms):
        raise ValueError("Stable atom ID assignment produced duplicates")
    return molecule_id


def indices_for_atom_ids(atoms, atom_ids):
    """Resolve stable IDs to current ASE indices; missing IDs are omitted."""
    wanted = {int(x) for x in atom_ids}
    return {
        int(atom_id): index
        for index, atom_id in enumerate(atoms.arrays[ATOM_ID])
        if int(atom_id) in wanted
    }


def deposited_center_indices(atoms, role=ROLE_PRECURSOR_A):
    """Return current indices for successfully deposited precursor centers."""
    mask = (
        (atoms.arrays[ATOM_ROLE] == int(role))
        & (atoms.arrays[IS_CENTER] == 1)
        & (atoms.arrays[IS_DEPOSITED_CENTER] == 1)
    )
    return np.flatnonzero(mask).astype(int).tolist()


def purge_detached_inserted_components(atoms, connectivity_cfg=None):
    """Remove inserted-provenance components disconnected from original substrate.

    A fresh full-structure graph is used for conservative component discovery.
    Any component containing inserted atoms and no original substrate atom is
    removed in its entirety, including singleton components. Every component
    containing at least one original substrate atom is preserved.
    """
    ensure_atom_provenance(atoms)
    cfg = connectivity_cfg or {}
    radii = build_base_cutoffs(atoms, cfg.get("atomic_radii_A"))
    # Purge should bias toward preservation: use the break-distance envelope
    # even for edges absent from the instantaneous formation graph.
    purge_scale = max(
        float(cfg.get("form_scale", 1.10)),
        float(cfg.get("break_scale", 1.70)),
    )
    bonds = detect_bonds_hysteresis(
        atoms,
        (),
        radii,
        purge_scale,
        purge_scale,
        pair_cutoffs=cfg.get("pair_cutoffs_A"),
    )
    substrate_ids = {int(x) for x in atoms.info.get("_ald_substrate_atom_ids", [])}
    atom_ids = atoms.arrays[ATOM_ID]
    roles = atoms.arrays[ATOM_ROLE]
    inserted_roles = {ROLE_PRECURSOR_A, ROLE_WATER}
    doomed = []
    removed_ids = []
    removed_components = []
    for component in find_connected_components(len(atoms), bonds):
        has_inserted = any(int(roles[i]) in inserted_roles for i in component)
        has_substrate = any(int(atom_ids[i]) in substrate_ids for i in component)
        if has_inserted and not has_substrate:
            doomed.extend(component)
            removed_ids.extend(int(atom_ids[i]) for i in component)
            removed_components.append([int(atom_ids[i]) for i in component])

    if doomed:
        del atoms[sorted(doomed)]
    if "_ald_surface_anchor_atom_ids" in atoms.info:
        current = indices_for_atom_ids(atoms, atoms.info["_ald_surface_anchor_atom_ids"])
        atoms.info["_ald_surface_anchor_indices"] = sorted(current.values())
    return {
        "removed_atom_ids": sorted(removed_ids),
        "removed_components_atom_ids": removed_components,
        "removed_atom_count": len(removed_ids),
        "remaining_atom_count": len(atoms),
    }


def classify_water_oh_endpoint(bonds, water_oxygen_index, original_hydrogen_indices,
                               deposited_si_indices):
    """Classify one water as Si-bound OH using stable graph-edge indices.

    Returns a deposited Si index only when the water oxygen has exactly one
    deposited-Si neighbor and exactly one of its original H atoms remains
    connected to the oxygen. Any other pattern is rejected with a reason.
    """
    edges = {tuple(sorted((int(i), int(j)))) for i, j in bonds}
    oxygen = int(water_oxygen_index)
    hydrogen_ids = [int(x) for x in original_hydrogen_indices]
    si_ids = [int(x) for x in deposited_si_indices]
    bonded_h = [h for h in hydrogen_ids if tuple(sorted((oxygen, h))) in edges]
    bonded_si = [si for si in si_ids if tuple(sorted((oxygen, si))) in edges]
    hydrogen_neighbors = {
        str(h): sorted(
            int(j) if int(i) == h else int(i)
            for i, j in edges if h in (int(i), int(j))
        )
        for h in hydrogen_ids
    }
    if len(bonded_si) != 1:
        reason = "no_deposited_si_bond" if not bonded_si else "ambiguous_multiple_si_bonds"
        return {"qualifies": False, "si_index": None, "reason": reason,
                "bonded_original_hydrogen_indices": bonded_h,
                "bonded_deposited_si_indices": bonded_si,
                "original_hydrogen_neighbors": hydrogen_neighbors}
    if len(bonded_h) != 1:
        return {"qualifies": False, "si_index": bonded_si[0],
                "reason": "requires_exactly_one_original_oh_bond",
                "bonded_original_hydrogen_indices": bonded_h,
                "bonded_deposited_si_indices": bonded_si,
                "original_hydrogen_neighbors": hydrogen_neighbors}
    return {"qualifies": True, "si_index": bonded_si[0], "reason": "si_bound_oh",
            "bonded_original_hydrogen_indices": bonded_h,
            "bonded_deposited_si_indices": bonded_si,
            "original_hydrogen_neighbors": hydrogen_neighbors}


def water_occupancy_status(occupancy, minimum_per_si=1, target_per_si=2):
    """Summarize per-Si distinct-water inventory with explicit minimum/full states."""
    minimum_per_si = int(minimum_per_si)
    target_per_si = int(target_per_si)
    if minimum_per_si < 1 or target_per_si < minimum_per_si:
        raise ValueError("Require 1 <= minimum_per_si <= target_per_si")
    counts = {
        str(si): len({int(water) for water in waters})
        for si, waters in occupancy.items()
    }
    has_sites = bool(counts)
    return {
        "occupancy_counts": counts,
        "minimum_success": has_sites and all(
            count >= minimum_per_si for count in counts.values()
        ),
        "full_occupancy": has_sites and all(
            count >= target_per_si for count in counts.values()
        ),
    }
