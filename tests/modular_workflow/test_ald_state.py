import unittest

import numpy as np
from ase import Atoms

from ald_state import (
    ATOM_ID,
    IS_DEPOSITED_CENTER,
    ROLE_PRECURSOR_A,
    ROLE_WATER,
    classify_water_oh_endpoint,
    deposited_center_indices,
    ensure_atom_provenance,
    purge_detached_inserted_components,
    register_inserted_molecule,
    water_occupancy_status,
)


class WaterEndpointTests(unittest.TestCase):
    def test_counts_only_si_bound_oh_with_one_original_h(self):
        bonds = {(0, 1), (0, 3), (2, 4)}  # one retained O-H; other H transferred
        result = classify_water_oh_endpoint(bonds, 0, [1, 2], [3])
        self.assertTrue(result["qualifies"])
        self.assertEqual(result["si_index"], 3)
        self.assertEqual(result["bonded_original_hydrogen_indices"], [1])

    def test_rejects_intact_water_and_si_bridges(self):
        intact = classify_water_oh_endpoint({(0, 1), (0, 2), (0, 3)}, 0, [1, 2], [3])
        bridge = classify_water_oh_endpoint({(0, 1), (0, 3), (0, 4)}, 0, [1, 2], [3, 4])
        self.assertFalse(intact["qualifies"])
        self.assertEqual(intact["reason"], "requires_exactly_one_original_oh_bond")
        self.assertFalse(bridge["qualifies"])
        self.assertEqual(bridge["reason"], "ambiguous_multiple_si_bonds")

    def test_minimum_and_full_occupancy_are_distinct(self):
        empty = water_occupancy_status({"si1": []})
        one = water_occupancy_status({"si1": [10], "si2": [20]})
        two = water_occupancy_status({"si1": [10, 11], "si2": [20, 21]})
        duplicate = water_occupancy_status({"si1": [10, 10]})
        self.assertFalse(empty["minimum_success"])
        self.assertFalse(empty["full_occupancy"])
        self.assertTrue(one["minimum_success"])
        self.assertFalse(one["full_occupancy"])
        self.assertTrue(two["minimum_success"])
        self.assertTrue(two["full_occupancy"])
        self.assertEqual(duplicate["occupancy_counts"]["si1"], 1)


class ProvenanceAndPurgeTests(unittest.TestCase):
    def test_purge_removes_detached_components_but_preserves_grafted_atoms(self):
        atoms = Atoms(
            symbols=["Si", "O"],
            positions=[[0.0, 0.0, 0.0], [1.6, 0.0, 0.0]],
            cell=[30.0, 30.0, 30.0],
            pbc=False,
        )
        ensure_atom_provenance(atoms)
        substrate_ids = set(map(int, atoms.arrays[ATOM_ID]))

        # Grafted A center is part of the substrate-connected component.
        atoms += Atoms("Si", positions=[[3.2, 0.0, 0.0]])
        graft_index = len(atoms) - 1
        register_inserted_molecule(atoms, [graft_index], "precursor_A", center_local=0)
        atoms.arrays[IS_DEPOSITED_CENTER][graft_index] = 1
        graft_id = int(atoms.arrays[ATOM_ID][graft_index])

        # Detached water molecule and detached inserted singleton must both go.
        atoms += Atoms("OH", positions=[[15.0, 0.0, 0.0], [15.96, 0.0, 0.0]])
        water_indices = [3, 4]
        register_inserted_molecule(atoms, water_indices, "precursor_B", center_local=0)
        atoms += Atoms("H", positions=[[22.0, 0.0, 0.0]])
        singleton_index = len(atoms) - 1
        register_inserted_molecule(atoms, [singleton_index], "precursor_A", center_local=0)
        singleton_id = int(atoms.arrays[ATOM_ID][singleton_index])

        result = purge_detached_inserted_components(
            atoms,
            {"pair_cutoffs_A": {"O-H": 1.0, "O-Si": 1.8}},
        )
        remaining_ids = set(map(int, atoms.arrays[ATOM_ID]))
        self.assertTrue(substrate_ids.issubset(remaining_ids))
        self.assertIn(graft_id, remaining_ids)
        self.assertNotIn(singleton_id, remaining_ids)
        self.assertEqual(result["removed_atom_count"], 3)
        self.assertEqual(len(np.unique(atoms.arrays[ATOM_ID])), len(atoms))
        self.assertEqual(deposited_center_indices(atoms, ROLE_PRECURSOR_A), [2])

    def test_water_inventory_never_assigns_one_molecule_to_multiple_si(self):
        result = classify_water_oh_endpoint(
            {(0, 1), (0, 3), (0, 4)}, 0, [1, 2], [3, 4]
        )
        self.assertFalse(result["qualifies"])
        self.assertIsNone(result["si_index"])


if __name__ == "__main__":
    unittest.main()
