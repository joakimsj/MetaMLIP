import copy
import unittest
from pathlib import Path

import yaml

from ald_workflow_surface_agnostic import build_plumed_input, load_config, validate_config


CONFIG_PATH = Path(__file__).with_name("ald_config.yaml")


class PhaseConfigTests(unittest.TestCase):
    def test_example_config_validates_phase_specific_plumed_settings(self):
        config = load_config(CONFIG_PATH)
        self.assertIn("plumed", config["chemistry"]["precursor_A"])
        self.assertIn("plumed", config["chemistry"]["precursor_B"])
        self.assertNotIn("plumed", config)

    def test_missing_phase_plumed_settings_is_an_error(self):
        with CONFIG_PATH.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        del config["chemistry"]["precursor_B"]["plumed"]
        with self.assertRaisesRegex(ValueError, r"chemistry\.precursor_B\.plumed is required"):
            validate_config(config)

    def test_plumed_builder_uses_the_active_phase_settings(self):
        config = load_config(CONFIG_PATH)
        phase_a = copy.deepcopy(config["chemistry"]["precursor_A"])
        phase_b = copy.deepcopy(config["chemistry"]["precursor_B"])
        phase_b["plumed"]["pace"] = 777
        phase_b["plumed"]["surface_r0_A"] = 3.1

        a_input = build_plumed_input(0, [1], [2], phase_a, config, "precursor_A")
        b_input = build_plumed_input(0, [1], [2], phase_b, config, "precursor_B")
        self.assertTrue(any("PACE=300" in line for line in a_input))
        self.assertTrue(any("PACE=777" in line for line in b_input))
        self.assertTrue(any("R_0=3.1" in line for line in b_input))


if __name__ == "__main__":
    unittest.main()
