import subprocess
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from shapely.geometry import Point

from gedicorrect.simulation import process_all_footprints


class SimulationCommandTests(unittest.TestCase):
    @patch("gedicorrect.simulation.subprocess.run")
    def test_failed_gedirat_command_discards_footprint(self, run_command):
        run_command.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=["gediRat"],
            stderr="simulation failed",
        )
        footprint = pd.Series({
            "shot_number_x": 1,
            "intersecting_las": ["tile.las"],
            "geometry": Point(10, 20),
        })
        original = pd.DataFrame({"shot_number_x": [1], "rx_sample_count": [100]})

        with tempfile.TemporaryDirectory() as temporary_directory:
            result = process_all_footprints(
                footprint,
                temp_dir=temporary_directory,
                las_dir="/data/als",
                original_df=original,
                crs="32629",
                grid=[(0, 0)],
            )

        self.assertEqual(result, [])
        self.assertTrue(run_command.call_args.kwargs["check"])


if __name__ == "__main__":
    unittest.main()
