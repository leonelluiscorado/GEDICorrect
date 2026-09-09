import unittest
from unittest.mock import mock_open, patch

from gedicorrect.system import _cgroup_cpu_limit, available_cpu_count


class SystemTests(unittest.TestCase):
    @patch("builtins.open", mock_open(read_data="250000 100000\n"))
    def test_reads_cgroup_v2_cpu_quota(self):
        self.assertEqual(_cgroup_cpu_limit(), 2)

    @patch("gedicorrect.system._cgroup_cpu_limit", return_value=3)
    @patch("gedicorrect.system.os.cpu_count", return_value=16)
    @patch("gedicorrect.system.os.sched_getaffinity", return_value=set(range(8)))
    def test_available_cpu_count_uses_smallest_limit(self, affinity, cpu_count, cgroup_limit):
        self.assertEqual(available_cpu_count(), 3)


if __name__ == "__main__":
    unittest.main()
