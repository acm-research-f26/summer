import unittest
from pathlib import Path

from RAVEN_python.validate_results import find_results_files, validate_results_file


class TestRavenResultsValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.results_dir = Path(__file__).resolve().parents[1] / 'results'

    def test_find_results_files(self) -> None:
        files = find_results_files(self.results_dir)
        self.assertTrue(files, f"No results.json files found in {self.results_dir}")
        for path in files:
            self.assertTrue(path.exists(), f"Expected result file does not exist: {path}")
            self.assertEqual(path.name, 'results.json')

    def test_validate_all_results_files(self) -> None:
        files = find_results_files(self.results_dir)
        for path in files:
            with self.subTest(path=path):
                validate_results_file(path)


if __name__ == '__main__':
    unittest.main()
