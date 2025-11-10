"""Unit tests for preprocessing utilities defined in src.utils.config."""
import unittest

from src.utils.config import DataSourceConfig, extract_template_fields


class DataSourceConfigTest(unittest.TestCase):
    def test_valid_name_source(self) -> None:
        cfg = DataSourceConfig(name="dataset/name", split="train", weight=0.5)
        self.assertEqual(cfg.name, "dataset/name")
        self.assertIsNone(cfg.path)
        self.assertEqual(cfg.weight, 0.5)

    def test_valid_path_source(self) -> None:
        cfg = DataSourceConfig(path="/tmp/data", split="validation", weight=2.0, max_samples=100)
        self.assertEqual(cfg.path, "/tmp/data")
        self.assertIsNone(cfg.name)
        self.assertEqual(cfg.max_samples, 100)

    def test_requires_name_or_path(self) -> None:
        with self.assertRaises(ValueError):
            DataSourceConfig()

    def test_disallows_name_and_path(self) -> None:
        with self.assertRaises(ValueError):
            DataSourceConfig(name="foo", path="/tmp")

    def test_requires_positive_weight(self) -> None:
        with self.assertRaises(ValueError):
            DataSourceConfig(name="foo", weight=0)

    def test_requires_positive_max_samples(self) -> None:
        with self.assertRaises(ValueError):
            DataSourceConfig(name="foo", max_samples=0)


class TemplateFieldExtractionTest(unittest.TestCase):
    def test_extracts_sorted_unique_fields(self) -> None:
        template = "User: {question}\n\nAssistant: {answer} {answer}"
        self.assertEqual(extract_template_fields(template), ["answer", "question"])

    def test_returns_empty_for_no_fields(self) -> None:
        self.assertEqual(extract_template_fields("No placeholders"), [])


if __name__ == "__main__":
    unittest.main()
