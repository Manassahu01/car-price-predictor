import unittest

import numpy as np
import pandas as pd

from car_price import config, data


def make_frame(n=6):
    return pd.DataFrame(
        {
            "Brand": ["BMW", "Audi", "Toyota", "BMW", "Audi", "Toyota"][:n],
            "Price": [5000.0, 7000.0, 9000.0, 5000.0, np.nan, 3000.0][:n],
            "Body": ["sedan"] * n,
            "Mileage": [100, 150, 200, 100, 50, 300][:n],
            "EngineV": [2.0, 1.6, np.nan, 2.0, 2.5, 99.99][:n],
            "Engine Type": ["Petrol"] * n,
            "Registration": ["yes"] * n,
            "Year": [2005, 2010, 2012, 2005, 2014, 1999][:n],
            "Model": ["320", "A4", "Corolla", "320", "A6", "Camry"][:n],
        }
    )


class SchemaValidation(unittest.TestCase):
    def test_valid_frame_passes(self):
        data.validate_schema(make_frame())

    def test_missing_column_is_reported_by_name(self):
        with self.assertRaisesRegex(ValueError, "Model"):
            data.validate_schema(make_frame().drop(columns=["Model"]))

    def test_non_numeric_price_rejected(self):
        df = make_frame()
        df["Price"] = df["Price"].astype(str)
        with self.assertRaisesRegex(ValueError, "Price"):
            data.validate_schema(df)

    def test_empty_frame_rejected(self):
        with self.assertRaises(ValueError):
            data.validate_schema(make_frame().iloc[0:0])

    def test_missing_file_gives_helpful_error(self):
        with self.assertRaises(FileNotFoundError):
            data.load_raw("does/not/exist.csv")


class Cleaning(unittest.TestCase):
    def setUp(self):
        self.clean, self.report = data.prepare_training_data(make_frame())

    def test_exact_duplicates_removed(self):
        # rows 0 and 3 are identical
        self.assertEqual(self.report["duplicates_removed"], 1)

    def test_missing_price_removed(self):
        self.assertEqual(self.report["missing_or_invalid_price_removed"], 1)
        self.assertFalse(self.clean[config.TARGET].isna().any())

    def test_missing_and_invalid_engine_rows_are_kept_not_dropped(self):
        # NaN EngineV (Corolla) and 99.99 EngineV (Camry) must survive; the pipeline handles them
        self.assertIn("Corolla", self.clean["Model"].tolist())
        self.assertIn("Camry", self.clean["Model"].tolist())
        self.assertEqual(self.report["engine_volume_missing_kept"], 1)
        self.assertEqual(self.report["engine_volume_invalid_kept"], 1)

    def test_row_accounting_adds_up(self):
        r = self.report
        self.assertEqual(
            r["rows_raw"] - r["duplicates_removed"] - r["missing_or_invalid_price_removed"],
            r["rows_final"],
        )

    def test_zero_price_is_invalid(self):
        df = make_frame()
        df.loc[0, "Price"] = 0.0
        _, report = data.prepare_training_data(df)
        self.assertGreaterEqual(report["missing_or_invalid_price_removed"], 2)


class RealDataset(unittest.TestCase):
    def test_shipped_dataset_loads_with_expected_shape(self):
        df = data.load_raw()
        self.assertGreater(len(df), 4000)
        for col in config.RAW_FEATURES + [config.TARGET]:
            self.assertIn(col, df.columns)

    def test_split_xy_has_no_target_leak(self):
        clean, _ = data.prepare_training_data(data.load_raw())
        X, y = data.split_xy(clean)
        self.assertNotIn(config.TARGET, X.columns)
        self.assertEqual(len(X), len(y))


if __name__ == "__main__":
    unittest.main()
