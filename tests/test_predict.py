import unittest
from unittest import mock

import numpy as np

from car_price import config
from car_price.predict import CarSpec, apply_interval, load_model, predict_price, validate_spec

FAKE_META = {
    "catalogue": {
        "brand_to_models": {"BMW": ["320", "X5"], "Audi": ["A4"]},
        "numeric_ranges": {
            "Year": {"min": 1969, "max": 2016},
            "Mileage": {"min": 0, "max": 980, "p99": 440.0},
            "EngineV": {"min": 0.6, "max": 6.5},
        },
    }
}


def spec(**overrides):
    base = dict(
        brand="BMW", model="320", body="sedan", engine_type="Petrol",
        registration="yes", year=2008, mileage=150, engine_v=2.0,
    )
    base.update(overrides)
    return CarSpec(**base)


class IntervalMaths(unittest.TestCase):
    def test_interval_brackets_the_estimate(self):
        lower, upper = apply_interval(10_000.0, -0.25, 0.25)
        self.assertLess(lower, 10_000.0)
        self.assertGreater(upper, 10_000.0)

    def test_interval_scales_with_price(self):
        lo1, up1 = apply_interval(5_000.0, -0.2, 0.2)
        lo2, up2 = apply_interval(50_000.0, -0.2, 0.2)
        self.assertGreater(up2 - lo2, up1 - lo1)
        self.assertAlmostEqual(up1 / 5_000, up2 / 50_000, places=2)

    def test_zero_offsets_return_the_estimate(self):
        lower, upper = apply_interval(8_000.0, 0.0, 0.0)
        self.assertAlmostEqual(float(lower), 8_000.0, places=6)
        self.assertAlmostEqual(float(upper), 8_000.0, places=6)

    def test_works_on_arrays(self):
        lower, upper = apply_interval(np.array([1000.0, 2000.0]), -0.1, 0.1)
        self.assertEqual(lower.shape, (2,))
        self.assertTrue(np.all(lower < upper))


class InputValidation(unittest.TestCase):
    def test_normal_input_has_no_warnings(self):
        self.assertEqual(validate_spec(spec(), FAKE_META), [])

    def test_negative_mileage_rejected(self):
        with self.assertRaisesRegex(ValueError, "Mileage"):
            validate_spec(spec(mileage=-1), FAKE_META)

    def test_non_positive_engine_rejected(self):
        with self.assertRaisesRegex(ValueError, "Engine"):
            validate_spec(spec(engine_v=0), FAKE_META)

    def test_nan_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_spec(spec(mileage=float("nan")), FAKE_META)

    def test_blank_category_rejected(self):
        with self.assertRaisesRegex(ValueError, "brand"):
            validate_spec(spec(brand="  "), FAKE_META)

    def test_year_outside_training_range_warns(self):
        warnings = validate_spec(spec(year=1950), FAKE_META)
        self.assertTrue(any("extrapolating" in w for w in warnings))

    def test_unknown_model_warns(self):
        warnings = validate_spec(spec(model="Z4"), FAKE_META)
        self.assertTrue(any("Z4" in w for w in warnings))

    def test_unknown_brand_warns(self):
        warnings = validate_spec(spec(brand="Ferrari"), FAKE_META)
        self.assertTrue(any("Ferrari" in w for w in warnings))

    def test_very_high_mileage_warns(self):
        warnings = validate_spec(spec(mileage=900), FAKE_META)
        self.assertTrue(any("Mileage" in w for w in warnings))


@unittest.skipUnless(config.MODEL_PATH.exists() and config.METADATA_PATH.exists(), "run python -m car_price.train first")
class ShippedArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.meta = load_model()

    def test_prediction_is_finite_positive_and_inside_its_range(self):
        r = predict_price(self.model, self.meta, spec())
        self.assertTrue(np.isfinite(r["estimate"]))
        self.assertGreater(r["estimate"], 0)
        self.assertLess(r["lower"], r["estimate"])
        self.assertGreater(r["upper"], r["estimate"])

    def test_newer_car_costs_more_than_older_identical_car(self):
        old = predict_price(self.model, self.meta, spec(year=2000))["estimate"]
        new = predict_price(self.model, self.meta, spec(year=2014))["estimate"]
        self.assertGreater(new, old)

    def test_higher_mileage_does_not_raise_price(self):
        low = predict_price(self.model, self.meta, spec(year=2010, mileage=40))["estimate"]
        high = predict_price(self.model, self.meta, spec(year=2010, mileage=350))["estimate"]
        self.assertGreater(low, high)

    def test_luxury_brand_costs_more_than_economy_brand(self):
        merc = predict_price(self.model, self.meta, spec(brand="Mercedes-Benz", model="E-Class", year=2010))
        renault = predict_price(self.model, self.meta, spec(brand="Renault", model="Megane", year=2010))
        self.assertGreater(merc["estimate"], renault["estimate"])

    def test_unknown_model_still_predicts_with_warning(self):
        r = predict_price(self.model, self.meta, spec(model="Never Seen Before"))
        self.assertTrue(np.isfinite(r["estimate"]))
        self.assertTrue(r["warnings"])

    def test_metadata_selected_the_lowest_cv_model_not_the_best_test_model(self):
        rows = [r for r in self.meta["comparison"] if "Baseline" not in r["model"]]
        by_cv = min(rows, key=lambda r: r["cv_rmsle"])["model"]
        self.assertEqual(self.meta["selection"]["best_model"], by_cv)

    def test_model_beats_the_naive_baseline_by_a_wide_margin(self):
        best, base = self.meta["test_metrics"], self.meta["baseline_test_metrics"]
        self.assertLess(best["rmsle"], 0.5 * base["rmsle"])
        self.assertGreater(best["r2"], 0.7)

    def test_interval_coverage_is_close_to_nominal(self):
        iv = self.meta["interval"]
        self.assertLess(abs(iv["empirical_test_coverage"] - iv["nominal_coverage"]), 0.08)


@unittest.skipUnless(config.MODEL_PATH.exists() and config.METADATA_PATH.exists(), "run python -m car_price.train first")
class FallbackWhenPickleIsUnusable(unittest.TestCase):
    """If the saved model cannot be unpickled (e.g. a scikit-learn upgrade on the
    deployment server), load_model() must refit from the CSV instead of crashing."""

    def test_refit_fallback_gives_the_same_predictions(self):
        saved_model, meta = load_model()
        with mock.patch("car_price.predict.joblib.load", side_effect=RuntimeError("version mismatch")):
            refit_model, _ = load_model()
        a = predict_price(saved_model, meta, spec())["estimate"]
        b = predict_price(refit_model, meta, spec())["estimate"]
        self.assertAlmostEqual(a, b, delta=0.01 * a)  # same data + params + seed => ~identical


if __name__ == "__main__":
    unittest.main()
