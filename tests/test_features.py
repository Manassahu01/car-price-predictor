import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from car_price import config, data, features


def sample_inputs(**overrides):
    row = {
        "Brand": "BMW", "Model": "320", "Body": "sedan", "Engine Type": "Petrol",
        "Registration": "yes", "Mileage": 150.0, "EngineV": 2.0, "Year": 2008,
    }
    row.update(overrides)
    return pd.DataFrame([row])[config.RAW_FEATURES]


class EngineerFeatures(unittest.TestCase):
    def test_output_columns_and_order(self):
        out = features.engineer_features(sample_inputs())
        self.assertEqual(list(out.columns), config.ENGINEERED_NUMERIC + config.ENGINEERED_CATEGORICAL)

    def test_car_age_is_measured_from_reference_year(self):
        out = features.engineer_features(sample_inputs(Year=2006))
        self.assertEqual(out.loc[0, "car_age"], config.REFERENCE_YEAR - 2006)

    def test_future_year_gives_zero_age_not_negative(self):
        out = features.engineer_features(sample_inputs(Year=2020))
        self.assertEqual(out.loc[0, "car_age"], 0)

    def test_mileage_per_year_is_finite_for_brand_new_car(self):
        out = features.engineer_features(sample_inputs(Year=config.REFERENCE_YEAR, Mileage=10))
        self.assertTrue(np.isfinite(out.loc[0, "mileage_per_year"]))

    def test_implausible_engine_volume_becomes_missing(self):
        for bad in (99.99, 15.0, 0.0, -1.0):
            out = features.engineer_features(sample_inputs(EngineV=bad))
            self.assertTrue(np.isnan(out.loc[0, "EngineV"]), f"{bad} should be NaN")

    def test_valid_engine_volume_is_untouched(self):
        out = features.engineer_features(sample_inputs(EngineV=3.0))
        self.assertEqual(out.loc[0, "EngineV"], 3.0)

    def test_negative_mileage_becomes_missing(self):
        out = features.engineer_features(sample_inputs(Mileage=-5))
        self.assertTrue(np.isnan(out.loc[0, "Mileage"]))

    def test_input_frame_is_not_mutated(self):
        df = sample_inputs(EngineV=99.99)
        features.engineer_features(df)
        self.assertEqual(df.loc[0, "EngineV"], 99.99)


class PipelineBehaviour(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        clean, _ = data.prepare_training_data(data.load_raw())
        X, y = data.split_xy(clean)
        cls.X, cls.y = X.iloc[:800], y.iloc[:800]
        cls.model = features.build_estimator(Ridge(alpha=1.0)).fit(cls.X, cls.y)

    def test_prediction_is_in_price_units_and_positive(self):
        pred = self.model.predict(sample_inputs())
        self.assertEqual(pred.shape, (1,))
        self.assertGreater(pred[0], 100)  # prices are in the thousands, not log units

    def test_missing_engine_volume_does_not_crash(self):
        pred = self.model.predict(sample_inputs(EngineV=np.nan))
        self.assertTrue(np.isfinite(pred[0]))

    def test_unseen_model_and_brand_do_not_crash(self):
        pred = self.model.predict(sample_inputs(Brand="Ferrari", Model="Totally New Model"))
        self.assertTrue(np.isfinite(pred[0]))

    def test_categoricals_are_one_hot_encoded_not_label_encoded(self):
        names = self.model.regressor_.named_steps["preprocess"].get_feature_names_out()
        self.assertTrue(any(n.startswith("cat__Brand_") for n in names))
        self.assertGreater(len(names), len(config.RAW_FEATURES) * 2)

    def test_imputer_statistics_come_from_training_data_only(self):
        pre = self.model.regressor_.named_steps["preprocess"]
        imputer = pre.named_transformers_["num"].named_steps["impute"]
        engineered = features.engineer_features(self.X)
        expected = engineered["EngineV"].median()
        engine_idx = config.ENGINEERED_NUMERIC.index("EngineV")
        self.assertAlmostEqual(imputer.statistics_[engine_idx], expected)

    def test_batch_prediction_matches_single_prediction(self):
        batch = self.model.predict(self.X.iloc[:5])
        singles = [self.model.predict(self.X.iloc[[i]])[0] for i in range(5)]
        np.testing.assert_allclose(batch, singles)


if __name__ == "__main__":
    unittest.main()
