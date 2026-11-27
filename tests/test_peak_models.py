"""Check validation threshold selection and isolation of model inputs."""

import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from scripts.build_peak_prediction_dataset import FEATURES
from scripts.train_peak_models import (best_threshold, build_pipeline, fit_candidate,
                                      metrics, predictor_columns)


class PeakModelsTest(unittest.TestCase):
    def frame(self):
        rows = []
        for i in range(24):
            row = {field: i % 5 for field in FEATURES}
            row.update(domain="a" if i < 12 else "b", topic="known", topic_type="event",
                       peak_next_7d=i % 2, page_id=str(i), qid="Q1", title="ignored")
            rows.append(row)
        return pd.DataFrame(rows)

    def test_threshold_matches_exhaustive_f1_and_highest_tie(self):
        y = np.array([0, 1, 1, 0, 1, 0, 1])
        probabilities = np.array([0.1, 0.2, 0.2, 0.4, 0.6, 0.8, 0.9])
        thresholds = np.unique(probabilities)
        scores = [f1_score(y, probabilities >= t) for t in thresholds]
        expected = max(t for t, score in zip(thresholds, scores) if np.isclose(score, max(scores)))
        self.assertEqual(best_threshold(y, probabilities), expected)
        self.assertAlmostEqual(metrics(y, probabilities, expected)["f1"], max(scores))
        self.assertEqual(best_threshold(np.array([1, 0, 0, 1]),
                                        np.array([0.1, 0.2, 0.3, 0.4])), 0.4)

    def test_only_declared_features_and_training_categories_are_used(self):
        train = self.frame()
        validation = train.copy()
        validation["topic"] = "unseen"
        model = build_pipeline("random_forest", {"n_estimators": 5, "max_depth": 2}, False, "global")
        model.fit(train, train["peak_next_7d"])
        encoder = model.named_steps["preprocessing"].named_transformers_["categorical"]
        self.assertNotIn("unseen", encoder.categories_[1])
        probabilities = model.predict_proba(validation)
        validation["peak_next_7d"] = 1 - validation["peak_next_7d"]
        validation["page_id"] = "changed"
        validation["forecast_day_of_week"] = 999
        validation["target_week_of_year"] = 999
        np.testing.assert_array_equal(probabilities, model.predict_proba(validation))
        numeric, categorical = predictor_columns(False, "local")
        self.assertNotIn("domain", categorical)
        self.assertNotIn("target_week_of_year", numeric)

    def test_local_models_route_each_domain_and_reject_unknown_domains(self):
        train = self.frame()
        validation = train.iloc[::-1].reset_index(drop=True)
        models, probabilities = fit_candidate(train, validation, "random_forest",
                                              {"n_estimators": 5, "max_depth": 2}, True, "local")
        for i, row in validation.iterrows():
            expected = models[row["domain"]].predict_proba(validation.iloc[[i]])[0, 1]
            self.assertEqual(probabilities[i], expected)
        validation.loc[0, "domain"] = "new"
        with self.assertRaises(ValueError):
            fit_candidate(train, validation, "random_forest", {"n_estimators": 5}, False, "local")


if __name__ == "__main__":
    unittest.main()
