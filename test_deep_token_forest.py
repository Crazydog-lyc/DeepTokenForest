"""
test_deep_token_forest.py

Lightweight standard-library tests. No PyTorch, datasets or LM checkpoint needed.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from deep_token_forest import (
    DeepTokenForestClassifier,
    fit_direction_geometry_from_doc_mean,
    ProjectionDistributionNewtonTree,
    TokenConcept,
    fit_concept_standardization,
    _concept_activation_block,
)


class DeepTokenForestTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(123)
        N,T,D=260,8,5
        self.X=rng.normal(size=(N,T,D)).astype(np.float32)
        self.M=np.ones((N,T),dtype=np.uint8)
        score=(self.X[:,:,0]>0.7).mean(1)+0.8*(self.X[:,:,1]<-0.6).mean(1)
        self.y=(score>np.median(score)).astype(np.int64)

    def _model(self,prefix=False,honest=0.0):
        return DeepTokenForestClassifier(
            token_dim=self.X.shape[2],
            output_dim=2,
            n_stages=2,
            trees_per_stage=2,
            learning_rate=0.15,
            max_depth=3,
            max_leaves=4,
            min_samples_leaf=10,
            n_random_directions=2,
            n_token_prototype_directions=2,
            n_local_perturbations=1,
            local_sigmas=(0.15,),
            beam_width=2,
            honest_fraction=honest,
            honest_top_candidates=3,
            concepts_per_stage=3,
            prefix_mixing=prefix,
            token_threshold_sample_rows=128,
            score_block_rows=128,
            direction_chunk_size=2,
            transform_block_rows=64,
            direction_geometry="covariance",
            direction_geometry_rcond=1e-5,
            random_state=123,
        )

    def test_fit_predict_growth(self):
        tr=np.arange(200)
        va=np.arange(200,260)
        with tempfile.TemporaryDirectory() as d:
            m=self._model(prefix=False)
            m.fit(
                self.X[tr],self.M[tr],self.y[tr],
                eval_set=(self.X[va],self.M[va],self.y[va]),
                work_dir=Path(d)/"fit",
                verbose=False,
            )
            logits=m.predict_logits(
                self.X[va],self.M[va],work_dir=Path(d)/"pred"
            )
        self.assertEqual(logits.shape,(len(va),2))
        self.assertTrue(np.all(np.isfinite(logits)))
        self.assertEqual(m.final_token_dim_,self.X.shape[2]+3)
        self.assertEqual(len(m.history_),4)

    def test_prefix_growth_dimension(self):
        tr=np.arange(200)
        va=np.arange(200,260)
        with tempfile.TemporaryDirectory() as d:
            m=self._model(prefix=True)
            m.fit(
                self.X[tr],self.M[tr],self.y[tr],
                eval_set=(self.X[va],self.M[va],self.y[va]),
                work_dir=Path(d)/"fit",
                verbose=False,
            )
            logits=m.predict_logits(
                self.X[va],self.M[va],work_dir=Path(d)/"pred"
            )
        self.assertTrue(np.all(np.isfinite(logits)))
        self.assertEqual(m.final_token_dim_,self.X.shape[2]+2*3)

    def test_honest_search_runs(self):
        tr=np.arange(220)
        va=np.arange(220,260)
        with tempfile.TemporaryDirectory() as d:
            m=self._model(prefix=False,honest=0.15)
            m.fit(
                self.X[tr],self.M[tr],self.y[tr],
                eval_set=(self.X[va],self.M[va],self.y[va]),
                work_dir=Path(d)/"fit",
                verbose=False,
            )
        gains=[]
        for stage in m.stages_:
            for tree in stage["trees"]:
                gains.extend(
                    [
                        n["effective_gain"]
                        for n in tree.internal_nodes()
                    ]
                )
        self.assertTrue(all(np.isfinite(g) for g in gains))

    def test_covariance_geometry_exact_reparameterization(self):
        rng = np.random.default_rng(11)
        Z = rng.normal(size=(4000, 2)).astype(np.float32)
        A = np.asarray(
            [[10.0, 0.0, 1.0, 1.0, 0.0],
             [0.0, 1.0, 0.0, 0.0, 0.2]],
            dtype=np.float32,
        )
        Z2 = Z @ A
        r = (
            0.7 * Z[:, 0] - 0.4 * Z[:, 1]
        ).astype(np.float32)

        g1 = fit_direction_geometry_from_doc_mean(
            Z, mode="covariance", rcond=1e-7
        )
        g2 = fit_direction_geometry_from_doc_mean(
            Z2, mode="covariance", rcond=1e-7
        )

        c1 = (Z - g1.mean[None, :]).T @ r
        v1 = g1.whitener.T @ c1
        w1 = g1.whitener @ v1

        c2 = (Z2 - g2.mean[None, :]).T @ r
        v2 = g2.whitener.T @ c2
        w2 = g2.whitener @ v2

        q1 = Z @ w1
        q2 = Z2 @ w2
        corr = np.corrcoef(q1, q2)[0, 1]
        self.assertGreater(corr, 0.99999)

    def test_concept_activation_positive_scale_invariant(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(300, 6, 4)).astype(np.float32)
        M = np.ones((300, 6), dtype=np.uint8)
        w = np.asarray([0.7, -0.3, 0.2, 0.5], dtype=np.float32)
        b = 0.15
        c1 = TokenConcept(w=w.copy(), token_threshold=b, temperature=0.5)
        c2 = TokenConcept(
            w=(7.0 * w).copy(),
            token_threshold=7.0 * b,
            temperature=0.5,
        )
        fit_concept_standardization(X, M, [c1], random_state=1)
        fit_concept_standardization(X, M, [c2], random_state=1)
        a1 = _concept_activation_block(X[:50], M[:50], [c1])
        a2 = _concept_activation_block(X[:50], M[:50], [c2])
        self.assertTrue(np.allclose(a1, a2, atol=2e-5, rtol=2e-5))

    def test_tree_pipeline_duplicate_scale_invariant(self):
        rng = np.random.default_rng(1234)
        N, T = 500, 10
        X = rng.normal(size=(N, T, 2)).astype(np.float32)
        M = np.ones((N, T), dtype=np.uint8)
        frac = ((X[..., 0] + 0.8 * X[..., 1]) > 0.3).mean(axis=1)
        y = (frac > np.median(frac)).astype(np.int64)
        A = np.asarray(
            [[10.0, 0.0, 1.0, 1.0, 0.0],
             [0.0, 1.0, 0.0, 0.0, 0.2]],
            dtype=np.float32,
        )
        X2 = X @ A

        def model(d):
            return DeepTokenForestClassifier(
                token_dim=d,
                output_dim=2,
                n_stages=1,
                trees_per_stage=1,
                learning_rate=0.1,
                max_depth=3,
                max_leaves=4,
                min_samples_leaf=20,
                n_random_directions=0,
                n_token_prototype_directions=0,
                n_local_perturbations=0,
                beam_width=4,
                honest_fraction=0.0,
                concepts_per_stage=0,
                feature_growth=False,
                token_threshold_sample_rows=500,
                max_doc_thresholds=64,
                direction_geometry="covariance",
                direction_geometry_rcond=1e-7,
                tree_device="cpu",
                threshold_n_jobs=1,
                random_state=9,
            )

        with tempfile.TemporaryDirectory() as d:
            m1, m2 = model(2), model(5)
            m1.fit(X, M, y, work_dir=Path(d) / "a", verbose=False)
            m2.fit(X2, M, y, work_dir=Path(d) / "b", verbose=False)
            p1 = m1.predict_logits(X, M)
            p2 = m2.predict_logits(X2, M)

        self.assertTrue(np.array_equal(p1, p2))


if __name__=="__main__":
    unittest.main()
