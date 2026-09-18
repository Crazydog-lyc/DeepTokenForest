"""
test_deep_token_forest.py

Lightweight standard-library tests. No PyTorch, datasets or LM checkpoint needed.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from deep_token_forest import DeepTokenForestClassifier


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


if __name__=="__main__":
    unittest.main()
