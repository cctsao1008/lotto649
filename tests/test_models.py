"""Fresh causal visual model and pattern diversity tests."""
import unittest
import numpy as np
from lotto7x7.core import encode
from lotto7x7.models import ImageForecaster, METHODS, diversity, translate


class DynamicTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(5)
        self.images=np.stack([encode(rng.choice(np.arange(1,50),6,replace=False)) for _ in range(65)])

    def test_translation_preserves_shape_and_six(self):
        first=encode(np.array([1,2,3,8,9,15]))
        for delta in ([2,3],[-7,-8],[20,20]):
            moved=translate(first,np.array(delta))
            self.assertEqual(int(moved.sum()),6)
            self.assertEqual(moved.shape,(7,7))

    def test_all_outputs_legal(self):
        model=ImageForecaster(self.images,train_end=40,window=3)
        for t in range(40,65):
            for name in METHODS:
                result=model.predict(t,name)
                self.assertEqual(int(result.sum()),6)

    def test_target_mutation_cannot_change_its_own_prediction(self):
        model=ImageForecaster(self.images,train_end=40,window=3)
        other=self.images.copy()
        other[40]=encode(np.array([1,2,3,4,5,6]))
        next_model=ImageForecaster(other,train_end=40,window=3)
        for name in METHODS:
            np.testing.assert_array_equal(model.predict(40,name),next_model.predict(40,name))

    def test_diversity_detects_collapse(self):
        x=np.repeat(self.images[:1],20,axis=0)
        result=diversity(x)
        self.assertEqual(result['unique'],1)
        self.assertEqual(result['top_fraction'],1)

    def test_synthetic_four_pattern_cycle(self):
        cycle=np.stack([encode(np.arange(i,i+6)) for i in (1,8,15,22)])
        inputs=np.tile(cycle,(15,1,1))
        model=ImageForecaster(inputs,train_end=40,window=3)
        for t in range(40,60):
            np.testing.assert_array_equal(model.predict(t,'temporal_analog'),inputs[t])

    def test_invalid_models(self):
        model=ImageForecaster(self.images,train_end=40,window=3)
        with self.assertRaises(ValueError): model.predict(30,'previous')
        with self.assertRaises(ValueError): model.predict(40,'old_mlp')


if __name__=='__main__': unittest.main()
