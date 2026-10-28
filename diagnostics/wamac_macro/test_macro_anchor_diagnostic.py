import math
import unittest
from macro_anchor_diagnostic import diagnose, diagnose_record


class MacroTests(unittest.TestCase):
    def setUp(self):
        self.record = dict(split='validation', guide_source='history_only_prediction',
            forecast_origin='2018-01-01', guide_input_max_date='2018-01-01',
            target_date='2018-01-02', lead=1, weighting='uniform_wet_cells',
            base=[2.,4.,8.,10.], truth=[1.,3.,9.,11.], region=[0,0,1,1],
            guide={'0':2.,'1':10.})

    def test_oracle_decomposition_and_real_anchor(self):
        result = diagnose_record(self.record)
        self.assertEqual(result['baseline_mse'], 1.)
        self.assertEqual(result['regional_bias_mse'], 1.)
        self.assertEqual(result['oracle_only_region_mean_replacement_mse'], 0.)
        self.assertEqual(result['full_guide_replacement_mse'], 0.)

    def test_spatial_anomaly_survives_region_correction(self):
        result = diagnose([3,3], [0,2], {0:1}, [0,0])
        self.assertEqual(result['baseline_mse'], 5.)
        self.assertEqual(result['oracle_only_region_mean_replacement_mse'], 1.)
        self.assertEqual(result['full_guide_replacement_mse'], 1.)

    def test_better_than_persistence_is_not_better_than_base(self):
        result = diagnose([1,3], [1,3], {0:2.1}, [0,0])
        self.assertEqual(result['baseline_mse'], 0.)
        self.assertGreater(result['full_guide_replacement_mse'], 0.)

    def test_area_and_cell_weights_are_different_protocols(self):
        uniform = diagnose([0,10], [0,0], {0:0}, [0,0])
        area = diagnose([0,10], [0,0], {0:0}, [0,0], [1,3])
        self.assertEqual(uniform['baseline_mse'], 50.)
        self.assertEqual(area['baseline_mse'], 75.)
        self.assertAlmostEqual(area['baseline_mse'], area['regional_bias_mse']+
                               area['within_region_anomaly_mse'])

    def test_analytical_gain_gradient(self):
        args = ([1,4,8], [2,3,9], {0:2.4,1:8.8}, [0,0,1], [1,2,3])
        h, gain = 1e-6, .4
        scores = diagnose(*args, gains=(gain-h,gain,gain+h))['fixed_gain_scores']
        numerical = (scores[2]['mse']-scores[0]['mse'])/(2*h)
        self.assertAlmostEqual(numerical, scores[1]['derivative_mse_wrt_gain'], places=8)

    def test_land_nan_is_excluded_before_arithmetic(self):
        result = diagnose([1,math.nan], [2,math.nan], {0:2}, [0,-1], [1,math.nan])
        self.assertEqual(result['wet_cells'], 1)
        self.assertEqual(result['full_guide_replacement_mse'], 0.)

    def test_shift_invariance(self):
        a = diagnose([3,4], [1,2], {0:2}, [0,0])
        b = diagnose([276.15,277.15], [274.15,275.15], {0:275.15}, [0,0])
        self.assertAlmostEqual(a['baseline_mse'], b['baseline_mse'])
        self.assertAlmostEqual(a['full_guide_replacement_mse'],b['full_guide_replacement_mse'])

    def test_forbidden_split_and_future_inputs(self):
        for patch in (dict(split='test'), dict(target_date='2022-01-01'),
                      dict(guide_input_max_date='2018-01-02'), dict(lead=8),
                      dict(guide_source='oracle'), dict(weighting='implicit')):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                diagnose_record(dict(self.record, **patch))

    def test_invalid_fields(self):
        for args in (([1],[2,3],{0:2},[0]), ([math.nan],[2],{0:2},[0]),
                     ([1],[2],{0:2},[0],[0]), ([1],[2],{1:2},[0]),
                     ([1],[2],{0:math.inf},[0])):
            with self.subTest(args=args), self.assertRaises(ValueError):
                diagnose(*args)


if __name__ == '__main__':
    unittest.main()
