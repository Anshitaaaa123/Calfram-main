import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import numpy as np
from calfram.calibration_framework import CalibrationFramework
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.StreamHandler(), logging.FileHandler("test_log.txt")])


class TestCalibrationFramework(unittest.TestCase):
    def setUp(self):
        logging.info("Setting up the test.")
        self.cf = CalibrationFramework()
        
        # Create some sample data for testing
        np.random.seed(42)
        self.y_true = np.random.randint(0, 2, 1000)
        self.y_prob = np.random.random((1000, 2))
        self.y_prob = self.y_prob / self.y_prob.sum(axis=1, keepdims=True)
        self.y_pred = np.argmax(self.y_prob, axis=1)
        logging.info("Sample data created.")

    def test_select_probability(self):
        logging.info("Running test_select_probability.")
        classes_scores = self.cf.select_probability(self.y_true, self.y_prob, self.y_pred)
        
        self.assertIn('0', classes_scores)
        self.assertIn('1', classes_scores)
        self.assertEqual(len(classes_scores['0']['proba']), 1000)
        self.assertEqual(len(classes_scores['0']['y']), 1000)
        self.assertEqual(classes_scores['0']['y_one_hot_nclass'].shape, (1000, 2))
        self.assertEqual(classes_scores['0']['y_prob_one_hotnclass'].shape, (1000, 2))
        self.assertEqual(classes_scores['0']['y_pred_one_hotnclass'].shape, (1000, 2))
        logging.info("test_select_probability passed.")

    def test_calibrationdiagnosis(self):
        logging.info("Running test_calibrationdiagnosis.")
        classes_scores = self.cf.select_probability(self.y_true, self.y_prob, self.y_pred)
        measures, binning_dict = self.cf.calibrationdiagnosis(classes_scores)
        
        self.assertIn('0', measures)
        self.assertIn('1', measures)
        self.assertIn('ece_acc', measures['0'])
        self.assertIn('ece_fp', measures['0'])
        self.assertIn('ec_g', measures['0'])
        self.assertIn('brier_loss', measures['0'])
        
        self.assertIn('0', binning_dict)
        self.assertIn('1', binning_dict)
        self.assertIn('bins', binning_dict['0'])
        self.assertIn('binids', binning_dict['0'])
        self.assertIn('binfr', binning_dict['0'])
        logging.info("test_calibrationdiagnosis passed.")

    def test_classwise_calibration(self):
        logging.info("Running test_classwise_calibration.")
        classes_scores = self.cf.select_probability(self.y_true, self.y_prob, self.y_pred)
        measures, _ = self.cf.calibrationdiagnosis(classes_scores)
        class_wise_metrics = self.cf.classwise_calibration(measures)
        
        self.assertIn('ec_g', class_wise_metrics)
        self.assertIn('ec_dir', class_wise_metrics)
        self.assertIn('ece_freq', class_wise_metrics)
        self.assertIn('ece_acc', class_wise_metrics)
        self.assertIn('ec_underconf', class_wise_metrics)
        self.assertIn('ec_overconf', class_wise_metrics)
        self.assertIn('brierloss', class_wise_metrics)
        logging.info("test_classwise_calibration passed.")

    def test_end_points(self):
        logging.info("Running test_end_points.")
        x = np.array([0.1, 0.5, 0.9])
        y = np.array([0.2, 0.6, 0.8])
        result = self.cf.end_points(x, y)
        
        self.assertEqual(result.shape, (4, 2))
        np.testing.assert_array_almost_equal(result[0], [0, 0])
        np.testing.assert_array_almost_equal(result[-1], [0.9, 0.8])
        logging.info("test_end_points passed.")

    def test_add_tilde(self):
        logging.info("Running test_add_tilde.")
        pts = np.array([[0, 0], [0.5, 0.6], [1, 1]])
        result = self.cf.add_tilde(pts)
        
        self.assertEqual(result.shape, (3, 2))
        np.testing.assert_array_almost_equal(result[1], [0.5, 0.5])
        logging.info("test_add_tilde passed.")

    def test_h_triangle(self):
        logging.info("Running test_h_triangle.")
        new_pts = np.array([[0, 0], [0.5, 0.6], [1, 1]])
        tilde = np.array([[0, 0], [0.5, 0.5], [1, 1]])
        result = self.cf.h_triangle(new_pts, tilde)
        
        self.assertEqual(result.shape, (2,))
        self.assertGreater(result[0], 0)
        logging.info("test_h_triangle passed.")

    def test_underbelow_line(self):
        logging.info("Running test_underbelow_line.")
        pts = np.array([[0, 0], [0.4, 0.3], [0.6, 0.7], [1, 1]])
        result = self.cf.underbelow_line(pts)
        
        self.assertEqual(len(result), 4)
        self.assertEqual(result[1], 'right')
        self.assertEqual(result[2], 'left')
        logging.info("test_underbelow_line passed.")

    def test_split_probabilities(self):
        logging.info("Running test_split_probabilities.")
        probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        prob_ranges, bin_edges = self.cf.split_probabilities(probs, 3)
        
        self.assertEqual(len(prob_ranges), 3)
        self.assertEqual(len(bin_edges), 3)
        self.assertAlmostEqual(bin_edges[-1], 0.9)
        logging.info("test_split_probabilities passed.")

    def test_compute_equal_mass_bin_heights(self):
        logging.info("Running test_compute_equal_mass_bin_heights.")
        data = [(0.1, 0), (0.3, 1), (0.5, 1), (0.7, 0), (0.9, 1)]
        result = self.cf.compute_equal_mass_bin_heights(data, 2)
        
        self.assertEqual(len(result), 2)
        self.assertGreaterEqual(result[0], 0)
        self.assertLessEqual(result[0], 1)
        logging.info("test_compute_equal_mass_bin_heights passed.")

    def test_is_monotonic(self):
        logging.info("Running test_is_monotonic.")
        self.assertTrue(self.cf.is_monotonic([0.1, 0.3, 0.5, 0.7]))
        self.assertFalse(self.cf.is_monotonic([0.1, 0.5, 0.3, 0.7]))
        logging.info("test_is_monotonic passed.")

    def test_monotonic_sweep_calibration(self):
        logging.info("Running test_monotonic_sweep_calibration.")
        data = [(0.1, 0), (0.3, 1), (0.5, 1), (0.7, 0), (0.9, 1)]
        result = self.cf.monotonic_sweep_calibration(data, 5)
        
        self.assertIsInstance(result, int)
        self.assertGreaterEqual(result, 2)
        self.assertLessEqual(result, 5)
        logging.info("test_monotonic_sweep_calibration passed.")

if __name__ == '__main__':
    unittest.main()