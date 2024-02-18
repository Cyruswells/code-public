
import unittest

class TestDSGEModel(unittest.TestCase):
    def test_joint_log_prob(self):
        # Set a group of test values for the model parameters
        test_c_t = np.array([1, 2, 3], dtype=np.float64)
        test_i_t = np.array([0.5, 1, 1.5], dtype=np.float64)
        test_k_t = np.array([1, 1.1, 1.2], dtype=np.float64)
        test_y_t = test_c_t + test_i_t  # Assume y_t = c_t + i_t for testing
        
        # Calculate the joint log probability
        result = joint_log_prob(
            tf.convert_to_tensor(test_c_t),
            tf.convert_to_tensor(test_i_t),
            tf.convert_to_tensor(test_k_t),
            tf.convert_to_tensor(test_y_t),
            tf.convert_to_tensor(0.3),
            tf.convert_to_tensor(25.0),  
            tf.convert_to_tensor(0.5)
        )
        # Check if the result is a TensorFlow tensor
        self.assertTrue(isinstance(result, tf.Tensor))
        
        # Check if a numeric result was obtained
        self.assertFalse(tf.math.is_nan(result))
        
if __name__ == '__main__':
    unittest.main()
