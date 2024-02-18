class TestDSGEIntegration(unittest.TestCase):
    def test_full_model_integration(self):
        # Test the entire model process, from loading data to calculating posterior means
        # This test ensures there are no issues when integrating the whole workflow
        samples, is_accepted = run_chain()
        
        # Check if samples have the correct shape and type
        self.assertTrue(isinstance(samples, list))
        self.assertTrue(all(isinstance(sample, tf.Tensor) for sample in samples))
        
        # Check if the model accepted some samples
        self.assertTrue(tf.reduce_any(is_accepted))

        # Calculate posterior means and assert
        alpha_mean = tf.reduce_mean(samples[0]).numpy()
        self.assertNotEqual(alpha_mean, 0) 