import unittest
import numpy as np
from code.Linear_Regression_Analytic import Linear_Regression_Analytic
import numpy as np



class myTest(unittest.TestCase):
	X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
	y = np.dot(X, np.array([1, 2])) + 3
	Linear_Model_Analytic = Linear_Regression_Analytic()
	Linear_Model_Analytic.fit(X,y)

	def test_weight(self):
		self.assertEqual(self.Linear_Model_Analytic.W[0], [3.])
		self.assertEqual(self.Linear_Model_Analytic.W[1], [1.])
		self.assertEqual(self.Linear_Model_Analytic.W[2], [2.])

	def test_predictions(self):
		self.assertEqual(self.Linear_Model_Analytic.predict([[0,3]]), [9.])

if __name__ == '__main__':
    unittest.main()