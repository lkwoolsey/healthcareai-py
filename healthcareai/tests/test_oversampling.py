import unittest

import pandas as pd
import numpy as np

import healthcareai as hc


class TestOverSampling(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame([
                  [1.3, 3.4, 'one'],
                  [1.5, 3.3, 'one'],
                  [1.4, 3.5, 'one'],
                  [1.4, 3.5, 'two'],
                  [1.7, 3.6, 'one'],
                  [1.2, 3.5, 'three'],            
                  [1.7, 3.6, 'two']  ])
        self.df.columns = ['x1','x2','y']
        
    def test_number_of_rows_returned(self):
        expected_rows = max(self.df.y.value_counts()) * len(self.df.y.unique())
        oo = hc.DevelopSupervisedModel(dataframe = self.df,
                                       model_type = 'classification',
                                       predicted_column = 'y')
        oo.over_sampling()
        self.assertEqual(expected_rows, oo.dataframe.shape[0])

    def test_dataframe_type_does_not_change(self):
        oo = hc.DevelopSupervisedModel(dataframe = self.df,
                                       model_type = 'classification',
                                       predicted_column = 'y')
        oo.over_sampling()
        self.assertEqual(type(self.df), type(oo.dataframe))

    def test_returns_correct_column_names(self):
        oo = hc.DevelopSupervisedModel(dataframe = self.df,
                                       model_type = 'classification',
                                       predicted_column = 'y')
        oo.over_sampling()
        self.assertEqual(
            sum(self.df.columns == oo.dataframe.columns),
            len(self.df.columns)  )

    def test_returns_balanced_targets(self):
        oo = hc.DevelopSupervisedModel(dataframe = self.df,
                                       model_type = 'classification',
                                       predicted_column = 'y')
        oo.over_sampling()
        self.assertEqual(max(oo.dataframe.y.value_counts()),
                         min(oo.dataframe.y.value_counts()) )
        
    def tearDown(self):
        del self.df

        
if __name__ == '__main__':
    unittest.main()
