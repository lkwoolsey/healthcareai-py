import unittest

import pandas as pd
import numpy as np

import healthcareai as hc


class TestFeatureScaling(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame([
                  [1.3, 3.4, 'one',1],
                  [1.5, 3.3, 'one',1],
                  [1.4, 3.5, 'one',1],
                  [1.4, 3.5, 'two',0],
                  [1.7, 3.6, 'one',0],
                  [1.2, 3.5, 'three',1],            
                  [1.7, 3.6, 'two',1],
                  [1.5, 3.7, 'three',0],
                  [1.3, 3.3, 'two',1],
                  [1.4, 3.3, 'one',0]   ])
        self.df.columns = ['x1','x2','x3','y']
        
    def test_X_train_type(self):
        oo = hc.DevelopSupervisedModel(dataframe = self.df,
                                       model_type = 'classification',
                                       predicted_column = 'y')
        oo.train_test_split()
        X_train_pre_scale = oo.X_train.copy()
        oo.feature_scaling(['x1','x2'])
        self.assertEqual(type(X_train_pre_scale), type(oo.X_train))

    def test_X_test_type(self):
        oo = hc.DevelopSupervisedModel(dataframe = self.df,
                                       model_type = 'classification',
                                       predicted_column = 'y')
        oo.train_test_split()
        X_test_pre_scale = oo.X_test.copy()
        oo.feature_scaling(['x1','x2'])
        self.assertEqual(type(X_test_pre_scale), type(oo.X_test))

    def test_X_train_shape(self):
        oo = hc.DevelopSupervisedModel(dataframe = self.df,
                                       model_type = 'classification',
                                       predicted_column = 'y')
        oo.train_test_split()
        X_train_pre_scale = oo.X_train.copy()
        oo.feature_scaling(['x1','x2'])
        self.assertEqual(X_train_pre_scale.shape, oo.X_train.shape)

    def test_X_test_shape(self):
        oo = hc.DevelopSupervisedModel(dataframe = self.df,
                                       model_type = 'classification',
                                       predicted_column = 'y')
        oo.train_test_split()
        X_test_pre_scale = oo.X_test.copy()
        oo.feature_scaling(['x1','x2'])
        self.assertEqual(X_test_pre_scale.shape, oo.X_test.shape)
        
    def test_X_train_column_names(self):
        oo = hc.DevelopSupervisedModel(dataframe = self.df,
                                       model_type = 'classification',
                                       predicted_column = 'y')
        oo.train_test_split()
        X_train_pre_scale = oo.X_train.copy()
        oo.feature_scaling(['x1','x2'])
        self.assertEqual(sum(X_train_pre_scale.columns == oo.X_train.columns),
                         len(X_train_pre_scale.columns) )

    def test_X_test_column_names(self):
        oo = hc.DevelopSupervisedModel(dataframe = self.df,
                                       model_type = 'classification',
                                       predicted_column = 'y')
        oo.train_test_split()
        X_test_pre_scale = oo.X_test.copy()
        oo.feature_scaling(['x1','x2'])
        self.assertEqual(sum(X_test_pre_scale.columns == oo.X_test.columns),
                         len(X_test_pre_scale.columns) )

    def test_X_train_zero_mean(self):
        oo = hc.DevelopSupervisedModel(dataframe = self.df,
                                       model_type = 'classification',
                                       predicted_column = 'y')
        oo.train_test_split()
        oo.feature_scaling(['x1','x2'])
        if( abs(oo.X_train.x1.mean()) < 0.01 and
            abs(oo.X_train.x2.mean()) < 0.01 ):
            test = True
        else:
            test = False
        self.assertEqual( test, True )

    def test_X_train_unit_variance(self):
        oo = hc.DevelopSupervisedModel(dataframe = self.df,
                                       model_type = 'classification',
                                       predicted_column = 'y')
        oo.train_test_split()
        oo.feature_scaling(['x1','x2'])
        if( abs(oo.X_train.x1.std() - 1) < 0.5 and
            abs(oo.X_train.x2.std() - 1) < 0.5 ):
            test = True
        else:
            test = False
        self.assertEqual( test, True )
        
    def tearDown(self):
        del self.df

        
if __name__ == '__main__':
    unittest.main()
