# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 14:44:24 2021

@author: gkako
"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Useful only for producing predicted masks (fill nan values)
bands_mean = np.array([0.05197577, 0.04783991, 0.04056812, 0.03163572, 0.02972606, 0.03457443,
 0.03875053, 0.03436435, 0.0392113,  0.02358126, 0.01588816]).astype('float32')

# Random Forest Initialization
random_forest=RandomForestClassifier(n_estimators = 125, 
                                     criterion='gini', 
                                     max_depth=20,
                                     min_samples_leaf=1, 
                                     min_impurity_decrease=0,
                                     oob_score=True, 
                                     class_weight='balanced_subsample',
                                     random_state=5,
                                     n_jobs=-1)

rf_classifier = Pipeline(steps=[('scaler', StandardScaler()), ('rf', random_forest)], verbose = 4) 