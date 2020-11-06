# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 12:02:40 2020

@author: Cassie Utkarsh Aaron 
"""

import pandas as pd

# Defining dataset
UFO_df = pd.read_csv('/DataMining/lab2/complete.csv')
# Cleaning data, there are quite a few pieces missing from the data set, so this 
# Needs to be addressed
UFO_df = UFO_df.dropna(how='all') # dropping all columns that are completely empty
UFO_df = UFO_df.drop(columns = 'city') # dropping city column, too much missing data

UFO_df.info
