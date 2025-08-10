#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 10:48:24 2025

@author: hannahgermaine

Healthcare AI Support Script
"""

import os

current_path = os.path.realpath(__file__)
current_dir = os.path.dirname(current_path)
os.chdir(os.path.split(current_dir)[0])


import string
from functions.clean_data import run_data_cleanup
from functions.import_data import run_data_imports

data = run_data_imports()

clean_results = run_data_cleanup(data)

#%%

dataframe = clean_results.clean_dataframe

from functions.basic_stats import run_basic_stats
basic_stats = run_basic_stats([dataframe,data.save_dir])
