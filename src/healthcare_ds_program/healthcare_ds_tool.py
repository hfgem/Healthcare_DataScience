#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 13:16:14 2025

@author: hannahgermaine
"""

import os

current_path = os.path.realpath(__file__)
current_dir = os.path.dirname(current_path)
os.chdir(current_dir)

from functions.import_data import run_data_imports
from functions.clean_data import run_data_cleanup
from functions.basic_stats import run_basic_stats

if __name__ == '__main__':
    
    #Import relevant data / params
    data = run_data_imports()
    
    #Clean up dataset
    clean_data = run_data_cleanup(data)
    
    #Run basic statistics
    basic_stats = run_basic_stats([clean_data.clean_dataframe,data.save_dir])
    
    