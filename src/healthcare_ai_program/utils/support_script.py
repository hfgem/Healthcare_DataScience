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
os.chdir(current_dir)

from functions.import_data import run_data_imports

data = run_data_imports()

#%%

from functions.clean_data import run_data_cleanup

clean_results = run_data_cleanup(data)


    
#%%