#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 13:29:14 2025

@author: hannahgermaine
"""

import os
import json
import kagglehub
import pandas as pd
from tkinter.filedialog import askdirectory

class run_data_imports():
    
    def __init__(self,):
        self.get_save_dir()
        self.import_params()
        self.import_dataset()
        
    def get_save_dir(self,):
        """Ask user input for directory to save results."""
        print("Please select a save directory for the results of this analysis.")
        self.save_dir = askdirectory()
        
    def import_params(self,):
        """Determine the param file path and import params into dictionary."""
        current_filepath = os.path.realpath(__file__)
        current_dir = os.path.dirname(current_filepath)
        params_dir = os.path.join(os.path.split(current_dir)[0],'params')
        param_files = os.listdir(params_dir)
        for pf in param_files:
            if pf == 'ai_params.json':
                self.param_path = os.path.join(params_dir,pf)
                with open(self.param_path, 'r') as params_file:
                    self.params_dict = json.load(params_file)
    
    def import_dataset(self,):
        """Import dataset to be analyzed"""
        # Download latest version
        path = kagglehub.dataset_download("prasad22/healthcare-dataset")
        data_file = os.listdir(path)[0]
        df = pd.read_csv(os.path.join(path,data_file))
        self.dataset = df