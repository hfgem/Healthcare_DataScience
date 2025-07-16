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

# %%


clean_results = run_data_cleanup(data)


# %%

import re
import os
import nltk
import tqdm
import string
import time
import csv
import numpy as np
import pandas as pd
from functions.ai_call_functions import clean_dataframe_column

dataframe = data.dataset
save_dir = data.save_dir
params = data.params_dict
# api_key = params['OPENAI_API_KEY']
# num_entries = dataframe.shape[0]
# max_chunk = params['Max_send']
dtype_dict = clean_results.dtype_dict
stopwords = clean_results.stopwords

# Check column datatype match
all_consistent = clean_results.all_consistent
# Check missing values
missing_df = clean_results.missing_df


df_copy = dataframe.copy()
column_name = 'Name'
df_copy[column_name] = df_copy[column_name].fillna('')
df_copy[column_name] = df_copy[column_name].str.lower()
translator = str.maketrans('', '', string.punctuation)
df_copy[column_name] = df_copy[column_name].apply(
    lambda x: x.translate(translator))
df_copy[column_name] = df_copy[column_name].str.replace(
    r'\s+', ' ', regex=True).str.strip()

honorifics = np.load(os.path.join(os.path.split(current_dir)[0],'utils','honorifics.npy'),\
        np.array(honorifics_degrees_lower),allow_pickle=True)