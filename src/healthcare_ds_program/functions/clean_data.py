#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 20:13:01 2025

@author: hannahgermaine
"""

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

current_path = os.path.realpath(__file__)
current_dir = os.path.dirname(current_path)

class run_data_cleanup():
    
    def __init__(self,data):
        self.data = data
        self.clean_test_data()
        
    def clean_test_data(self,):
        """
        Run commands to clean the given dataset.
        """
        dataframe = self.data.dataset
        save_dir = self.data.save_dir
        params = self.data.params_dict
        # api_key = params['OPENAI_API_KEY']
        # num_entries = dataframe.shape[0]
        # max_chunk = params['Max_send']
        self.dtype_dict = self.import_dtype_list(dataframe,save_dir)
        self.stopwords = nltk.corpus.stopwords.words('english')
        
        #Check column datatype match
        self.all_consistent = self.check_data_type_consistency(dataframe)
        
        #Check missing values
        self.missing_df = self.check_missing_values(dataframe)
        
        #
        
        # outlier_data = []
        # outlier_notes = []
        # for k_i, kname in enumerate(keys):
        #     data_type = dataframe[kname].dtypes
        #     #Manual NLTK Usage
        #     if data_type == 'O':
        #         dataframe, k_outlier_data, k_outlier_notes = self.text_clean(dataframe, kname, num_entries)
        #     elif (data_type == 'int') or (data_type == 'float'):
        #         dataframe, k_outlier_data, k_outlier_notes = self.digit_clean(dataframe,kname)
        #     outlier_data.extend(k_outlier_data)
        #     outlier_notes.extend(k_outlier_notes)
        
        # self.clean_dataframe = dataframe
    
        #AI call to clean
        # data_token_counts = np.zeros(num_entries)
        # for e_i in range(num_entries):
        #     data_token_counts[e_i] = len(list(dataframe[kname][e_i]))
        # data_token_cumsum = np.cumsum(data_token_counts)
        # start_inds = [0]
        # end_inds = [int(np.where(data_token_cumsum <= max_chunk)[0][-1])]
        # while end_inds[-1] != num_entries-1:
        #     start_inds.append(end_inds[-1])
        #     next_ind = int(np.where((data_token_cumsum-data_token_cumsum[end_inds[-1]]) <= max_chunk)[0][-1])
        #     end_inds.append(next_ind)
        # start_end_inds = list(zip(start_inds,end_inds))
        # if data_type == 'O':
        #     for s_i, e_i in start_end_inds:
        #         dataframe[kname][s_i:e_i] = clean_dataframe_column(dataframe[kname][s_i:e_i] , \
        #                         "Capitalize only the first letter for each sentence and lowercase all others." + \
        #                             " If not a string, convert to NaN.",\
        #                             api_key)
        # elif (data_type == 'int') or (data_type == 'float'):
        #     for s_i, e_i in start_end_inds:
        #         dataframe[kname][s_i:e_i] = clean_dataframe_column(dataframe[kname][s_i:e_i] , \
        #                     "Confirm that each value is reasonable given it falls in the " + kname + " category." + \
        #                         " If not reasonable, convert to NaN for future removal.",\
        #                         api_key)
            
    def import_dtype_list(self, df: pd.DataFrame, savedir: str) -> list:
        """
        Imports previously identified by user datatype list for dataframe or
        prompts the user for datatype identification and saves results.

        Parameters
        ----------
        df : pd.DataFrame
            Given dataframe.
        savedir : str
            Given save directory.

        Returns
        -------
        dtype_dict: list
            List containing data types expected from dataframe columns.
        """
        
        try:
            dtype_dict = np.load(os.path.join(savedir,'dtype_dict.npy'),allow_pickle=True).item()
            if len(dtype_dict) < len(df.columns):
                dtype_dict = self.get_user_dtype_input(df)
        except:
            dtype_dict = self.get_user_dtype_input(df)
        
        print("\n--- Expected data types: ---")
        print(dtype_dict)
        np.save(os.path.join(savedir,'dtype_dict.npy'),dtype_dict,allow_pickle=True)
        
        return dtype_dict
            
    def get_user_dtype_input(self, df: pd.DataFrame) -> list:
        """
        Cycles through all dataframe column names and asks the user to identify 
        the expected datatype. Stores outputs in a .csv in the save folder and
        returns the outputs as a list.
        

        Parameters
        ----------
        df : pd.DataFrame
            Given dataframe for column datatype identification.

        Returns
        -------
        list
            List of data types identified by user.
        """
        print("\n--- Acquiring dataframe key datatypes. ---")
        dtype_dict = dict()
        dtype_options = ['int64', 'float64', 'object', 'datetime64[ns]']
        for column_name in df.columns:
            prompt = 'What datatype is expected in column ' + column_name + '?'
            dtype_ind = self.user_integer_selection(prompt,dtype_options)
            dtype_dict[column_name] = dtype_options[dtype_ind]
        
        return dtype_dict
            
    def user_integer_selection(self, prompt: str, dtype_options: list) -> int:
        """
        Prompts the user using 'prompt' to select one of the options in 
        'options' by entering an integer.
        
        Parameters
        -------
        prompt : str
            Provided prompt for user
        dtype_options : list
            Provided list of options for user to select from

        Returns
        -------
        int_val : int
            Selected option index.
        """
        options_text = prompt
        for i, o in enumerate(dtype_options):
            options_text = options_text + "\n" + str(i) + " : " + str(o)
        correct_type = False
        attempts = 0
        while (not correct_type) and (attempts <= 3):
            print(options_text)
            user_input = input("Please enter the index of your selection: ")
            try:
                int_val = int(user_input)
                correct_type = True
            except:
                print("Incorrect input value. Please enter an integer.")
                if attempts < 3:
                    attempts += 1
                else:
                    print("Maximum attempts achieved. Quitting program.")
                    time.sleep(3)
                    quit()
        
        return int_val
    
    def check_data_type_consistency(self, df: pd.DataFrame) -> bool:
        """
        Verifies if DataFrame columns match the expected data types.

        Parameters
        ----------
            df: pd.DataFrame
                The input pandas DataFrame.
            expected_schema: dict
                A dictionary where keys are column names and values
                             are the expected data types (e.g., np.number, 'object').

        Returns
        -------
            True if all specified columns match their expected data types, False otherwise.
        """
        print("\n--- Checking Data Type Consistency ---")
        all_consistent = True
        not_consistent = []
        for column in df.columns:
            is_consistent = str(df[column].dtype) == self.dtype_dict[column]
            if not is_consistent:
                try: #try forcing it
                    df[column] = df[column].astype(self.dtype_dict[column])
                except:
                    all_consistent = False
                    not_consistent.append(column)
        
        if all_consistent:
            print("\n All checked columns have consistent data types.")
        else:
            print("\n Some columns have data type inconsistencies.")
            print(not_consistent)
            
        return all_consistent
    
    def check_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the number and percentage of missing values for each column.

        Parameters
        ----------
            df: pd.DataFrame
                The input pandas DataFrame.

        Returns
        -------
            A DataFrame summarizing the missing values, sorted by the percentage
            of missing values in descending order.
        """
        print("--- Checking for Missing Values ---")
        missing_count = df.isnull().sum()
        missing_count += df.isna().sum()
        missing_percentage = (missing_count / len(df)) * 100
        missing_df = pd.DataFrame({
            'missing_count': missing_count,
            'missing_percentage': missing_percentage
        })
        # Sort by the percentage of missing values to prioritize columns
        missing_df = missing_df[missing_df['missing_count'] > 0].sort_values(
            by='missing_percentage', ascending=False
        )
        if missing_df.empty:
            print("\n No missing values found.")
        else:
            print("\n Found missing values in the following columns:")
            print(missing_df)
        return missing_df
    
#%%    
    def clean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans each column based on its datatype.

        Parameters
        ----------
        df : pd.DataFrame
            DESCRIPTION.
        dtype_dict : dict
            DESCRIPTION.

        Returns
        -------
        A DataFrame with cleaned columns.

        """
        # Create a copy to avoid SettingWithCopyWarning
        df_copy = df.copy()
        
        for column in df_copy.columns:
            if self.dtype_dict[column] == 'object':
                df_copy = self.clean_text_column(df_copy, column)
            elif not np.intersect1d(['int64','float64'],self.dtype_dict[column]).isempty():
                df_copy = self.clean_num_column(df_copy, column)
        
        
        return df_copy

    def clean_text_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """
        Cleans a text column by converting to lowercase, removing punctuation,
        and standardizing whitespace.

        Parameters
        ----------
            df: pd.DataFrame
                The input pandas DataFrame.
            column_name: str
                The name of the text column to clean.

        Returns
        -------
            The DataFrame with the specified column cleaned.
        """
        
        print(f"\n--- Cleaning Text in Column '{column_name}' ---")
        # Create a copy to avoid SettingWithCopyWarning
        df_copy = df.copy()

        # Fill NA values with empty string to avoid errors in string operations
        df_copy[column_name] = df_copy[column_name].fillna('')

        # 1. Convert to lowercase
        df_copy[column_name] = df_copy[column_name].str.lower()

        # 2. Remove punctuation
        # Create a translation table to remove punctuation
        translator = str.maketrans('', '', string.punctuation)
        df_copy[column_name] = df_copy[column_name].apply(lambda x: x.translate(translator))

        # 3. Remove duplicate spaces and strip leading/trailing whitespace
        df_copy[column_name] = df_copy[column_name].str.replace(r'\s+', ' ', regex=True).str.strip()

        print(f"✅ Successfully cleaned column '{column_name}'.")
        
        return df_copy
    
    def clean_num_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """
        Cleans a column of dtype 'int' or 'float' by checking for NaNs and 
        testing for outliers.

        Parameters
        ----------
            df: pd.DataFrame
                The input pandas DataFrame.
            column_name: str
                The name of the text column to clean.

        Returns
        -------
            The DataFrame with the specified column cleaned.
        """
        
        print(f"\n--- Cleaning values in column '{column_name}' ---")
        if column_name not in df.columns:
            print(f"❌ Column '{column_name}' not found in DataFrame.")
            return df
        
        # Create a copy to avoid SettingWithCopyWarning
        df_copy = df.copy()

        # Check for NaNs and remove associated rows
        
        # Check for unreasonable outliers and report
        
        
        
        print(f"✅ Successfully cleaned column '{column_name}'.")
        
        return df_copy

    def check_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
        """
        Identifies and returns any duplicate rows in the DataFrame.

        Parameters
        ----------
            df: The input pandas DataFrame.

        Returns
        -------
            A DataFrame containing only the duplicated rows. Returns an empty
            DataFrame if no duplicates are found.
        """
        print("\n--- Checking for Duplicate Rows ---")
        duplicate_rows = df[df.duplicated(keep=False)]
        if duplicate_rows.empty:
            print("✅ No duplicate rows found.")
        else:
            print(f"⚠️ Found {len(duplicate_rows.drop_duplicates())} duplicate row(s).")
            print("Duplicate rows:")
            print(duplicate_rows.sort_values(by=df.columns.tolist()))
        return duplicate_rows

    def check_unique_values(df: pd.DataFrame, column_name: str) -> bool:
        """
        Checks if a specific column contains only unique values.

        Parameters
        ----------
            df: The input pandas DataFrame.
            column_name: The name of the column to check for uniqueness.

        Returns
        -------
            True if the column has all unique values, False otherwise.
        """
        print(f"\n--- Checking for Unique Values in '{column_name}' ---")
        if column_name not in df.columns:
            print(f"❌ Column '{column_name}' not found in DataFrame.")
            return False

        is_unique = df[column_name].is_unique
        if is_unique:
            print(f"✅ Column '{column_name}' contains all unique values.")
        else:
            # Find and display the values that are not unique
            duplicates = df[df[column_name].duplicated(keep=False)]
            print(f"⚠️ Column '{column_name}' has duplicate values.")
            print("Rows with duplicate values:")
            print(duplicates[[column_name]].sort_values(by=column_name))
        return is_unique

    def check_value_set(df: pd.DataFrame, column_name: str, allowed_values: set) -> pd.DataFrame:
        """
        Checks for values in a column that are not in a predefined set.

        Parameters
        ----------
            df: The input pandas DataFrame.
            column_name: The name of the column to check.
            allowed_values: A set containing the allowed values for the column.

        Returns
        -------
            A DataFrame containing rows with unexpected values in the specified column.
        """
        print(f"\n--- Checking for Allowed Values in '{column_name}' ---")
        if column_name not in df.columns:
            print(f"❌ Column '{column_name}' not found in DataFrame.")
            return pd.DataFrame()

        unexpected_rows = df[~df[column_name].isin(allowed_values)]
        if unexpected_rows.empty:
            print(f"✅ All values in '{column_name}' are within the allowed set.")
        else:
            print(f"⚠️ Found {len(unexpected_rows)} rows with unexpected values in '{column_name}'.")
            print("Rows with unexpected values:")
            print(unexpected_rows[[column_name]].drop_duplicates())
        return unexpected_rows