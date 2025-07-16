#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 20:13:01 2025

@author: hannahgermaine
"""

import re
import os
import nltk
import string
import time
import numpy as np
import pandas as pd

current_path = os.path.realpath(__file__)
current_dir = os.path.dirname(current_path)

#TODO: convert into package that can be imported above

honorifics = np.load(os.path.join(os.path.split(current_dir)[0],'utils','honorifics.npy'),\
        allow_pickle=True)

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
        self.dtype_dict = self.import_dtype_list(dataframe,save_dir)
        self.stopwords = nltk.corpus.stopwords.words('english')
        
        #Check column datatype match
        self.all_consistent = self.check_data_type_consistency(dataframe)
        
        #Check missing values
        self.missing_df = self.check_missing_values(dataframe)
        
        #Clean columns
        self.clean_dataframe = self.clean_columns(dataframe)
        
        #Check for duplicates
        self.clean_dataframe = self.check_duplicate_rows(self.clean_dataframe)
            
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
            elif len(np.intersect1d(['int64','float64'],self.dtype_dict[column])) > 0:
                df_copy = self.clean_num_column(df_copy, column)
        
        
        return df_copy

    def clean_text_column(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
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

        # 3. Remove honorifics
        for hr in honorifics:
            pattern = r'\b' + re.escape(hr) + r'\b'
            df_copy[column_name] = df_copy[column_name].str.replace(pattern, '', regex=True)
        
        # 4. Remove duplicate spaces and strip leading/trailing whitespace
        df_copy[column_name] = df_copy[column_name].str.replace(r'\s+', ' ', regex=True).str.strip()

        # 4. Remove honorifics
        for hr in honorifics:
            df_copy[column_name] = df_copy[column_name].str.replace(hr, '', regex=False)
        
        print(f"✅ Successfully cleaned column '{column_name}'.")
        
        return df_copy
    
    def clean_num_column(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
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
        df_cleaned = df_copy.dropna(subset=[column_name])
        
        # Check for unreasonable outliers and report
        df_mean = np.mean(df_cleaned[column_name])
        df_std = np.std(df_cleaned[column_name])
        df_zscore = (df_cleaned[column_name] - df_mean)/df_std
        threshold = 3
        outliers = np.where(np.abs(df_zscore) > threshold)[0]
    
        if len(outliers) > 0:
            print(f" Column '{column_name}' contains " + str(len(outliers)) + " outliers.")
        
        print(f"✅ Successfully cleaned column '{column_name}'.")
        
        return df_copy

    def check_duplicate_rows(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def check_unique_values(self, df: pd.DataFrame, column_name: str) -> bool:
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
