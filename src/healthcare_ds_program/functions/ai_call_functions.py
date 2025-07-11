#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 15:13:22 2025

@author: hannahgermaine
"""

import os
from openai import OpenAI
import pandas as pd

def clean_dataframe_column(column_data,instruction,api_key):
    """
    This function takes in a pandas dataframe and uses openai to clean the 
    columns automatically.

    Parameters
    ----------
    dataframe : Pandas DataFrame
        Pandas dataframe containing patient data.
    api_key : String
        API key to use openai.

    Returns
    -------
    dataframe : Pandas DataFrame
        Pandas dataframe containing cleaned patient data.

    """
    client = OpenAI(api_key = api_key)
    
    prompt = f"Clean this list of values:\n{column_data.tolist()}\nInstruction: {instruction}\nReturn the cleaned list only."

    response = client.responses.create(
        model="gpt-4",
        instructions="You are a helpful data cleaning assistant.",
        input=prompt,
    )

    cleaned = response.choices[0].message.content
    try:
        # Evaluate safely if GPT returns a Python list
        return eval(cleaned)
    except Exception:
        print("Could not parse GPT response:", cleaned)
        return column_data
    
    
    return cleaned
    