import time

import platform
import io
import glob

import pandas as pd
import numpy as np

# -*- coding: utf-8 -*-

def give_claim(comment):
    df=pd.read_csv('data/mythfact.csv')
    df.columns = ['Keyword','Claim','False']

    keyword_bank = df.Keyword.unique().tolist()

    for keyword in keyword_bank:
      if (keyword in comment.lower()):
          new_df = (df.loc[df['Keyword'].str.contains(keyword),:])
          return new_df.iloc[0]['Claim']

    return "There's a lot of information about vaccines online, but not all of it is accurate. Find credible, expert-approved information at http://vaccines.gov ."
