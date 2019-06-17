import time

import platform
import io
import glob

import pandas as pd
import numpy as np

def give_claim(comment):
    df=pd.read_csv('C:\\Users\\bosil\\Documents\\AI4Good\\factually\\mythfact.csv')
    df.columns = ['Keyword','Claim','False']

    keyword_bank = df.Keyword.unique().tolist()

    for keyword in keyword_bank:
      if (keyword in comment.lower()):
          new_df = (df.loc[df['Keyword'].str.contains(keyword),:])
          return new_df.iloc[0]['Claim']

    return ''
