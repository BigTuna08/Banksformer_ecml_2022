import numpy as np
import pickle
from math import sin, cos, pi
# from datetime import date
# import os
# import pandas as pd


# # from .constants import *



class DataEncoder:
    
    def __init__(self, categorical_fields):
        print("iniy!")
        self.categorical_fields = categorical_fields
    
    
    def fit_transform(self, df):
        

        
        df["log_amount"] = np.log10(df["amount"]+1)
        self.LOG_AMOUNT_SCALE = df["log_amount"].std()
        df["log_amount_sc"] = df["log_amount"] / self.LOG_AMOUNT_SCALE
        
        self.TD_SCALE = df["td"].std()
        df["td_sc"] = df["td"] / self.TD_SCALE
        
        self.ATTR_SCALE = df["age"].std()
        df["age_sc"] = df["age"] / self.ATTR_SCALE
        
        self.START_DATE = df["datetime"].min()
        
        
        
        for field in self.categorical_fields:
            field = field.replace("_num", "")
            cat_to_num = dict([(tc, i) for i, tc in enumerate(df[field].unique())])
            self.__setattr__(f"{field}_to_num".upper(), cat_to_num)
            
            self.__setattr__(f"num_to_{field}".upper(), 
                            dict([(i, tc) for i, tc in enumerate(df[field].unique())]))
            
        
            df[field + "_num"] = df[field].apply(lambda x: cat_to_num[x])
            
            self.__setattr__(f"n_{field}s", len(cat_to_num)) 
            
            
    def get_n_cats(self, field):
        field = field.replace("_num", "")
        field = f"n_{field}s"
        return self.__getattribute__(field)
    

    def get_code_num(self, field, code):
        field = field.replace("_num", "")
        d = self.__getattribute__(f"{field}_to_num".upper())
        return d[code]

    

    def get_code_from_num(self, field, num):
        field = field.replace("_num", "")
        d = self.__getattribute__(f"num_to_{field}".upper())
        return d[code]

        
        

def preprocess_df(df, catfields, ds_suffix = None):
    de = DataEncoder(catfields)
    de.fit_transform(df)
    
    if ds_suffix == None:
        print("No ds_suffix set. Using ds_suffix = 'default'. (ds_suffix is used for keeping track of different dataset versions)")
        ds_suffix = 'default'
    
    
    with open(f"stored_data/DataEncoder-{ds_suffix}.pickle", "wb") as f:
        pickle.dump(de, f) 
        print("Wrote encoding info to", f.name)
        
    return de

        

def load_data_encoder(ds_suffix):
    with open(f"stored_data/DataEncoder-{ds_suffix}.pickle", "rb") as f:
        return pickle.load(f) 

  
def encode_time_value(val, max_val):
    return sin(2* pi * val/max_val), cos(2*pi * val/max_val)

def bulk_encode_time_value(val, max_val):
    x = np.sin(2* np.pi/max_val * val)
    y = np.cos(2*np.pi /max_val * val)
    return np.stack([x,y], axis=1)
        