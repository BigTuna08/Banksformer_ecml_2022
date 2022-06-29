import sys
import calendar
import datetime

import pandas as pd

from .dataset_params import CAT_DATE_FIELDS, CAT_CODE_FIELDS, CONT_FIELDS, TCODE_SEP, CONVERT_SHORTNAMES


# assumes leap year, maps the (day, month) of date to a number
# indicating days from Jan 1st
def get_day_of_year(date):
    return (datetime.date(1992, date.month, date.day) - datetime.date(1992, 1, 1)).days
    
    
# converts code columns to strings 
def stringify_code_cols(df, cat_code_fields = CAT_CODE_FIELDS):

    for ccf in cat_code_fields:
        df[ccf] = df[ccf].astype(str)
    
    
# create tcode by concating fields in "cat_code_fields"
def set_tcode(df, cat_code_fields = CAT_CODE_FIELDS):
    tcode = df[cat_code_fields[0]].astype(str)
    for ccf in cat_code_fields[1:]:
        tcode += TCODE_SEP + df[ccf].astype(str)

    df["tcode"] = tcode
    
    
    

# def set_code_fields_from_nums


def set_code_fields(df, cat_code_fields = CAT_CODE_FIELDS):
    
    df[cat_code_fields] = df.tcode.str.split(TCODE_SEP, expand=True)
    

## Sets the values of 'tcode' and individual code fields
## This function works in the following cases
# 1 - If there is a column in df for each field in 'cat_code_fields', it will create a tcode by concatination
# 2 - If there is a tcode field and not the columns from 'cat_code_fields', it will create the code columns by splitting the tcode
# 3 - If both 1 & 2 are true, it checks that they both agree
# def set_code_fields(df, cat_code_fields = CAT_CODE_FIELDS, convert_shortnames = False, raise_if_nonmatch = True):
    
#     if convert_shortnames:
        
#         if "tcode" not in df.columns:
#             print("No 'tcode' field found, skipping convert_shortnames")
            
#         else:
#             from codenames import code_names as raw_code_names

#             # code_names = [(x, y.replace("-", TCODE_SEP)) for x,y in raw_code_names]
            
#             code_names = [(x, TCODE_SEP.join([y.split("-")[i] for i in [1,2,0]])) for x,y in raw_code_names]
#             short_to_long_names = dict(code_names)

#             df["short_tcode"] = df["tcode"]
#             df["raw_tcode"] = df.tcode.apply(lambda x: short_to_long_names[x])

#             df["tcode"] = df["raw_tcode"]  # just added

#     elif "tcode" in df.columns:
#         df["raw_tcode"] = df["tcode"]



#     # if all parts of tcode exist, create tcode col
#     if all([x in df.columns for x in cat_code_fields]):

#         set_tcode(df, cat_code_fields)

#         # if there was a tcode col, check it agrees with the newly created tcode
#         if "raw_tcode" in df.columns:

#             if not all(df.raw_tcode == df.tcode):
#                 print("Warning! tcode field doesn't match the concatinated codes!!", file=sys.stderr)
#                 print("Check where df.raw_tcode != df.tcode!", file=sys.stderr)
#                 print("Using concatinated codes version ...", file=sys.stderr)  
                
#                 if raise_if_nonmatch:
#                     raise Exception("df.raw_tcode != df.tcode")

#     # otherwise, if there is a tcode, split it into parts
#     elif "raw_tcode" in df.columns:
#         df[cat_code_fields] = df.raw_tcode.str.split(TCODE_SEP, expand=True)

#     else:
#         raise Exception("Dataframe must have either 'tcode' column, or all columns in 'cat_code_fields'")



    

def set_date_features(df):
    df["datetime"] = pd.to_datetime(df["datetime"])
    
    df["month"] = df["datetime"].dt.month
    df["day"] =   df["datetime"].dt.day
    df["dow"] =   df["datetime"].dt.dayofweek
    df["year"] =  df["datetime"].dt.year
    
    # day of year (# of days from Jan 1st in a leap year)
    df["doy"] = df["datetime"].apply(get_day_of_year)
    
    #days till month end
    df["dtme"] =  df.datetime.apply(lambda dt: calendar.monthrange(dt.year, dt.month)[1] - dt.day) # dtme - days till month end

    # time delta
    df["td"] = df[["account_id", "datetime"]].groupby("account_id").diff()
    df["td"] = df["td"].apply(lambda x: x.days)
    df["td"].fillna(0.0, inplace=True)


    
def preproc_df(df):
    
    set_date_features(df)
    stringify_code_cols(df)
    set_tcode(df)
    
