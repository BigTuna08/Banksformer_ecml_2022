# from .constants import code_order

import os

# BASE_DIR = os.getcwd()
# os.chdir(config.MY_LIB_IN)
from my_lib.encoding import load_data_encoder
# os.chdir(BASE_DIR)




# CAT_FIELDS  = ["tcode_num"]
CAT_FIELDS = ["k_symbol_num", "operation_num", "type_num"]
# DATE_FIELDS = ['dow', 'month', "day", 'td_sc']  
DATE_FIELDS = ['dow', 'month', "day", 'dtme', 'td_sc']  
CONT_FIELDS = ['log_amount_sc']  

DATA_KEY_ORDER = CAT_FIELDS + DATE_FIELDS + CONT_FIELDS

cat_code_fields = ['type', 'operation', 'k_symbol'] # not used in network! only for creating tcode with correct order
TCODE_SEP = "__"

NON_OUT_FIELDS = ['bal_sc']




date_loss = "scce"

LOSS_TYPES = {"day": date_loss,
              "dtme": date_loss,
           "dow": date_loss,
           "month": date_loss,
            "td_sc": "pdf",
            "log_amount_sc": "pdf",
#             "tcode_num": "scce",
             }





# cl - clock encoding (2d)
# oh - One-hot encoding
# raw - no encoding
# cl-i -  clock integer: transforms [1, 2, ..., n] -> [1, 2, ..., n-1, 0]
INP_ENCODINGS = {"day": "cl",
                 "dtme": "cl",
           "dow": "cl",
           "month": "cl",
            "td_sc": "raw",
            "log_amount_sc": "raw",
            "tcode_num": "oh",}

TAR_ENCODINGS = {"day": "cl-i",
                 "dtme": "cl-i",
           "dow": "cl-i",
           "month": "cl-i",
            "td_sc": "raw",
            "log_amount_sc": "raw",
            "tcode_num": "raw",}



CLOCK_DIMS = {"day": 31,
              "dtme": 31,
           "dow": 7,
           "month": 12,}




for field in CAT_FIELDS:
    LOSS_TYPES[field] = "scce"
    INP_ENCODINGS[field] = "oh"
    TAR_ENCODINGS[field] = "raw"


FIELD_DIMS_IN  = {}
FIELD_DIMS_TAR = {}
FIELD_DIMS_NET = {}


ENCODING_INP_DIMS_BY_TYPE = {'cl':2, 
                             'oh':None, 
                             'raw':1}

ENCODING_TAR_DIMS_BY_TYPE = {'cl-i': 1, 
                             'raw': 1}

# ENCODING_NET_DIMS_BY_TYPE 

for k in DATA_KEY_ORDER:
    
    FIELD_DIMS_IN[k] = ENCODING_INP_DIMS_BY_TYPE[INP_ENCODINGS[k]]
    FIELD_DIMS_TAR[k] = ENCODING_TAR_DIMS_BY_TYPE[TAR_ENCODINGS[k]]
    
    if TAR_ENCODINGS[k] == "raw":
        FIELD_DIMS_NET[k] = 2
    elif TAR_ENCODINGS[k] == "cl-i":
        FIELD_DIMS_NET[k] = CLOCK_DIMS[k]
    else:
        raise Exception(f"Error getting network dim for field = {k}")
    
    
    
#     FIELD_DIMS_NET 


# FIELD_DIMS_IN = {"day": 2,
#            "dow": 2,
#            "month": 2,
#             "td_sc": 1,
#             "log_amount_sc": 1,
#             "tcode_num": None, # gets set based on training data
#                 }

# FIELD_DIMS_TAR = {"day": 1,
#            "dow": 1,
#            "month": 1,
#             "td_sc": 1,
#             "log_amount_sc": 1,
#             "tcode_num": 1, # 
#                  }

# FIELD_DIMS_NET = {"day": 31,
#            "dow": 7,
#            "month": 12,
#             "td_sc": 2,
#             "log_amount_sc": 2,
#             "tcode_num": None, # gets set based on training data
#                 }





print("DATA_KEY_ORDER is", DATA_KEY_ORDER)
print("LOSS_TYPES are:", ", ".join([f"{x} - {y}" for x,y in LOSS_TYPES.items()]))
print("If this is not correct, edit field_config.py and re-run notebook")


def get_field_info(ds_suffix):
    data_encoder = load_data_encoder(ds_suffix)
    
#     print("DE =", data_encoder.__dict__)

#     ONE_HOT_DIMS = {" "day": 31,
#            "dow": 7,
#            "month": 12,}
    
    

#     FIELD_DIMS_IN["tcode_num"] = data_encoder.n_tcodes
# #     FIELD_DIMS_TAR["tcode_num"] = data_encoder.n_tcodes
#     FIELD_DIMS_NET["tcode_num"] = data_encoder.n_tcodes
    
    for field in CAT_FIELDS:
        LOSS_TYPES[field] = "scce"
        INP_ENCODINGS[field] = "oh"
        TAR_ENCODINGS[field] = "raw"
        
        n = data_encoder.get_n_cats(field)
        FIELD_DIMS_IN[field] = n
#         ONE_HOT_DIMS[field] = n
        FIELD_DIMS_NET[field] = n

        

    FIELD_STARTS_IN = {}
    start = 0
    for k in DATA_KEY_ORDER:

        FIELD_STARTS_IN[k] = start
        start += FIELD_DIMS_IN[k]



    FIELD_STARTS_TAR = {}
    start = 0
    for k in DATA_KEY_ORDER:

        FIELD_STARTS_TAR[k] = start
        start += FIELD_DIMS_TAR[k]
        
        
    FIELD_STARTS_NET = {}
    start = 0
    for k in DATA_KEY_ORDER:

        FIELD_STARTS_NET[k] = start
        start += FIELD_DIMS_NET[k]
        
        
    return FIELD_DIMS_IN, FIELD_STARTS_IN, FIELD_DIMS_TAR, FIELD_STARTS_TAR, FIELD_DIMS_NET, FIELD_STARTS_NET

    
    
    
