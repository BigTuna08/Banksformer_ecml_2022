# the column names of categorical date features
CAT_DATE_FIELDS = ["day", "dow", "month", "dtme"]

# the column names of categorical code features
# CAT_CODE_FIELDS = ["k_symbol", "operation", "type"]
CAT_CODE_FIELDS = ['description', 'flag', 'type']

# the column names of continous features
CONT_FIELDS = ["amount", "raw_amount", "td"]

# seperator character used when concatinating codes into tcode
TCODE_SEP = "__"

# Set true for backwards compatibility w/ czech data, otherwise False
CONVERT_SHORTNAMES = True