import pickle


class Result:
    
    def __init__(self, 
                 real_df_name, 
                 gen_df_name, 
                 univariate_cont_res,
                 ngram_res,
                 amount_codes_res,
                 amount_code_date_res,
                ):
        

        self.real_df_name = real_df_name
        self.gen_df_name = gen_df_name
        
        
        self.univariate_cont_res = univariate_cont_res
        self.ngram_res = ngram_res
        self.amount_codes_res = amount_codes_res
        self.amount_code_date_res = amount_code_date_res
        
        
    def save(self, loc):
        if not loc.split(".")[-1] == "result":
            loc += ".result"
            
        with open(loc, "wb") as f:
            pickle.dump(self, f)
            
            
    def load(loc):
        if not loc.split(".")[-1] == "result":
            loc += ".result"
        with open(loc, "rb") as f:
            return pickle.load(f)
        

        
        
