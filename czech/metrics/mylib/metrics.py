import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


from scipy.stats import energy_distance
from scipy.stats import wasserstein_distance
from scipy.stats import ks_2samp


import nltk
from nltk.util import ngrams

from scipy.special import rel_entr
from scipy.special import entr
from scipy.spatial import distance


from .dataset_params import *
from .result import Result



def ks_dist(real_obs, gen_obs):
    stat, pval = ks_2samp(real_obs, gen_obs)
    
    return stat


## Computes a set of metrics based on N-Grams
# field - Name of column in both dfs
# pseudo_counts - Used for label smoothing
# plot_n - max # of bars to plot
def compute_ngram_metrics(real_df, gen_df, field, n , pseudo_counts=0.0, plot=False, plot_n = 30, plot_sort_by = "counts_real", tick_lbls= False):

    
    n_codes_unique = len(set(real_df[field].unique()).union(set(gen_df[field].unique())))

    
    # create combo_df, which contains counts of all ngrams for both datasets (note: it omits any ngrams which do not occur in either dataset)
    real_ngrams = create_ngramcount_df(real_df, n, field)
    gen_ngrams = create_ngramcount_df(gen_df, n, field)
    combo_df = pd.merge(real_ngrams, gen_ngrams, on="ngram", how="outer", suffixes=("_real", "_gen")).fillna(0.0)
    
    plot_n = min(plot_n, len(combo_df))


    N_obs_real = real_ngrams["counts"].sum()
    N_obs_gen = gen_ngrams["counts"].sum()
    N_possible_ngrams = n_codes_unique**n 

    
    # add psudo-counts
    combo_df["counts_real"] += pseudo_counts
    combo_df["ps_real"] = combo_df["counts_real"] / (N_obs_real + N_possible_ngrams*pseudo_counts)
    combo_df["counts_gen"] += pseudo_counts
    combo_df["ps_gen"] = combo_df["counts_gen"] / (N_obs_gen + N_possible_ngrams*pseudo_counts)
    

        
        
    # compute jsd (note: contribution to jsd from any ngram not in either dataset is 0)
    combo_df["ps_mid"] = (combo_df["ps_real"] + combo_df["ps_gen"])/2
    kl_real_M = sum(rel_entr(combo_df["ps_real"], combo_df["ps_mid"])) 
    kl_gen_M = sum(rel_entr(combo_df["ps_gen"], combo_df["ps_mid"]))

    jsd = (kl_real_M + kl_gen_M)/2
        
        
    # compute entropy for both distributions
    n_unobs = N_possible_ngrams - len(combo_df)

    entr_r = entr(combo_df["ps_real"]).sum()  # from observed
    if pseudo_counts > 0.0:
        p_each_unobs = pseudo_counts / (N_obs_real + N_possible_ngrams*pseudo_counts)
        entr_r += -p_each_unobs*np.log(p_each_unobs)*n_unobs   # from unobserved

    entr_g = entr(combo_df["ps_gen"]).sum()  # from observed
    if pseudo_counts > 0.0:
        p_each_unobs = pseudo_counts / (N_obs_gen + N_possible_ngrams*pseudo_counts)
        entr_g += -p_each_unobs*np.log(p_each_unobs)*n_unobs   # from unobserved


    # package results    
    try:
        results = {"jsd":jsd, 
                      "entr_r":entr_r, 
                      "entr_g":entr_g,
                      "NED": entr_r - entr_g,
                      "l1":distance.minkowski(combo_df["ps_real"], combo_df["ps_gen"], p=1), 
                      "l2":distance.minkowski(combo_df["ps_real"], combo_df["ps_gen"], p=2),
                      "jac": distance.jaccard(combo_df["counts_real"]>0, combo_df["counts_gen"] > 0),
                      "count_r": len(real_ngrams),
                      "coverage_r": len(real_ngrams)/N_possible_ngrams,
                      "count_g": len(gen_ngrams),
                      "coverage_g": len(gen_ngrams)/N_possible_ngrams,
                      "count_max": N_possible_ngrams,
                      "field": field, 
                       "n":n, 
                       "pseudo_counts":pseudo_counts}
        
    except Exception as e:
        print("Error getting results. Error was:", e)
        return combo_df
            
        
    if plot:

        plt.figure(figsize=(14,6))
        
        combo_df = combo_df.sort_values(by=plot_sort_by, ascending=False)

        barplt = sns.barplot(data = combo_df.iloc[:plot_n],x="ngram", y="ps_real", color="b", alpha=0.5, label="real")
        barplt = sns.barplot(data = combo_df.iloc[:plot_n],x="ngram", y="ps_gen", color="orange", alpha=0.5, label="gen")


        if not tick_lbls:
            barplt.set_xticklabels(["" for _ in barplt.get_xticklabels()])
        else:
            for item in barplt.get_xticklabels():
                item.set_rotation(90)
                
        plt.title(f"Distributions of top {plot_n} {n}-grams ({field})")    
        plt.legend()
        plt.show()
        
        
    return results




# Creates dataframe with ngram counts.
def create_ngramcount_df(df, n, field):
    gb = df.sort_values(by=["account_id", "datetime"]).groupby("account_id", sort=False)[field]
    ngram_list = gb.apply(lambda x: list(ngrams(x, n=n)))

    counts = {}
    for ngram_seq in ngram_list:
        for ngram in ngram_seq:
            ngram = str(ngram)[1:-1]
            counts[ngram] = counts.get(ngram, 0) + 1
            
            
    df = pd.DataFrame.from_dict(counts, orient="index", columns=["counts"]).sort_values("counts", ascending=False)
    
            
    return df.reset_index().rename(columns={"index": "ngram"})



def jensenshannon_diverg(p,q):
    return distance.jensenshannon(p,q, base=2) ** 2


def compute_2d_categorical_cont_metrics(real_df, gen_df, field1):


    f1_opts = set(real_df[field1].unique()).union(set(gen_df[field1].unique())) # all unique options for field1 (in both dfs)

    n_opts_total = len(f1_opts)

    cont_metric_results = {}
    
    
    for code_1 in f1_opts:
        

        cond_r = real_df[field1] == code_1
        cond_g = gen_df[field1] == code_1

        p_r = np.sum(cond_r)  / len(cond_r) 
        p_g = np.sum(cond_g)  / len(cond_g) 


        for field in CONT_FIELDS:                             # loop over continous fields 
            cont_metric_results[field] = cont_metric_results.get(field, {})


            if np.sum(cond_r) > 0 and np.sum(cond_g) > 0:
                vals_r = real_df[cond_r][field]
                vals_g = gen_df[cond_g][field]

            elif np.sum(cond_r) > 0:
                vals_r = real_df[cond_r][field]
                vals_g = [np.mean(vals_r)]

            elif np.sum(cond_g) > 0:
                vals_g = gen_df[cond_g][field]
                vals_r =[ np.mean(vals_g)]

            else:
                continue

            for name, fn in CONTINUOUS_METRICS.items():                      # loop over continous metrics 


                to_add_r = p_r * fn(vals_r, vals_g) # expectation over R (check this makes sense)
                to_add_g = p_g * fn(vals_r, vals_g) # expectation over G (check this makes sense)

                kr = name + "_r"
                kg = name + "_g"


                cont_metric_results[field][kr] = cont_metric_results[field].get(kr, 0.) + to_add_r
                cont_metric_results[field][kg] = cont_metric_results[field].get(kg, 0.) + to_add_g

    return cont_metric_results




def compute_2d_categorical_metrics(real_df, gen_df, field1, field2, pseudo_counts = 0.0, plot = False, barplot_params = {}, 
                                   compute_cont_metrics = True, plot_ecdfs = False, plot_ecdf_params = {}):


    f1_opts = set(real_df[field1].unique()).union(set(gen_df[field1].unique()))
    f2_opts = set(real_df[field2].unique()).union(set(gen_df[field2].unique()))

    n_opts_total = len(f1_opts) * len(f2_opts)


    kl_r_m = 0.
    kl_g_m = 0.
    entr_r = 0.
    entr_g = 0.
    l1_d = 0.
    l2_d = 0.
    count_g = 0.
    count_r = 0.
    
    observed_opts = 0
    
    cont_metric_results = {}
    
    
    for code_1 in f1_opts:
        
        if plot:
            print(f"{field2.capitalize()}s of real transaction with {field1} =", 
                  code_1, f"{len(real_df[real_df[field1]==code_1])/len(real_df)*100:.2f}% of real data")
            if plot_ecdfs:
                fig, ax = plt.subplots(1, 3, figsize=(24,7), gridspec_kw={'width_ratios': [3, 1, 1]})
#                 fig, ax = plt.subplots(1, 2, figsize=(20,7), gridspec_kw={'width_ratios': [3, 1]})
                
                make_barplot(real_df[real_df[field1]==code_1], gen_df[gen_df[field1]==code_1], field=field2, ax=ax[0], **barplot_params)
                ax[0].legend()
            
            
                plot_ecdf(real_df[real_df[field1]==code_1]['raw_amount'], plot_params={"label":"real"}, ax=ax[2])
                plot_ecdf(gen_df[gen_df[field1]==code_1]['raw_amount'], plot_params={"label":"gen"}, ax=ax[2])
                
                ax[2].legend()
                ax[2].set_title(f"ECDF") # of amounts with {field1} = {code_1}
                
                ax[1].hist(real_df[real_df[field1]==code_1]['raw_amount'], label="real", bins = "auto", density=True, alpha=0.5)
                ax[1].hist(gen_df[gen_df[field1]==code_1]['raw_amount'], label="gen", bins="auto", density=True, alpha=0.5)

                ax[1].legend()
                ax[1].set_title(f"Hist of transaction amounts with \n {field1} = {code_1}")

            else:
                make_barplot(real_df[real_df[field1]==code_1], gen_df[gen_df[field1]==code_1], field=field2, **barplot_params)
                ax = [None, None]
                
            plt.show()
                
        
        
        for code_2 in f2_opts:

            cond_r = np.logical_and(real_df[field1] == code_1, real_df[field2] == code_2)
            cond_g = np.logical_and(gen_df[field1] == code_1, gen_df[field2] == code_2)

            p_r = (np.sum(cond_r) + pseudo_counts) / (len(cond_r) + pseudo_counts*n_opts_total)    
            p_g = (np.sum(cond_g) + pseudo_counts) / (len(cond_g) + pseudo_counts*n_opts_total)    
            p_m = (p_r + p_g) / 2.


            if np.sum(cond_r) + np.sum(cond_g) > 0:
                observed_opts += 1


            count_r += int(np.sum(cond_r) > 0)
            count_g += int(np.sum(cond_g) > 0)

            l1_d += np.abs(p_r - p_g)
            l2_d += (p_r - p_g) ** 2 
            

            if p_r > 0:
                kl_r_m += p_r * np.log(p_r / p_m)
                entr_r += - p_r * np.log(p_r)

            if p_g > 0.:
                kl_g_m += p_g * np.log(p_g / p_m)
                entr_g += - p_g * np.log(p_g)
                
                
            if compute_cont_metrics:

                for field in CONT_FIELDS:                             # loop over continous fields 
                    cont_metric_results[field] = cont_metric_results.get(field, {})
                    

                    if np.sum(cond_r) > 0 and np.sum(cond_g) > 0:
                        vals_r = real_df[cond_r][field]
                        vals_g = gen_df[cond_g][field]

                    elif np.sum(cond_r) > 0:
                        vals_r = real_df[cond_r][field]
                        vals_g = [np.mean(vals_r)]

                    elif np.sum(cond_g) > 0:
                        vals_g = gen_df[cond_g][field]
                        vals_r =[ np.mean(vals_g)]

                    else:
                        continue

                    for name, fn in CONTINUOUS_METRICS.items():                      # loop over continous metrics 


                        to_add_r = p_r * fn(vals_r, vals_g) # expectation over R (check this makes sense)
                        to_add_g = p_g * fn(vals_r, vals_g) # expectation over G (check this makes sense)

                        kr = name + "_r"
                        kg = name + "_g"

                        
                        cont_metric_results[field][kr] = cont_metric_results[field].get(kr, 0.) + to_add_r
                        cont_metric_results[field][kg] = cont_metric_results[field].get(kg, 0.) + to_add_g
                

                
    # compute jaccard
    sr = set(zip(real_df[field1].to_list(), real_df[field2].to_list()))
    sg = set(zip(gen_df[field1].to_list(), gen_df[field2].to_list()))
    s_union = len(sr.union(sg))
    s_inter = len(sr.intersection(sg))
    jacc_d = (s_union - s_inter) / s_union

    # finshed l2
    l2_d = np.sqrt(l2_d)

    # coverage
    coverage_g = count_g / n_opts_total
    coverage_r = count_r / n_opts_total

    #jsd
    jsd = (kl_r_m + kl_g_m) / 2


    result = {'jsd': jsd,
                 'entr_r': entr_r,
                 'entr_g': entr_g,
                 'l1': l1_d,
                 'l2': l2_d,
                 'jac': jacc_d,
                 'count_r': count_r,
                 'coverage_r': coverage_r,
                 'count_g': count_g,
                 'coverage_g': coverage_g,
                 'count_max': n_opts_total,
                 'cont_metric_results': cont_metric_results,}

    
    
    return result



#####################    Set CONTINUOUS_METRICS    #####################


# This is the list of metrics, with format - (name, function)
CONTINUOUS_METRICS = {"wasser": wasserstein_distance,
                "ks": ks_dist,
                "energy_d": energy_distance}


#####################    Compute all metrics    #####################

def compute_all_metrics(real_data_loc, gen_data_loc):
    
    
    from .preprocessing import preproc_df

    real_df = pd.read_csv(real_data_loc, parse_dates=["datetime"])
    preproc_df(real_df)


    gen_df = pd.read_csv(gen_data_loc, parse_dates=["datetime"])
    preproc_df(gen_df)
    

    ## Metrics for univariate continous features 

    univariate_cont_res = {}

    for field in CONT_FIELDS:
        univariate_cont_res[field] = {}
        for name, fn in CONTINUOUS_METRICS.items():
            univariate_cont_res[field][name] = fn(real_df[field], gen_df[field])

    ## N-Grams (sequences of categorical features)

    ngram_res = {}

    ngram_res["code"] = {}
    ngram_res["tcode"] = {} 
    ngram_res["date"] = {}

    for n in [1, 3]:


        ngram_res["code"][n] = {}
        for code in CAT_CODE_FIELDS:
            ngram_res["code"][n][f"{code}"] =  compute_ngram_metrics(real_df, gen_df, field=code, n=n, plot = False)


        ngram_res["tcode"][n] = compute_ngram_metrics(real_df, gen_df, field="tcode", n=n, plot = False)


        ngram_res["date"][n] = {}
        for code in CAT_DATE_FIELDS:
            ngram_res["date"][n][f"{code}"] = compute_ngram_metrics(real_df, gen_df, field=code, n=n, plot = False)


    ## Joint distributions

    ### Amount, codes  

    amount_codes_res = {}


    ## Code fields
    amount_codes_res["code"] = {}

    for code in CAT_CODE_FIELDS:
        amount_codes_res["code"][f"{code}"] =  compute_2d_categorical_cont_metrics(real_df, gen_df, code)

    amount_codes_res["tcode"] = compute_2d_categorical_cont_metrics(real_df, gen_df, code)


    ## Date fields (categorical)
    amount_codes_res["date"] = {}
    for code in CAT_DATE_FIELDS:
        amount_codes_res["date"][f"{code}"] =  compute_2d_categorical_cont_metrics(real_df, gen_df, code)


    ### Date, codes (and continous variables)

    amount_code_date_res = {}


    ## Code fields
    amount_code_date_res["code"] = {}
    amount_code_date_res["tcode"] = {}

    for date_field in CAT_DATE_FIELDS:

        amount_code_date_res["code"][date_field] = {}

        for code in CAT_CODE_FIELDS:

            amount_code_date_res["code"][date_field][code] =  compute_2d_categorical_metrics(real_df, gen_df, code, date_field, plot = False, plot_ecdfs=False)

        amount_code_date_res["tcode"][date_field] = compute_2d_categorical_metrics(real_df, gen_df, code, date_field, plot = False, plot_ecdfs=False)


    ## Create Result Object

    full_result =  Result(real_df_name = real_data_loc, 
                     gen_df_name = gen_data_loc, 
                     univariate_cont_res = univariate_cont_res,
                     ngram_res = ngram_res,
                     amount_codes_res = amount_codes_res, 
                     amount_code_date_res = amount_code_date_res,
                    )
    return full_result