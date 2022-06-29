Workflow:



Step 1) Ensure the correct columns exist in the generated data.
Notebook - prep_generated_data.ipynb

Specifically, for each transaction we need:

1 - 'datetime' 
2 - 'raw_amount'
3 - At least one of: 
  -- (A) columns for each code in 'cat_code_fields' or 
  -- (B) tcode column which is a concatination of the codes in 'cat_code_fields'
  -- (C) tcode column with 'shortnames' and information in 'codenames.py' for converting (deprecated)
  
For step one, the notebook 'prep_generated_data.ipynb' may be helpful. This notebook is built for the czech dataset, and may require edits to work with
other data




Step 2) Create Results
Notebooks - make_results.ipynb
          - visualize_metrics.ipynb [This does not need to be ran. It prodives visuals for the metrics in make_results]
          
          
The 'make_results.ipynb' notebook will create a Result object for each generated dataset in the 'generated_datasets' folder




Step 3) View Results
Notebooks - view_results.ipynb

This notebook reads and displays information from the Result objects in the 'results' folder
