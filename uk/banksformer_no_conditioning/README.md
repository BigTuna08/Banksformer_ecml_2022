# UK data experiments - Banksformer NC

Code for running the syntetic data generation experiments from *Banksformer: A Deep Generative Model for Synthetic Transaction Sequences* 

Note: This code can be used for running variants BF-NC and TF-V used in ablation experiments. To change version, set the VERSION variable in 'field_config.py' to either 'BF-NC' or 'TF-V'. If you change any information in 'field_config.py', you should run clear_old.ipynb and then restart the workflow, otherwise unexpected behavior may occur.


Workflow:
Run the notebooks in the order: nb1, nb2, nb3. Specifically:
- 'nb1_preprocess_czech.ipynb' - Here you need to set some information if you are using a new dataset.  This nb ensures correct fields exist in dataframe, and stores information about the configurations used. 
- 'nb2_encode_data.ipynb' - This nb takes the dataframe produced by nb1, and encodes the data as tensors, used for training the networks.
- 'nb3_banksformer-v2.ipynb' - This nb trains banksformer models on the tensors, and then uses the trained models to generate synthetic datasets.

