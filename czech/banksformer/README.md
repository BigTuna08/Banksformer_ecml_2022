# Czech data experiments - Banksformer

Code for running the syntetic data generation experiments from *Banksformer: A Deep Generative Model for Synthetic Transaction Sequences* 

Note: This code can be used for running the base version of Banksformer, as well as the BF-ND version used in ablation experiments. To change version, set the VERSION variable in 'field_config.py' to either 'BASE' or 'BF-ND'. If you change any information in 'field_config.py', you should run clear_old.ipynb and then restart the workflow, otherwise unexpected behavior may occur.


Workflow:
Run the notebooks in the order: nb1, nb2, nb3. If you do not have a preprocessed version of the dataset, follow the instructions in nb0 to obtain and preprocess the raw data. Specifically:
- 'nb0_create_dataset.ipynb' - This notebook contains directions for obtaining the raw files used to create the czech dataset. Once the raw files are obtained, run the notebook to create a preprocessed version of the dataset.
- 'nb1_preprocess_czech.ipynb' - Here you need to set some information if you are using a new dataset.  This nb ensures correct fields exist in dataframe, and stores information about the configurations used. 
- 'nb2_encode_data.ipynb' - This nb takes the dataframe produced by nb1, and encodes the data as tensors, used for training the networks.
- 'nb3_banksformer-v2.ipynb' - This nb trains banksformer models on the tensors, and then uses the trained models to generate synthetic datasets.

