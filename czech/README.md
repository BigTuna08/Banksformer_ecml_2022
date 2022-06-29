# Czech data experiments

Code for running the syntetic data generation experiments from *Banksformer: A Deep Generative Model for Synthetic Transaction Sequences* 

- banksformer/ Contains proper version of Banksformer, where each feature prediction is conditional on the previous features within a transaction. This code can be used for running the full Banksformer model, as well as the BF-ND variant used in ablation experiments. NOTE: This code is the most well commented.
- banksformer_no_conditioning/ Contains ablated version of Banksformer, where each feature prediction is not conditional on the previous features within a transaction. This code can be used for running variants BF-NC and TF-V used in ablation experiments.
- metrics/ Contains code for evaluating generated data. 
