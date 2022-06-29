# UK data metrics

Code for running the syntetic data generation experiments from *Banksformer: A Deep Generative Model for Synthetic Transaction Sequences* 

- generated_data/: Place synthetically generated datasets here for analysis
- mylib/: python helper files
- real_data/: Place the real dataset which you are comparing the synthetic data against here
- results/: Result objects are created here upon running 'nb1__make_results.ipynb'

- field_config.py: Contains configuration information. This must match the 'field_config.py' file used to generate the data.
- nb1__make_results.ipynb: Creates result objects. Note: This must be run before 'nb2__view_results.ipynb'
- nb2__view_results.ipynb: Displays data created by nb1, used to make the table in our paper.
- paper_figures-ecml.ipynb: Creates and displays figures from our paper