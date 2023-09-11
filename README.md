# Conditional Average Treatment Effect Estimation via Meta-Learners
###### Master Thesis by Arberim Bibaj (August 2023, ETHZ, MSc Statistisc)
In the following you can see a description of the files and instructions to reproduce the results of the thesis.
Additionally, there is a short description of the project.

### Short Description of the Project
The goal was to compare the performances of the most essential meta-learners for
conditional average treatment effect (CATE) estimation.

Meta-Learners: T-, S-, X-, R-, DR-, RA-, PW-, and U-learner.

Base-Learners: random forests, lasso-based regression, neural networks.

### Description of the Files
`run_experiments.py`: main file that runs the experiments and saves the results in json format. \
`meta_learner.py`: contains the meta-learners classes. \
`neural_networks`: neural network architecture and helper functions. \
`data_generation_process.py`: contains the functions needed for the DGP. \
`default_values.py`: some default values, e.g., arguments for the base-learners. \
`preprocess_ihdp.py`: helper function to pre-process the IHDP dataset. You don't need to run it. \
`requirements.txt`: required packages with the corresponding versions for reproduction of the results.
`Plots.ipynb`: to plot the results in jupyter notebook.

### Folders
`final_results_final`: contains the results as json files. \
`ihdp`: contains the raw and processed ihdp data. \
`plots_final`: contains the finished plots in 'eps' format.

## Reproduction of the Results
First, please make sure to install the required packages on Python (3.11.2) with the corresponding versions:
````
pip install -r requirements.txt
````
Additionally, you need to install R (4.2.2) and the package mpower (0.1.0) as it is used for the Data Generating Process
within Python.

Next, to reproduce the results on the semi-synthetic data (IHDP), go to the directory and run the following command:
````
python run_experiments.py --data=ihdp
````
To reproduce the results on the fully-synthetic data for a specific setting from the thesis, e.g., setting 4, run the following command:
````
python run_experiments.py --setting=4
````
You can choose from settings 1 to 24.

### Options
The fully-synthetic experiment has 10 runs as default, the semi-synthetic has 100 runs. You can optionally change the
number uf runs using `--runs` for the fully-synthetic experiment and `--runs_ihdp` for the semi-synthetic experiment.
For example, to execute setting 10 with 2 runs:
````
python run_experiments.py --setting=10 --runs=2
````
To execute five runs of the semi-synthetic experiment:
````
python run_experiments.py --data=ihdp --runs_ihdp=5
````
