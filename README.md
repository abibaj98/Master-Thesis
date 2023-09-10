# Conditional Average Treatment Effect Estimation via Meta-Learners
###### Master Thesis by Arberim Bibaj (August 2023, ETHZ, MSc Statistisc)
In the following you see how to reproduce the results and a short description of the files.

## Description of the Thesis

## Reproduction of the Results
First, please make sure to install the required packages on Python (3.11.2.) with the corresponding versions:
````
pip install -r requirements.txt
````
Additionally, you need to install R (4.2.2) and the package mpower (0.1.0) as it is used for the Data Generating Process.

Next, to reproduce the results on the semi-synthetic data (IHDP), go to the directory and run the following command:
````
python run_experiments.py --data=ihdp
````
To reproduce the results on the fully-synthetic data for a specific setting from the thesis, e.g., setting 4, run the following command:
````
python run_experiments.py --setting=4
````
You can choose from settings 1 to 24. 

The fully-synthetic experiment has 10 run as default, the semi-synthetic has 100 run. You can optionally change the
number uf runs using `--runs` for the fully-synthetic experiment and `--runs_ihdp` for the semi-synthetic experiment.
For example:
````
python run_experiments.py --setting=10 --runs=2
````
to execute setting 10 with 2 runs.
or:
````
python run_experiments.py --data=ihdp --runs_ihdp=1
````
to do one run of the semi-synthetic experiment.
