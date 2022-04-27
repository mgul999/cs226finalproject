# CS 226r Final Project: Reconstruction Attacks 
## Jordan Barkin, Michael Gul, Jessica Shand 

### Overview 

This repository contains the code for our final project for CS 226r. For the project, we implemented multiple reconstruction attacks
along with differentially private mechanisms, and we ran a number of experiments in order to investigate how these attacks worked and how differential privacy can thwart them.

### Structure 
* attacks.py: Implementations of the attacks from papers by Dinur and Nissim (2003) and Cohen and Nissim (2019).
* mechanisms.py: Implementations of mechanisms to answer subset sum queries. In particular, we implemented the trival, Laplace, and Gaussian mechanisms. 
* run_experiments.py: Functions for running experiments on the different parameters for the attacks and mechanisms.
* utils.py: Helper functions used in the experiments.
* visualization.py: Functions for generating figures based on experimental data.
* experiments.ipynb: A Jupyter notebook for running the experiments and printing the visualizations of the results. 
* Hospital Data Attack.ipynb: a Jupyter notebook that executes the Cohen-Nissim attack on a medical dataset from the New York State Department of Health. 
