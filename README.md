# Cardinality Constrained Portfolio Optimisation

## Context

This project was originally a part of a group coursework during my Masterβs degree, in which I was tasked with implementing the logic of the algorithms into code. Since this was a lengthy process, the analysis was completed mainly by other team members. Being interested in this matter, I have therefore decided to start developing the project further with the aim of completing the analysis part myself.

## Introduction

In this project we aimed to use  and then compare the performance of Random Search and Genetic algorithms to optimise cardinality constrained portfolios. Given relatively extensive nature of the implementation and subsequent analysis the project is broken down into 4 parts:

- Implementation of Random Search algorithm (Random Search.py)
- Implementation of Genetic algorithm (Genetic Algorithm.py)
- Implementation of Genetic algorithm using different Population Sizes (Genetic Algorithm Population Size.py)
- Analysis of the results (Analysis.ipynb)

### Data
The data was given to us as part of the coursework in 5 .txt files. These 5 files represent 5 capital market trading indixes from around the world:
- `assets1.txt` - Hang Seng (Hong Kong)
- `assets2.txt` - DAX 100 (Germany)
- `assets3.txt` - FTSE 100(UK)
- `assets4.txt` - S&P 100 (USA)
- `assets5.txt` - Nikkei 225 (Japan)

## Each dataset is structured in the following fashion:
![Screenshot 2021-06-15 at 22 19 48](https://user-images.githubusercontent.com/85829899/122634002-e565c780-d0db-11eb-949a-bfc2c4c8989b.png)

## Main Objective

The main goal of the implementation was to minimize the following function 50 times, using 50 equally spaced values of π (from 0 to 1):

                                      π(π ) = π Β· πΆππππ(π ) β (1 β π) Β· π(π )  

Where: 						
                                
                                                πΆππππ = ββπ€ππ€ππππππππ 	

                                                     π = βπ€πππ
          
- s - a candidate solution (i.e. a portfolio)
- πΆππππ(π ) - Covariance of a portfolio
- π(π ) - Expected return of a portfolio
- π - the tradeoff between risk/return 
- π€π - proportion of total investment invested in asset i
- ππ  - expected return of asset i
- π€π - proportion of total investment invested in asset j
- ππj - correlation between assets i and j 
- ππ - standard deviation of return of asset i
- ππ - standard deviation of return of asset j

## Algorithm Implementation

The logic of the algorithms used is depicted below. The actual heuristics will not be discussed, however they are explained in great detail in T.-J. Chang, et. al (2000) paper, which was followed for transforming the algorithms into code. The paper can also be found in the repository. 

## Random Search Algorithm

![Screenshot 2021-06-15 at 21 58 05](https://user-images.githubusercontent.com/85829899/122634011-f7476a80-d0db-11eb-8a90-8397a1a59642.png)

## Genetic Algorithm

![Screenshot 2021-06-15 at 21 05 19](https://user-images.githubusercontent.com/85829899/122634018-ff070f00-d0db-11eb-83c1-39d504d6baa2.png)
 
## Limitations
Like many others, this project too has areas for improvement. The main one is the efficiency of codes for optimization of the portfolios (i.e. Random Search.py and Genetic Algorithm.py). Run times of these codes is quite significant. This is due to relatively high number of iterations (Number of assets in an index * 1000 for 50 values of π) coupled with ineffective ways for calaculations. Reducing run times of the codes by making their calculations more efficient, would significantly improve chances of the project being used for a real world application.

## Installation

### Install the requirements:

Install the requirements using `pip install -r requirements.txt`
-	You may also choose to use virtual environment for this

### To install the project you can choose to either:

#### A. Reproduce the analysis part:

In a directory of your choice, run:

 1. `git clone https://github.com/PeterZelinka/Cardinality-Constrained-Portfolio-Optimisation.git`

 2. Install [Anaconda](https://www.anaconda.com/products/individual)
 3. Open JupyterNotebook
 4. Navigate to the location where you cloned the directory
 5. Open and run Analysis.ipynb

#### B. Reproduce the whole project:

In a directory of your choice, run:

 1. `git clone https://github.com/PeterZelinka/Cardinality-Constrained-Portfolio-Optimisation.git`

Then run (one by one):

 2. `python Random\ Search.py`

 3. `python Genetic\ Algorithm.py`

 4. `python Genetic\ Algorithm\ Population\ Size.py`

(As stated above the running times are quite substantial)

The .csv files in `Generated data` will be overwritten or alternatively if you decide to remove `Generated data` then please make sure, before running the .py files, to run : 

- `mkdir Generated\ data/Different\ Lambdas`

- `mkdir Generated\ data/Different\ Populations`

After that repeat steps 2-5 in A.

## Credits
Big thanks to: Eileen Neumann, Alexandra Firkowska, Kristina Popelkova, Shriya Raina and Denis Mclaughlin who were part of my coursework team and have worked tirelessly on delivering the initial version of the project.
