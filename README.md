# Cardinality Constrained Portfolio Optimisation

## Context

This project was originally a part of a group coursework during my Master’s degree. As part of the team I was tasked with implementing the logic of the algorithms into code. Since this was a lengthy process, the analysis was completed mainly by other team members. Being interested in this matter I have therefore decided to start developing the project further with the aim of completing the analysis part myself.
Introduction

In this project we aimed to use  and then compare the performance of Random Search and Genetic algorithms to optimise a cardinality constrained portfolio. Given relatively extensive nature of the of the implementation and subsequent analysis the project is broken down into 3 parts:

Implementation of Random Search algorithm (Random Search.py)
Implementation of Genetic algorithm (Genetic Algorithm.py)
Implementation of Genetic algorithm using different Population Sizes (Genetic Algorithm Population Size.py)
Analysis of the results (BLA BLA)
Data
The data was given to us as part of the course work in 5 .txt files.:
- assets1.txt
- assets2.txt
- assets3.txt
- assets4.txt
- assets5.txt

These 5 files represent 5 capital market trading indices from around the world (respectively):
- Hang Seng (Hong Kong)
- DAX 100 (Germany)
- FTSE 100(UK)
- S&P 100 (USA)
- Nikkei 225 (Japan)





Each dataset is structured in the following fashion:

Main Objective
The main goal of the implementation was to minimize the following function 50 times, using 50 equally spaced values of 𝜆 (from 0 to 1):

                                                𝑓(𝑠) = 𝜆 · 𝐶𝑜𝑉𝑎𝑟(𝑠) − (1 − 𝜆) · 𝑅(𝑠)  

Where: 						
                                
                                                          𝐶𝑜𝑉𝑎𝑟 = ∑∑𝑤𝑖𝑤𝑗𝜌𝑖𝑗𝜎𝑖𝜎𝑗 	

                                                               𝑅 = ∑𝑤𝑖𝜇𝑖
          
- s - a candidate solution (in this case a portfolio)
- 𝐶𝑜𝑉𝑎𝑟(𝑠) - Covariance of a portfolio
- 𝑅(𝑠) - Expected return of a portfolio
- 𝜆 - expresses the tradeoff between risk/return 
- 𝑤𝑖 - proportion of total investment invested in asset i
- 𝜇𝑖  - expected return of asset i
- 𝑤𝑗 - proportion of total investment invested in asset j
- 𝜌𝑖j - correlation between assets i and j 
- 𝜎𝑖 - standard deviation of return of asset i
- 𝜎𝑗 - standard deviation of return of asset j

Algorithm Implementation
The heuristics of the algorithms used are depicted below. The actual heuristics will not be discussed in detail however it is explained in great detail in T.-J. Chang, et. al (2000) paper which was followed for transforming the algorithms into code. The paper can also be found in the repository. 
Random Search Algorithm








Genetic Algorithm

Reproducing the Analysis

To reproduce our analysis one shall first run Random Search.py, Genetic Algorithm.py and Genetic Algorithm Population Size.py ideally in the same folder as the datasets. The scripts will create a number of .csv file used for analysis presented in the NAME
