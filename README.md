# Capstone Project

## Overview

This repo (https://github.com/richardbatchelor/capstone) contains my response to the Capstone project challenge with documentation and Python notebooks that I used in my application of Bayesian optimisation for each function and the inputs and outputs as each week of the project progresses.

The _Capstone_ 13-week project is part of the [_Professional Certificate in Machine Learning and Artificial Intelligence_](https://www.imperial.ac.uk/business-school/executive-education/technology-analytics-data-science/professional-certificate-machine-learning-and-artificial-intelligence-programme/online/) offered by [Imperial College Business School](https://www.imperial.ac.uk/business-school/). It is a simulated machine learning problem with 8 real-world black box optimisation challenges.
Each challenge is a real-world maximisation problem with a description explaining the function, the data and output structure, along with an initial set of valid input and output data provided.

The purpose of the project is twofold:
- To gain and demonstrate hands-on proficiency in Bayesian optimisation, an industry-relevant approach for machine learning scenarios where evaluation is expensive, time-consuming, or resource-intensive.
- To create a portfolio project showcasing applied machine learning optimisation techniques in realistic settings.

## Inputs and Outputs

From a mathematical point of view the black box challenges can be considered as separate unknown functions with n-dimensions input and 1d output.   The inputs are normalised for each input dimension in the range 0..1.  For each function, some initial sample points are provided.

- Function 1:  2D – 10 initial samples
- Function 2:  2D – 10 initial samples
- Function 3:  3D – 15 initial samples
- Function 4:  4D – 30 initial samples
- Function 5:  4D – 20 initial samples
- Function 6:  5D – 20 initial samples
- Function 7:  6D – 30 initial samples
- Function 8:  8D – 40 initial samples

All inputs and outputs are provided to 6 dps.  

Query point submissions may be uploaded to a website in a ‘-‘ delimited string format, such as `0.407603-0.411492-0.350202-0.437970` for function 4.
The response is a single output value provided by email response within 48hrs. Serialised Numpy array files are also provided in the email response that are the cumulative submissions to date and their appropriate evaluated results.

_Note_: The real world description of each function is documented in [FunctionDescriptions.md](./FunctionDescriptions.md)

## Objective

The project objective is to find the inputs for each function that provide the maximum output possible. Each week a new submission of a single query point for each function can be made, and a response will be provided shortly after that provides the output for the provided input query point.

## Approach

I have approached the project as a Bayesian optimization problem adopting a Gaussian Process surrogate function to fit the sample data I have and then using acquisition functions to propose the next set of samples to gain more information about the unknown functions.

I expect to spend a first significant period of the Capstone project focussed on exploring the function input space, tuning the acquisition functions used to propose the next points to evalkuate accordingly.

### Week 1
I used an Upper Context Bound acquisition function with kappa at 6 to encourage some initial exploration of the less certain areas, with a random sampling of points to propose the submissions.

### Week 2
I experimented with random search vs a grid search for the acquisition function evaluation, with the latter being very time consuming. I did some research which led to me changing my approach to use a Sobol sequence for the inputs for the acquisition function input spaces to provide a more balanced search with lots of point evaluation before taking a more limited set for minimising to reduce the evaluation time. I also experimented with Expected Improvement (xi = 0.01) and UCB( K=6,20) acquisition functions but in the end submitted the EI points suggested since they looked more uniformly spread with less extreme points.

### Week 3
The approach I have taken is to generate suggested points to search using EI with Xi from 0 to 5 and UCB with kappa values ranging from 0.1 to 20. I have then compared the points suggested by each approach and made an observational judgement call on which point to select for each function – but in all cases looking for more exploratory points.
