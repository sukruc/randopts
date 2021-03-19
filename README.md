# RANDomized stuff On randomized oPTimization - RANDOPTS

This package offers some tools on randomized optimization, namely preset problems,
utilities to create new problems and predefined optimization algorithms providing
limited customization in terms of parameters.

## Optimizers:
- HillClimber
- RandomizedRestarts (can be used in combination with any optimizer)
- Simulated Annealing
- Genetic Algorithm
- MIMIC

## Predefined Problems:
- Alternating bits (1010101)
- Bulls & cows
- Knapsack
- Guess my bits
- Neural Network weight optimization

## Usage

Below is a generic usage for problems and solvers which is applicable to all solver and problem classes offered under this package:

```python
import randopts as rd

>>> problem = rd.problem.AlternateProblem(nbits=7)
>>> solver = rd.solver.HillClimber()
>>> solver.fit(problem)  # parses parameters such as required bits
>>> solver.ara(problem)  # attempts to solve the problem
'1010101'

>>> problem.query_cost_td
45.0

>>> solver.maks
6

>>> solver.solution_arr
[3,
 3,
...
 5,
 6]

>>> solver.solution_time
0.001
```
