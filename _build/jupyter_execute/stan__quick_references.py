#!/usr/bin/env python
# coding: utf-8

# $\mathcal{P}$

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import stan

from scipy import stats

plt.style.use('seaborn-darkgrid')


# In[2]:


# Rodar esse comando antes de import a stan (pystan versão 3.x)
import nest_asyncio
nest_asyncio.apply()


# In[3]:


# Data generating data
mu = 70
sigma = 100

data1 = stats.norm(mu, sigma).rvs(1000)


# In[4]:


# Essa estrutura está definida na página 77 - SR2

model_data1 = """ 
    // Dado é a variável que foi medida!
    data {   
        int N;
        real y[N];
    }
    
    // Parâmetro é a váriável não observada.
    parameters {  
        real mu;
        real<lower=0> sigma;
    }
    
    model {
        mu ~ normal(60, 10);
        sigma ~ exponential(1);
        
        y ~ normal(mu, sigma);
    }
"""

stan_data1 = {
    'N': len(data1),
    'y': data1,
}

posteriori = stan.build(model_data1, data=stan_data1)
results_data1 = posteriori.sample(num_chains=4, num_samples=1000)


# In[5]:


results_data1_mu = np.mean(results_data1['mu'])
results_data1_sigma = np.mean(results_data1['sigma'])

print('Inference of mu: ' , results_data1_mu, ' - Real value: ', mu)
print('Inference of sigma: ', results_data1_sigma, ' - Real value: ', sigma)

plt.hist(results_data1['mu'].flatten(), rwidth=0.9, density=True)
plt.title('Posteriori')
plt.xlabel('y')
plt.ylabel('probability')
plt.show()


# ### Stan -  Reference Manual

# ### 1. Comments Char
# 
# Comments line: `//` or `#`
# 
# Comments in block: `/* ... */`

# ### 2. Includes

# Use `#include` to import stan file in another file.
# For example, the `#include my-stan-file-function.stan` on top of file this will be replaced by content in `my-stan-file-function.stan`. 

# ### 3. Comments

# Example:
# 
# ```data {
#   int<lower=0> N;  // number of observations
#   array[N] real y;  // observations
# }```

# ### 4. Whitespace

# No indentation is need! 

# ### 5. Data Type and Declarations

# All variables should be declared in data type, like c\c++.
# 
# Stan is strong and static typing:
# 1. Force the programmer's declarate a variable.
# 2. Checking erros in compile time and flags erros.
# 3. Don't propagate errors ever to the results.

# #### 5.1 Overview of data type
# 

# 
# **Two primitive data type**: `real` and `int`.
# 
# 
# **Complex type**: `complex`, there is a complex number, real and imaginary component, both is `real`.
# 
# 
# **Vector and Matrix type:**:  `vector` to column vector, `row_vector` to row vectors, and `matrix`. (To complex type, `complex_vector`, `complex_row_vector` and `complex_matrix`)
# 
# 
# **Array types**: Any type can be made into an array type:
#     
#  - `array[5] real a;` Array, labeled `a`, that have 5 postions with real type;
#  
#  - `array[10, 2] int b;` Array with 10 lines and 2 columns of the int type, called `b`.
#  
#  - `array[10, 10, 5] matrix[3, 3] c;` Array with $[10, 10 , 5]$ positions to matrix$[3,3]$ format, called `c`.
#  
#  - `array[12, 8, 15] complex z;` Declare a array of the complex type.
# 
# 
# **Constrained data type**: This variables are parameters, is helpful provided them  with constraints to aid internal check erros.
# 
# - `int<lower=0> N;`
# 
# - `real<upper=0> log_p;`
# 
# - `vector<lower=-1, upper=1>[3] rho;`
# 
# 
# There are $4$ constrained vector data type:
# 
# 1. `simplex` to simple units.
# 
# 2. `unit-vector` to arrays\[ \] - unit-length vector.
# 
# 3. `ordered` to ordered vectors.
# 
# 4. `positive_ordered` to ordered positive vectors
# 
# 
# And there are constrained to matrix data type:
# 
# 
# 1. `corr_matrix` for correlations matrices (*symmetric, positive definite, unit diagonal*)
# 
# 
# 2. `cov_matrix` for covariance matrices (*symmetric, positive definite*)
# 
# 
# 3. `cholesky_factor_cov` is the Cholesky factors of covariance matrices(*lower triangular, positive diagonal, product with own transpose is a covariance matrix*)
# 
# 
# 4. `cholesky_factor_corr` is the Cholesky factors of correlations matrices (*lower triangular, positive diagonal, unit-length rows*)
# 
# 
# It's constrains will help check erros only in variables defined in the `data`, `transformed data`, `transformed parameters`, `generate quantities` blocks.
# 
# Unconstrained variable will be declared as real type (${\rm I\!R}^n$) by default. 

# #### 5.2 Primitive numerical data type

# **Integer Precision**: 32-bits (4-bytes) {$-2^{31}$, $2^{31} - 1$}
# 
# **Real Precision**: 64-bits (8-bytes), slightly larger than $+/- 10^{307} $, with until 15 decimal digits of accuracy.
# 
# **Not-a-number**: returns not-a-number functions errors if argument is not-a-number. And comparison operators: `not-a-number` == true, is *false* for every cases.
# 
# **Infinite values**: Great than all numbers, equivalent to negative case.

# #### 5.3 Complex numerical data type

# - `complex z = 2 - 1.3i;`
# 
# - `real re = get_real(z);  // re has value 2.0`
# 
# - `real im = get_imag(z);  // im has value -1.3`

# Promoting real to complex:
# 
# - `real x = 5.0;`
# 
# - `complex z = x;  // get_real(z) == 5.0, get_imag(z) == 0`

# #### 5.4 Scalar datatype and variable declarations

# - `int N;`  Unconstrained
# 
# -  `int<lower=1> N;`  $N >= 1, \forall$ $N$ in $\mathbb{Z} $
# 
# - `int<lower=0, upper=1> cond;`  $\{0, 1\}$ 

# - `real<lower=0> sigma;` $\sigma >=0 $

# - `real<upper=-1> x;` $x <= -1$

# - `real<lower=-1, upper=1> rho;` $-1 <= \rho <= 1$

# - `positive_infinity()` and `negative_infinty()` could be use to set limits, but this values are ignored in Stan.

# **Affinely transformed real**: The transformation:
# $$x ↦ \mu + \sigma * x$$
# 
# - $\mu$: Offset
# 
# - $\sigma$: Multiplier (positive)
# 
# Like constraint declarations, making the sampling process more efficient. Like a soft constraint:
# 
# 
# - `real<offset=1> x;`   $1 + 1 \times x$
# 
# - `real<multiplier=2>;`   $0 + 2\times x$
# 
# - `real<offset=1, multiplier=2> x;`   $1 + 2\times x$
# 

# Example:
# 
# ```
# parameters {
#   real<offset=mu, multiplier=sigma> x;
# }
# model {
#   x ~ normal(mu, sigma);
# }
# 
# ```

# The theorical model that received the data from $x ~ normal(0, 1)$, can writer in stan model like:
# 
# ```
# parameter {
#     real x;  // This x ~ normal(0, 1)
# }
# 
# model {
#     x ~ normal(mu, sigma);
# }
# ```
# 
# this code is equivalent to:
# 
# ```
# parameter {
#     real<offset=0, multiplier=1> x;  // This x ~ normal(0, 1)
# }
# model {
#     x ~ normal(mu, sigma);
# }
# ```

# **Expressions as bounds and offset/multiplier**: We can use the variables, that have been declared before, to setting the values of the offset and multiplier.
# 
# ```
# data {
#     real lb;
# }
#  
# parameters {
#     rea<lower=lb> phi;
# }
# ```

# Variables used in constraints can be any variable that has been defined at the point the constraint is used. For instance:
# 
# ```
# data {
#    int<lower=1> N;
#    array[N] real y;
# }
# parameters {
#    real<lower=min(y), upper=max(y)> phi;
# }
# ```

# **Declaring optional variable**: Variable that depends on a boolean constant.
# 
# ```
# data {
#     int<lower=0, upper=1> include_alpha;   // Only {0, 1}
# }
# parameters {
#     vector[include_alpha ? N : 0] alpha;
# }
# ```
# 
# If `include_alpha == True` then `alpha` vector exists, else  it will be exclude in output results automaticaly.

# #### 5.5 Vector and matrix data types

# Three types of container objects: `arrays`, `vector` and `matrix`. Vector and matrices are structure limited, vector 1-dimensional real or complex values. Matrix that two dimensional. Array is not matrix.

# ######  Vector:
# 
# `vector[3] u;`  3-dimensional real vector.
# 
# `vector<lower=0>[3] u;`  vector with non-negative values.
# 
# `vector<offset=42, multiplier=3>[3] u;` vector with offset and multiplier
# 

# ###### Complex vectors
# 
# `complex_vector[3] v;`
# 
# it's do not support any constraints

# ###### Unit vector
#  
# `unit_vector[5] theta;` Is declared to be a unit $5-vector$. Useful to validate unit length.
# 

# ###### Ordered vector
# 
# `ordered[5] c;`  All entries are sorted in ascending order. The vector often employed as cut points in oderder logistic regression models.

# ###### Positive, ordered vector
# 
# `positive_ordered[5] d;`  Vector with positive real values and sorted ascending.

# ###### Row vectors
# 
# `row_vector[1093] u;` It's a 1093-dimensional row vector.
# 
# `row_vector<lower=-1, upper=1>[10] u;`
# 
# `row_vector<offset=-42, multiplier=3>[3] u;` 

# ###### Complex row Vectors
# 
# `complex_row_vector[12] v;` 
# 
# Not allow constraints.

# ###### Matrices
# 
# `matrix[M, N] A;` Where $M$ and $N$ are integer type.
# 
# `matrix<upper=0>[3, 4] B;` Matrix with positive values.
# 
# `matrix<offset=3, multiplier=2>[4, 3] C;` Matrix with offset and multiplier
# 
# `matrix<multiplier=2>[4, 3] C;` Matrix with just multiplier.

# ###### Assigning to rows of matrix
# 
# `matrix[M, N] a;`
# 
# `row_vector[N] b;`
# 
# ...
# 
# `a[1] = b;`
# 
# Copies the values row vector `b` to `a[1]`, where `a[1]` is the first row of matrix `a`.

# ##### Covariance matrices
# 
# `cov_matrix[k] Omega;` It's a $k \times k$ covariance matrix, symmetric and positive definite.

# ###### Correlation matrices
# 
# `corr_matrix[3] Sigma;` symmetric, positive definite has entries between $-1$ and $1$ and has a unit diagonal.

# ###### Cholesky factor of covariance matrices
# 
# This a better than use covariance matrix directly.
# 
# `cholesky_factor_cov[4] L;` Where $\Sigma = LL^{T}$ and $\Sigma$ is a covariance matrix.
# 
# 

# ##### Cholesky factors of positive semi-definite matrices
# 
# We also use the general declarations to cholesky factor.
# 
# `cholesky_factor_cov[M, N];` To be a positive semi-definite matrices of rank M.

# ###### Cholesky factors of correlation matrices
# 
# `cholesky_factor_corr[k] L;`  Represent Cholesky factor of a correlation matrix.

# ###### Assigning constrained variables
# 
# Constrained are not block to assigning between variable with same primitive data. 

# - `real` with `real<lower=0, upper=1>`
# 
# - `matrix[3,3]` with `cov_matrix[3]`
# 
# - `matrix[3,3]` with `cholesky_factor_cov[3]`

# ###### Expressions as size declarations
# 
# Declare once the data and using in other blocks.
# 
# `
# data {
#     int<lower=0> N_observed, N_missing;
# }
# transformed parameters {
#     vector[N_observed + N_missing] y;
# }
# `

# ###### Accessing vector and matrix elements
# 
# `
# matrix[M, N] m;
# row_vector[N] v;
# real x;
# //...
# v = m[2];  // m[2] is row_vector
# x = v[3];  // equivalent to x = m[2][3] or x = m[2, 3]
# `

# ###### Array index style
# 
# The more efficient form to access array is by `m[2, 3]`.

# ###### Size declaration restrictions
# 
# `vector[M + N] y;`  Also to matrices and arrays.

# #### 5.6 Array data type
# 
# https://mc-stan.org/docs/reference-manual/array-data-types.html
