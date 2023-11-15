import math

import torch
import torch.autograd as autograd
import torch.distributions as dist

from importlib import reload

from _01_data_generation import generate_data
from _02_method import *
from _00_funcs import *



torch.manual_seed(1)

# generate data
dat = generate_data(250, 4)

dat["X"].dtype
dat["y"].dtype
dat["b"].dtype

# set prior params
mu = torch.zeros(4, dtype=torch.double)
sig = torch.ones(4, dtype=torch.double) * 4

# parameters of the variational distribution
m = torch.randn(4, requires_grad=True, dtype=torch.double)
u = torch.tensor([-1.,-1., -1., -1.], requires_grad=True, dtype=torch.double)
s = torch.exp(u)

t = torch.ones(250, dtype=torch.double, requires_grad=True)

# check if values of functions match
KL(m, s, mu, sig)
KL_MC(m, s, mu, sig)
d1 = dist.Normal(mu, sig)
d2 = dist.Normal(m, s)
torch.sum(torch.distributions.kl.kl_divergence(d2, d1))

# check for MultiVariateNormal
KL_mvn(m, torch.diag(s), mu, torch.diag(sig)) 
d1 = dist.MultivariateNormal(mu, torch.diag(sig))  
d2 = dist.MultivariateNormal(m, torch.diag(s))
torch.distributions.kl.kl_divergence(d2, d1)   


ELL_MC(m, s, dat["y"], dat["X"])
ELL_TB(m, s, dat["y"], dat["X"])
ELL_Jak(m, s, t, dat["y"], dat["X"])

ELBO_MC(m, s, dat["y"], dat["X"], mu, sig)
ELBO_TB(m, s, dat["y"], dat["X"], mu, sig)
ELBO_Jak(m, u, t, dat["y"], dat["X"], mu, sig)


autograd.gradcheck(ELBO_TB, (m, s, dat["y"], dat["X"], mu, sig)) 
autograd.gradcheck(ELBO_Jak, (m, s,t,  dat["y"], dat["X"], mu, sig)) 
autograd.gradcheck(ELBO_MC, (m, u, dat["y"], dat["X"], mu, sig)) 


from importlib import reload
import sys
reload(sys.modules["_02_method"])

dat = generate_data(5000, 25, dgp=2, seed=99)

f0 = LogisticVI(dat, method=0, intercept=False, verbose=True)
f0.fit()
f0.runtime

f1 = LogisticVI(dat, method=1, intercept=False, verbose=True, n_iter=1500)
f1.fit()
f1.runtime

f6 = LogisticMCMC(dat, intercept=False, n_iter=1000, burnin=500, verbose=True)
f6.fit()

from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        f1.fit()
