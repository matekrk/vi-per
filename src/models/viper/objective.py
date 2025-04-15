import math
import torch
import torch.distributions as dist
from torch.special import log_ndtr, ndtr
from torch.nn.functional import logsigmoid

"""## Logistic LL: Original functions"""

def KL_mvn(m, S, mu, Sig):
    """
    Can also be computed via:
        mvn1 = dist.MultivariateNormal(m, S)
        mvn2 = dist.MultivariateNormal(mu, Sig)
        dist.kl.kl_divergence(mvn1, mvn2)
    """
    p = m.size()[0]
    res = 0.5 * (torch.logdet(Sig) - torch.logdet(S) -  p +
                 torch.trace(Sig.inverse() @ S) +
                 (mu - m).t() @ Sig.inverse() @ (mu - m))
    return res


def KL(m, s, mu, sig):
    """
    Compute the KL divergence between two Gaussians
    :param m: mean of variational distribution
    :param s: standard deviation of variational distribution
    :param mu: mean of prior
    :parma sig: standard deviation of prior
    :return: KL divergence
    """
    res = torch.log(sig / s) + 0.5 * ((s ** 2 + (m - mu) ** 2) / sig ** 2 - 1)
    return torch.sum(res)


def KL_MC(m, s, mu, sig):
    """
    Compute the KL divergence between two Gaussians with monte carlo
    :param m: mean of variational distribution
    :param s: standard deviation of variational distribution
    :param mu: mean of prior
    :parma sig: standard deviation of prior
    :return: KL divergence
    """
    d1 = dist.Normal(m, s)
    d2 = dist.Normal(mu, sig)

    x = d1.sample((1000,))
    return torch.mean(torch.sum(d1.log_prob(x) - d2.log_prob(x), 1))


def neg_ELL_TB(m, s, y, X, l_max = 10.0, XX=None):
    """
    Compute the expected negative log-likelihood
    :return: -ELL
    """
    M = X @ m

    if XX is None:
        S = torch.sum(X ** 2 * s ** 2, dim=1)
    else:
        S = torch.sum(XX * s ** 2, dim=1)

    S = torch.sqrt(S)

    if torch.any(S < 1e-10):
        print("Warning: Very small S values detected - clamp to 1e-10")
        S = torch.clamp(S, min=1e-10)

    l = torch.arange(1.0, l_max*2, 1.0, requires_grad=False, dtype=torch.float64).to(M.device)

    M = M.unsqueeze(1)
    S = S.unsqueeze(1)
    l = l.unsqueeze(0)

    res =  \
        torch.dot(- y, X @ m) + \
        torch.sum(
            S / math.sqrt(2 * torch.pi) * torch.exp(- 0.5 * M**2 / S**2) + \
            M * ndtr(M / S)
        ) + \
        torch.sum(
            (-1.0)**(l - 1.0) / l * (
                torch.exp( M @ l + 0.5 * S**2 @ (l ** 2) + log_ndtr(-M / S - S @ l)) + \
                torch.exp(-M @ l + 0.5 * S**2 @ (l ** 2) + log_ndtr( M / S - S @ l))
            )
        )

    return res

def neg_ELL_TB_mvn(m, S, y, X, l_max = 10.0):
    """
    Compute the expected negative log-likelihood
    :return: -ELL
    """
    M = X @ m
    # this might be faster
    # S = (X.unsqueeze(1) @ S @ X.unsqueeze(2)).squeeze()

    try:
        U = torch.linalg.cholesky(S)
        S = torch.sum((X @ U) ** 2, dim=1)
    except:
        S = torch.sum(X * (S @ X.t()).t(), dim=1)

    S = torch.sqrt(S)

    if torch.any(S < 1e-10):
        print("Warning: Very small S values detected - clamp to 1e-10")
        S = torch.clamp(S, min=1e-10)

    l = torch.arange(1.0, l_max*2, 1.0, requires_grad=False, dtype=torch.float64).to(M.device)

    M = M.unsqueeze(1)
    S = S.unsqueeze(1)
    l = l.unsqueeze(0)

    res =  \
        torch.dot(- y, M.squeeze()) + \
        torch.sum(
            S / math.sqrt(2 * torch.pi) * torch.exp(- 0.5 * M**2 / S**2) + \
            M * ndtr(M / S)
        ) + \
        torch.sum(
            (-1.0)**(l - 1.0) / l * (
                torch.exp( M @ l + 0.5 * S**2 @ (l ** 2) + log_ndtr(-M / S - S @ l)) + \
                torch.exp(-M @ l + 0.5 * S**2 @ (l ** 2) + log_ndtr( M / S - S @ l))
            )
        )

    return res


def neg_ELL_MC(m, s, y, X, n_samples=1000):
    """
    Compute the expected negative log-likelihood with monte carlo
    :return: -ELL
    """

    M = X @ m
    # S = torch.sqrt(X ** 2 @ s ** 2)
    S = torch.sum(X ** 2 * s ** 2, dim=1)
    S = torch.sqrt(S)

    norm = dist.Normal(torch.zeros_like(M), torch.ones_like(S))
    samp = norm.sample((n_samples, ))
    samp = M + S * samp

    res =  torch.dot( - y, M) + \
        torch.sum(torch.mean(torch.log1p(torch.exp(samp)), 0))

    return res


def neg_ELL_MC_mvn(m, S, y, X, n_samples=1000):
    """
    Compute the expected negative log-likelihood with monte carlo
    :return: -ELL
    """
    M = X @ m
    # S = torch.diag(X @ S @ X.t())

    try:
        U = torch.linalg.cholesky(S)
        S = torch.sum((X @ U) ** 2, dim=1)
    except:
        S = torch.sum(X * (S @ X.t()).t(), dim=1)

    S = torch.sqrt(S)

    norm = dist.Normal(torch.zeros_like(M), torch.ones_like(S))
    samp = norm.sample((n_samples, ))
    samp = M + S * samp

    res =  torch.dot( - y, M) + \
        torch.sum(torch.mean(torch.log1p(torch.exp(samp)), 0))

    return res


def neg_ELL_Jak(m, s, t, y, X):
    """
    Compute the expected negative log-likelihood using the bound introduced
    by Jaakkola and Jordan (2000)
    :return: -ELL
    """
    M = X @ m
    a_t = (torch.sigmoid(t) - 0.5) / t
    S = torch.diag(s**2) + torch.outer(m, m)

    try:
        U = torch.linalg.cholesky(S)
        B = a_t * torch.sum((X @ U) ** 2, dim=1)
    except:
        B = a_t * torch.sum(X * (S @ X.t()).t(), dim=1)

    res = - torch.dot(y, M) - torch.sum(logsigmoid(t)) + \
        0.5 * torch.sum(M + t) + 0.5 * torch.sum(B)   - \
        0.5 * torch.sum(a_t * t ** 2)

    return res


def neg_ELL_Jak_mvn(m, S, t, y, X):
    """
    Compute the expected negative log-likelihood using the bound introduced
    by Jaakkola and Jordan (2000)
    :return: -ELL
    """
    M = X @ m
    a_t = (torch.sigmoid(t) - 0.5) / t
    SS = S + torch.outer(m, m)

    try:
        U = torch.linalg.cholesky(SS)
        B = a_t * torch.sum((X @ U) ** 2, dim=1)
    except:
        B = a_t * torch.sum(X * (SS @ X.t()).t(), dim=1)

    res = - torch.dot(y, M) - torch.sum(logsigmoid(t)) + \
        0.5 * torch.sum(M + t) + 0.5 * torch.sum(B)   - \
        0.5 * torch.sum(a_t * t ** 2)

    return res


def ELBO_TB(m, u, y, X, mu, sig, l_max = 10.0, XX=None):
    """
    Compute the negative of the ELBO
    :return: ELBO
    """
    s = torch.exp(u)
    return neg_ELL_TB(m, s, y, X, l_max=l_max, XX=XX) + KL(m, s, mu, sig)


def ELBO_TB_mvn(m, u, y, X, mu, Sig, l_max = 10.0):
    """
    Compute the negative of the ELBO
    :return: ELBO
    """
    p = Sig.size()[0]
    L = torch.zeros(p, p, dtype=torch.double)
    L[torch.tril_indices(p, p, 0).tolist()] = u
    S = L.t() @ L

    return neg_ELL_TB_mvn(m, S, y, X, l_max=l_max) + KL_mvn(m, S, mu, Sig)


def ELBO_MC(m, u, y, X, mu, sig, n_samples=1000):
    """
    Compute the negative of the ELBO
    :return: ELBO
    """
    s = torch.exp(u)
    return neg_ELL_MC(m, s, y, X, n_samples) + KL(m, s, mu, sig)


def ELBO_MC_mvn(m, u, y, X, mu, Sig, n_samples=1000):
    """
    Compute the negative of the ELBO
    :return: ELBO
    """
    p = Sig.size()[0]
    L = torch.zeros(p, p, dtype=torch.double)
    L[torch.tril_indices(p, p, 0).tolist()] = u
    S = L.t() @ L

    return neg_ELL_MC_mvn(m, S, y, X, n_samples) + KL_mvn(m, S, mu, Sig)


def ELBO_Jak(m, s, t, y, X, mu, sig):
    """
    Compute the negative of the ELBO using the bound introduced by
    Jaakkola and Jordan (2000)
    :return: ELBO
    """
    return neg_ELL_Jak(m, s, t, y, X) + KL(m, s, mu, sig)


def ELBO_Jak_mvn(m, S, t, y, X, mu, Sig, cov=None):
    """
    Compute the negative of the ELBO using the bound introduced by
    Jaakkola and Jordan (2000)
    :return: ELBO
    """
    return neg_ELL_Jak_mvn(m, S, t, y, X) + KL_mvn(m, S, mu, Sig)

"""## Logistic LL: Multi-head functions"""

def KL_MH(m_list, s_list, mu_list, sig_list):
    total_KL = sum(KL(m, s, mu, sig) for m, s, mu, sig in zip(m_list, s_list, mu_list, sig_list))
    return total_KL

def KL_mvn_MH(m_list, S_list, mu_list, Sig_list):
    total_KL = sum(KL_mvn(m, S, mu, Sig) for m, S, mu, Sig in zip(m_list, S_list, mu_list, Sig_list))
    return total_KL

def neg_ELL_TB_MH(m_list, s_list, y_list, X, l_max=10.0, XX=None):
    total_neg_ELL = sum(neg_ELL_TB(m, s, y, X, l_max=l_max, XX=XX) for m, s, y in zip(m_list, s_list, y_list))
    return total_neg_ELL

def neg_ELL_TB_mvn_MH(m_list, S_list, y_list, X, l_max=10.0):
    total_neg_ELL = sum(neg_ELL_TB_mvn(m, S, y, X, l_max=l_max) for m, S, y in zip(m_list, S_list, y_list))
    return total_neg_ELL

def neg_ELL_MC_MH(m_list, s_list, y_list, X, n_samples=1000):
    total_neg_ELL = sum(neg_ELL_MC(m, s, y, X, n_samples=n_samples) for m, s, y in zip(m_list, s_list, y_list))
    return total_neg_ELL

def neg_ELL_MC_mvn_MH(m_list, S_list, y_list, X, n_samples=1000):
    total_neg_ELL = sum(neg_ELL_MC_mvn(m, S, y, X, n_samples=n_samples) for m, S, y in zip(m_list, S_list, y_list))
    return total_neg_ELL
