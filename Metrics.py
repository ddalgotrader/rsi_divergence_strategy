import numpy as np
import pandas as pd


def simple_return(returns):
    return returns.cumsum().apply(np.exp)[-1]
    
def mean_return(returns):

    annual=(returns.count() / ((returns.index[-1] - returns.index[0]).days / 365))
    mean_ret=returns.mean()*annual
    return mean_ret


def stddev(returns):

    annual=(returns.count() / ((returns.index[-1] - returns.index[0]).days / 365))
    annual_stddev=returns.std()*np.sqrt(annual)
    return annual_stddev


def sharpe_ratio(returns, rf_rate=0):

    annual=np.sqrt(returns.count() / ((returns.index[-1] - returns.index[0]).days / 365))
    ratio=(returns.mean()-rf_rate)/returns.std()*annual
    return ratio


def sortino_ratio(returns, dr=0):

    annual=np.sqrt(returns.count() / ((returns.index[-1] - returns.index[0]).days / 365))
    dreturns=(returns-dr)
    downside=np.where(dreturns<0, dreturns,0)
    ddeviation=np.sqrt(np.mean(downside**2))
    if ddeviation==0:
        return np.nan
    else:
        ratio=(returns.mean()-dr)/ddeviation * annual
        return ratio


def max_dd(returns):

    cret=returns.cumsum()
    ret=np.exp(cret)
    cmax=ret.cummax()
    dd=(cmax-ret)/cmax
    max_dd=dd.max()
    return max_dd


def cagr(returns):

    annual=(returns.index[-1] - returns.index[0]).days / 365
    ret=np.exp(returns.sum())
    cagr=ret**(1/annual)-1
    return cagr


def calmar_ratio(returns):
    mdd=max_dd(returns)
    if mdd==0:
        return np.nan
    else:
        cagr_val=cagr(returns)
        ratio=cagr_val/mdd
        return ratio


def kelly(returns):

    win=np.where(returns>=0,1,0).sum()
    loss=np.where(returns<0,1,0).sum()
    w=(win/(win+loss))
    win_loss=win/loss
    kelly=w-((1-w)/win_loss)
    return kelly
