import pandas as pd
import scipy.stats as st

def drawdown(return_series: pd.Series):
    """
    Takes a time series of asset returns
    Computes and returns a Dataframe that contains:
    the wealth index
    previous peaks
    percent drawdowns
    """
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdown = (wealth_index-previous_peaks)/previous_peaks
    return pd.DataFrame({
        'Wealth': wealth_index,
        'Peaks': previous_peaks,
        'Drawdown': drawdown
    })

def get_ffme_returns():
    """
    Load the Fama-French dataset for the returns of the top and bottom deciles by market cap
    """
    me_m = pd.read_csv('Portfolios_Formed_on_ME_monthly_EW.csv', header=0, index_col=0, parse_dates=True, na_values=-99.99)
    rets = me_m[['Lo 10', 'Hi 10']]
    rets.columns = ['smallcap', 'largecap']
    rets = rets/100
    rets.index = pd.to_datetime(rets.index, format='%Y%m').to_period('M')
    return rets

def get_hfi_returns():
    """
    Load and format the EDHEC hedge fund index returns
    """
    hfi = pd.read_csv('edhec-hedgefundindices.csv', header=0, index_col=0, parse_dates=True)
    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')
    return hfi

def skewness(r):
    """
    Computes the skewness of the supplied series or dataframe
    Returns a float or a series
    Alternative to scipy.stats.skew()
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    """
    Computes the kurtosis of the supplied series or dataframe
    Returns a float or a series
    Alternative to scipy.stats.kurtosis()
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

def is_normal(r, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a series is normal or not
    Test is applied at the 1% level by default
    Returns True if the hypothesis of normality is accepted, False otherwise
    """
    statistic, p_value = st.jarque_bera(r)
    return p_value > level
