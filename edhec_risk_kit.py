import pandas as pd

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