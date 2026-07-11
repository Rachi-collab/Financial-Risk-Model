# Financial-Risk-Model

Intern Id: CITS2603
---

A Python project for portfolio risk modeling...

## Project structure

- `main.py` - command-line pipeline that loads sample portfolio data, calculates performance and risk metrics, and prints a summary report.
- `dashboard/app.py` - Streamlit dashboard for interactive risk analysis and visualization.
- `analysis/` - performance, drawdown, backtesting, and risk test utilities.
- `data/` - data loading, return calculation, and price-fetching utilities.
- `models/` - risk model implementations such as VaR and volatility estimation.

## Requirements

- Python 3.9+
- pandas
- numpy
- scipy
- streamlit
- plotly

Install dependencies with:

```powershell
pip install -r requirements.txt
```

> If `requirements.txt` is not present, install packages manually:
> `pip install pandas numpy scipy streamlit plotly`

## Usage

Run the main report:

```powershell
python main.py
```

Launch the interactive dashboard:

```powershell
streamlit run dashboard/app.py
```

## Notes

- The command-line pipeline uses a sample portfolio and computes VaR/CVaR, EWMA volatility, and Kupiec backtest statistics.
- The dashboard supports configurable tickers, portfolio weights, VaR confidence levels, and date range selection.
- Add or extend models, data sources, and analysis functions in the respective package folders.
