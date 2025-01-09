import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Example data for daily returns
data = {
    'Asset A': [0.001, 0.002, -0.001, 0.003, 0.001],
    'Asset B': [0.002, -0.001, 0.003, 0.001, 0.002],
    'Asset C': [0.001, 0.001, -0.002, 0.002, 0.002],
    'Asset D': [0.003, 0.002, -0.001, 0.001, 0.003]
}

# Convert to DataFrame
returns = pd.DataFrame(data)

# Calculate the mean returns and covariance matrix
mean_returns = returns.mean()
cov_matrix = returns.cov()

# 2. Generate random portfolios
num_assets = len(mean_returns)
num_portfolios = 10000
results = np.zeros((num_portfolios, 3))

for i in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)

    portfolio_return = np.sum(weights * mean_returns) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))

    sharpe_ratio = (portfolio_return - 0.01) / portfolio_volatility

    results[i, 0] = portfolio_return
    results[i, 1] = portfolio_volatility
    results[i, 2] = sharpe_ratio

# 3. Convert results to a DataFrame
results_df = pd.DataFrame(results, columns=['Return', 'Volatility', 'Sharpe Ratio'])

# 4. Plot the Efficient Frontier
plt.figure(figsize=(10, 6))
plt.scatter(results_df['Volatility'], results_df['Return'], c=results_df['Sharpe Ratio'], cmap='viridis', marker='o')
plt.title('Efficient Frontier')
plt.xlabel('Volatility (Risk)')
plt.ylabel('Return')
plt.colorbar(label='Sharpe Ratio')
plt.show()
