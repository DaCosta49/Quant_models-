import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd

class BlackScholes:
    def __init__(self):
        """Initialize Black-Scholes Option Pricing Model"""
        pass
    
    @staticmethod
    def d1(S, K, T, r, sigma):
        """Calculate d1 parameter"""
        return (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    
    @staticmethod
    def d2(S, K, T, r, sigma):
        """Calculate d2 parameter"""
        return (np.log(S/K) + (r - sigma**2/2)*T) / (sigma*np.sqrt(T))
    
    def call_price(self, S, K, T, r, sigma):
        """
        Calculate call option price using Black-Scholes formula
        
        Parameters:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate
        sigma (float): Volatility of the underlying asset
        
        Returns:
        float: Call option price
        """
        d1 = self.d1(S, K, T, r, sigma)
        d2 = self.d2(S, K, T, r, sigma)
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    
    def put_price(self, S, K, T, r, sigma):
        """Calculate put option price using Black-Scholes formula"""
        d1 = self.d1(S, K, T, r, sigma)
        d2 = self.d2(S, K, T, r, sigma)
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    
    def call_delta(self, S, K, T, r, sigma):
        """Calculate call option delta"""
        return norm.cdf(self.d1(S, K, T, r, sigma))
    
    def put_delta(self, S, K, T, r, sigma):
        """Calculate put option delta"""
        return -norm.cdf(-self.d1(S, K, T, r, sigma))
    
    def call_gamma(self, S, K, T, r, sigma):
        """Calculate option gamma (same for call and put)"""
        d1 = self.d1(S, K, T, r, sigma)
        return norm.pdf(d1)/(S*sigma*np.sqrt(T))
    
    def call_vega(self, S, K, T, r, sigma):
        """Calculate option vega (same for call and put)"""
        d1 = self.d1(S, K, T, r, sigma)
        return S*np.sqrt(T)*norm.pdf(d1)
    
    def call_theta(self, S, K, T, r, sigma):
        """Calculate call option theta"""
        d1 = self.d1(S, K, T, r, sigma)
        d2 = self.d2(S, K, T, r, sigma)
        theta = -S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2)
        return theta
    
    def put_theta(self, S, K, T, r, sigma):
        """Calculate put option theta"""
        d1 = self.d1(S, K, T, r, sigma)
        d2 = self.d2(S, K, T, r, sigma)
        theta = -S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2)
        return theta
    
    def plot_option_surface(self, S_range, T_range, K, r, sigma, option_type='call'):
        """
        Plot 3D surface of option prices
        
        Parameters:
        S_range (array): Range of stock prices
        T_range (array): Range of times to maturity
        K (float): Strike price
        r (float): Risk-free rate
        sigma (float): Volatility
        option_type (str): 'call' or 'put'
        """
        S_mesh, T_mesh = np.meshgrid(S_range, T_range)
        Z = np.zeros_like(S_mesh)
        
        for i in range(len(T_range)):
            for j in range(len(S_range)):
                if option_type.lower() == 'call':
                    Z[i,j] = self.call_price(S_range[j], K, T_range[i], r, sigma)
                else:
                    Z[i,j] = self.put_price(S_range[j], K, T_range[i], r, sigma)
        
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surface = ax.plot_surface(S_mesh, T_mesh, Z, cmap='viridis')
        
        plt.colorbar(surface)
        ax.set_xlabel('Stock Price')
        ax.set_ylabel('Time to Maturity')
        ax.set_zlabel('Option Price')
        plt.title(f'Black-Scholes {option_type.capitalize()} Option Prices')
        plt.show()

def main():
    # Initialize Black-Scholes model
    bs = BlackScholes()
    
    # Example parameters
    S = 100  # Current stock price
    K = 100  # Strike price
    T = 1.0  # Time to maturity (1 year)
    r = 0.05  # Risk-free rate (5%)
    sigma = 0.2  # Volatility (20%)
    
    # Calculate option prices
    call_price = bs.call_price(S, K, T, r, sigma)
    put_price = bs.put_price(S, K, T, r, sigma)
    
    print(f"Call Option Price: ${call_price:.2f}")
    print(f"Put Option Price: ${put_price:.2f}")
    
    # Calculate Greeks
    call_delta = bs.call_delta(S, K, T, r, sigma)
    put_delta = bs.put_delta(S, K, T, r, sigma)
    gamma = bs.call_gamma(S, K, T, r, sigma)
    vega = bs.call_vega(S, K, T, r, sigma)
    call_theta = bs.call_theta(S, K, T, r, sigma)
    
    print("\nGreeks:")
    print(f"Call Delta: {call_delta:.4f}")
    print(f"Put Delta: {put_delta:.4f}")
    print(f"Gamma: {gamma:.4f}")
    print(f"Vega: {vega:.4f}")
    print(f"Call Theta: {call_theta:.4f}")
    
    # Plot option surface
    S_range = np.linspace(50, 150, 50)
    T_range = np.linspace(0.1, 2, 50)
    bs.plot_option_surface(S_range, T_range, K, r, sigma, 'call')

if __name__ == "__main__":
    main()
