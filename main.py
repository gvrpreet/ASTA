import numpy as np
import pandas as pd
import yfinance as yf
import talib
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from tqdm import tqdm
import warnings
from src.trading_env import *
from src.dqn_agent import *
from src.data_handler import *
warnings.filterwarnings('ignore')
# Set device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {torch.cuda.get_device_name(device)}")

# ---------------------------
# Aggregate Data Function
# ---------------------------
def aggregate_data(data_dict):
    """
    Given a dictionary of DataFrames (keyed by symbol), find the common date
    index and average the features.
    """
    common_index = None
    for df in data_dict.values():
        if common_index is None:
            common_index = df.index
        else:
            common_index = common_index.intersection(df.index)
    reindexed = []
    for symbol, df in data_dict.items():
        reindexed.append(df.loc[common_index])
    concatenated = pd.concat(reindexed, axis=1, keys=data_dict.keys())
    # Average features across symbols; this applies to every column including CloseOrig.
    aggregated = concatenated.groupby(axis=1, level=1).mean()
    return aggregated

# ---------------------------
# Training Function
# ---------------------------
def train_agent(agent, training_datasets, episodes):
    training_losses = []
    best_reward = float('-inf')
    
    # Training proceeds across episodes.
    # In each episode we randomly choose one aggregated dataset (i.e. one country) 
    # and simulate a complete trading episode over its available historical period.
    for episode in tqdm(range(episodes), desc="Training Episodes"):
        # Randomly choose one country's aggregated data
        country, data = random.choice(list(training_datasets.items()))
        env = TradingEnvironment(data, initial_balance=100000)
        state = env.reset()
        total_reward = 0
        episode_losses = []
        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            loss = agent.replay()
            if loss:
                episode_losses.append(loss)
            state = next_state
            total_reward += reward
            if done:
                avg_loss = np.mean(episode_losses) if episode_losses else 0
                training_losses.append(avg_loss)
                if total_reward > best_reward:
                    best_reward = total_reward
                    torch.save(agent.state_dict(), 'final_model.pth')
                print(f"\nEpisode {episode+1}/{episodes} | Country: {country} | Total Reward: {total_reward:.2f} | Avg Loss: {avg_loss:.4f} | Epsilon: {agent.epsilon:.4f} | Final Portfolio: ${env.portfolio_value_history[-1]:.2f}")
                break
    return training_losses

# ---------------------------
# Main Routine for Training on Multiple Indexes
# ---------------------------
if __name__ == '__main__':
    try:
        # Define training period with a larger dataset
        TRAIN_START_DATE = '1995-01-01'
        TRAIN_END_DATE   = '2024-01-01'
        EPISODES = 100  # adjust number of episodes as needed
        
        # Dictionary of indexes and their top 10 companies (tickers)
        index_companies = {
            "USA": ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'UNH', 'XOM'],
            "India": ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'LT.NS', 'AXISBANK.NS', 'ITC.NS', 'HINDUNILVR.NS'],
            "Japan": ['7203.T', '6758.T', '9984.T', '8306.T', '6902.T', '9432.T', '7267.T', '7974.T', '6501.T', '8801.T'],
            "UK": ['HSBA.L', 'BP.L', 'VOD.L', 'GSK.L', 'RIO.L', 'BT-A.L', 'ULVR.L', 'DGE.L', 'AZN.L', 'BATS.L'],
            "France": ['OR.PA', 'MC.PA', 'SAN.PA', 'AI.PA', 'BNP.PA', 'DG.PA', 'EN.PA', 'RI.PA', 'KER.PA', 'SU.PA']
        }
        
        # ... rest of your training code remains unchanged ...
        
        data_handler = DataHandler()
        training_datasets = {}
        
        # For each country/index, fetch data for its top companies and aggregate.
        for country, tickers in index_companies.items():
            print(f"\nProcessing {country} data:")
            company_data = data_handler.fetch_multiple_data(tickers, TRAIN_START_DATE, TRAIN_END_DATE)
            if company_data:
                agg_data = aggregate_data(company_data)
                training_datasets[country] = agg_data
            else:
                print(f"No valid data fetched for {country}.")
        
        if not training_datasets:
            raise ValueError("No aggregated training data available from any index!")
        
        # All training datasets must have the expected technical indicator columns.
        # We assume each aggregated DataFrame contains at least the following columns:
        # ['Close', 'Returns', 'SMA_20', 'EMA_20', 'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_middle', 'BB_lower', 'OBV', 'MOM', 'CloseOrig']
        
        # Initialize the training environment parameters are set within the episode loop.
        state_size = 12  # corresponds to the feature vector (excluding CloseOrig)
        action_size = 5  # five possible actions
        
        # Initialize the DQN agent
        agent = DQNAgent(state_size, action_size)
        
        print("\nStarting training on multiple indexes...")
        training_losses = train_agent(agent, training_datasets, EPISODES)
        
        print("\nTraining Completed. Best model saved as final_model.pth")
    
    except Exception as e:
        print(f"Error during training: {e}")
        raise
