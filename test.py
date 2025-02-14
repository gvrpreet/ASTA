import torch
import plotly.graph_objects as go
import warnings
from src.trading_env import *
from src.dqn_agent import *
from src.data_handler import *
warnings.filterwarnings('ignore')

# Set the device for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -------------------------
# Test Simulation Function: Returns Portfolio History, Actions, and Signals
# -------------------------
def test_agent(agent, env):
    state = env.reset()
    done = False
    actions_taken = []
    signals = []  # record "buy", "sell", or "hold" for each step.
    while not done:
        action = agent.act(state)
        actions_taken.append(action)
        # Define buy signals for actions 3 & 4, sell signals for actions 0 & 1.
        if action in [3, 4]:
            signals.append("buy")
        elif action in [0, 1]:
            signals.append("sell")
        else:
            signals.append("hold")
        state, reward, done = env.step(action)
    return env.portfolio_value_history, actions_taken, signals

# -------------------------
# Plotting Functions
# -------------------------
def plot_portfolio(portfolio_values, signals, title):
    # Prepare markers for buy and sell signals along the portfolio evolution
    buy_indices = [i for i, s in enumerate(signals) if s == "buy"]
    sell_indices = [i for i, s in enumerate(signals) if s == "sell"]
    buy_values = [portfolio_values[i] for i in buy_indices]
    sell_values = [portfolio_values[i] for i in sell_indices]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(portfolio_values))),
        y=portfolio_values,
        mode='lines+markers',
        name='Portfolio Value'
    ))
    fig.add_trace(go.Scatter(
        x=buy_indices,
        y=buy_values,
        mode='markers',
        name='Buy Signal',
        marker=dict(color='green', size=10, symbol='triangle-up')
    ))
    fig.add_trace(go.Scatter(
        x=sell_indices,
        y=sell_values,
        mode='markers',
        name='Sell Signal',
        marker=dict(color='red', size=10, symbol='triangle-down')
    ))
    fig.update_layout(title=title,
                      xaxis_title='Trading Steps',
                      yaxis_title='Portfolio Value ($)')
    fig.show()

def plot_price_signals(test_data, signals, title):
    # Plot price (CloseOrig) with buy and sell markers.
    # Use the environment's reset data (i.e. a sequential index)
    prices = test_data['CloseOrig'].values
    steps = list(range(len(prices)))
    
    buy_indices = [i for i, s in enumerate(signals) if s == "buy"]
    sell_indices = [i for i, s in enumerate(signals) if s == "sell"]
    buy_prices = [prices[i] for i in buy_indices]
    sell_prices = [prices[i] for i in sell_indices]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps,
        y=prices,
        mode='lines',
        name='Price'
    ))
    fig.add_trace(go.Scatter(
        x=buy_indices,
        y=buy_prices,
        mode='markers',
        name='Buy Signal',
        marker=dict(color='green', size=10, symbol='triangle-up')
    ))
    fig.add_trace(go.Scatter(
        x=sell_indices,
        y=sell_prices,
        mode='markers',
        name='Sell Signal',
        marker=dict(color='red', size=10, symbol='triangle-down')
    ))
    fig.update_layout(title=title,
                      xaxis_title='Trading Steps',
                      yaxis_title='Price ($)')
    fig.show()

# -------------------------
# Main Testing Routine
# -------------------------
if __name__ == '__main__':
    try:
        # Dictionary of indexes to test: name -> ticker symbol
        indexes = {
            "USA_S&P500": "^GSPC",        # S&P 500 Index
            "India_Nifty50": "^NSEI",      # Nifty 50 Index
            "Japan_Nikkei225": "^N225",    # Nikkei 225 Index
            "UK_FTSE100": "^FTSE",         # FTSE 100 Index
            "France_CAC40": "^FCHI"        # CAC 40 Index
        }
        # Testing period of one year:
        TEST_START_DATE = '2024-01-01'
        TEST_END_DATE   = '2025-01-01'
        TEST_INITIAL_BALANCE = 10000
        
        # Load the pre-trained model.
        state_size = 12  # Must match state dimensions used in training.
        action_size = 5  # Five discrete actions.
        agent = DQNAgent(state_size, action_size)
        model_path = 'final_model.pth'
        checkpoint = torch.load(model_path, map_location=device)
        agent.load_state_dict(checkpoint)  # Adjust if checkpoint is nested.
        agent.eval()
        print("Pre-trained model loaded successfully.")
        
        # Initialize the DataHandler.
        data_handler = DataHandler()
        
        # Test the model on each index.
        for index_name, ticker in indexes.items():
            try:
                print("\n============================================")
                print(f"Testing on {index_name} ({ticker})")
                test_data = data_handler.fetch_data(ticker, TEST_START_DATE, TEST_END_DATE)
                # Initialize the TradingEnvironment using the prepared data.
                env = TradingEnvironment(test_data, initial_balance=TEST_INITIAL_BALANCE)
                portfolio_history, actions_taken, signals = test_agent(agent, env)
                final_value = portfolio_history[-1]
                return_pct = ((final_value / TEST_INITIAL_BALANCE) - 1) * 100
                print(f"Initial Balance: ${TEST_INITIAL_BALANCE:.2f}")
                print(f"Final Portfolio Value: ${final_value:.2f}")
                print(f"Cumulative Return: {return_pct:.2f}%")
                
                title_portfolio = f"{index_name} Portfolio Evolution (Return: {return_pct:.2f}%)"
                plot_portfolio(portfolio_history, signals, title_portfolio)
                
                title_price = f"{index_name} Price with Buy/Sell Signals"
                # Use the original test_data (not reset) for price plotting with the same sequential order.
                test_data_reset = test_data.reset_index(drop=True)
                plot_price_signals(test_data_reset, signals, title_price)
            except Exception as inner_ex:
                print(f"Error testing on {ticker}: {inner_ex}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        raise
