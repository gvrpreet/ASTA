import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')
# ---------------------------
# Trading Environment Class
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class TradingEnvironment:
    def __init__(self, data, initial_balance=100000):
        if data is None or data.empty:
            raise ValueError("Data cannot be empty")
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        # Features used as state (these are the normalized technical indicators)
        self.features = ['Close', 'Returns', 'SMA_20', 'EMA_20', 'RSI',
                         'MACD', 'MACD_signal', 'BB_upper', 'BB_middle',
                         'BB_lower', 'OBV', 'MOM']
        # "CloseOrig" is used for actual portfolio value calculation.
        if 'CloseOrig' not in self.data.columns:
            raise ValueError("Missing 'CloseOrig' column in data.")
        self.reset()
    
    def reset(self):
        self.balance = self.initial_balance
        self.position = 0.0  # shares held
        self.current_step = 0
        self.portfolio_value_history = [self.initial_balance]
        return self._get_state()
    
    def _get_state(self):
        return self.data.iloc[self.current_step][self.features].values
    
    def step(self, action):
        # Retrieve actual price from the unscaled "CloseOrig"
        current_price = float(self.data.iloc[self.current_step]['CloseOrig'])
        prev_value = self.portfolio_value_history[-1]
        # Action space (5 discrete actions):
        # 0: Sell all, 1: Sell half, 2: Hold, 3: Buy with 50% cash, 4: Buy with full cash.
        if current_price > 0:
            if action == 0 and self.position > 0:
                # Sell all
                self.balance += current_price * self.position
                self.position = 0.0
            elif action == 1 and self.position > 0:
                # Sell half
                shares_to_sell = self.position * 0.5
                self.balance += current_price * shares_to_sell
                self.position -= shares_to_sell
            elif action == 3 and self.balance > 0:
                # Buy with 50% cash
                cash_to_use = self.balance * 0.5
                shares_to_buy = cash_to_use / current_price
                self.position += shares_to_buy
                self.balance -= cash_to_use
            elif action == 4 and self.balance > 0:
                # Buy with full cash
                shares_to_buy = self.balance / current_price
                self.position += shares_to_buy
                self.balance = 0
            # Action 2 (Hold) does nothing.
        new_value = self.balance + (self.position * current_price)
        self.portfolio_value_history.append(new_value)
        reward = (new_value - prev_value) / prev_value if prev_value > 0 else 0
        self.current_step += 1
        done = (self.current_step >= len(self.data) - 1)
        next_state = self._get_state() if not done else np.zeros(len(self.features))
        return next_state, reward, done
