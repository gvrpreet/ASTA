import numpy as np
import yfinance as yf
import talib
import torch
import warnings
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ---------------------------
# DataHandler Class
# ---------------------------
class DataHandler:
    def __init__(self):
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler()
    
    def fetch_data(self, symbol, start_date, end_date):
        print(f"Fetching data for {symbol} from {start_date} to {end_date}...")
        df = yf.download(symbol, start=start_date, end=end_date)
        if df.empty:
            raise ValueError(f"No data found for {symbol}")
        return self.prepare_data(df)
    
    def prepare_data(self, df):
        df = df.copy()
        # Preserve the original closing price for portfolio simulation
        df['CloseOrig'] = df['Close'].astype(float)
        
        # Convert columns explicitly to one-dimensional arrays (to avoid TA-Lib errors)
        close_prices = np.array(df['Close'], dtype=np.float64).flatten()
        volume = np.array(df['Volume'], dtype=np.float64).flatten()
        
        # Compute technical indicators
        df['Returns'] = df['Close'].pct_change()
        df['SMA_20'] = talib.SMA(close_prices, timeperiod=20)
        df['EMA_20'] = talib.EMA(close_prices, timeperiod=20)
        df['RSI'] = talib.RSI(close_prices, timeperiod=14)
        macd, signal, _ = talib.MACD(close_prices)
        df['MACD'] = macd
        df['MACD_signal'] = signal
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices)
        df['BB_upper'] = bb_upper
        df['BB_middle'] = bb_middle
        df['BB_lower'] = bb_lower
        df['OBV'] = talib.OBV(close_prices, volume)
        df['MOM'] = talib.MOM(close_prices, timeperiod=14)
        
        # List of features to normalize (do not scale CloseOrig)
        features = ['Close', 'Returns', 'SMA_20', 'EMA_20', 'RSI', 
                    'MACD', 'MACD_signal', 'BB_upper', 'BB_middle', 
                    'BB_lower', 'OBV', 'MOM']
        df[features] = df[features].fillna(method='ffill').fillna(method='bfill')
        df[features] = self.scaler.fit_transform(df[features])
        df = df.dropna()
        print(f"Prepared data shape for {df.index[-1]} rows and {len(df.columns)} columns.")
        return df
    
    def fetch_multiple_data(self, symbols, start_date, end_date):
        data_dict = {}
        for symbol in symbols:
            try:
                data = self.fetch_data(symbol, start_date, end_date)
                data_dict[symbol] = data
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
        return data_dict
