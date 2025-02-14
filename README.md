# ASTA Project

## Overview

The ASTA project is a comprehensive trading simulation framework that leverages Deep Q-Learning (DQN) to train and test trading agents on historical stock data. The project is designed to fetch, prepare, and aggregate data from multiple stock indexes, simulate trading environments, and train DQN agents to optimize trading strategies.

## Project Structure

- **DataHandler Class**: Handles data fetching and preparation, including downloading historical stock data, computing technical indicators, and normalizing features.
- **TradingEnvironment Class**: Simulates a trading environment where the agent can interact by taking actions (buy, sell, hold) and receiving rewards based on portfolio performance.
- **DQNAgent Class**: Defines the neural network architecture and the reinforcement learning algorithm for training the agent.
- **Main Training Routine**: Aggregates data from multiple indexes, initializes the trading environment, and trains the DQN agent over multiple episodes.
- **Testing Phase**: Loads the pre-trained model, tests it on new data, and visualizes the results.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/ASTA.git
    cd ASTA
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Training

To train the DQN agent, run the main training script:
```sh
python train.py
```
This will fetch historical data, prepare it, and train the agent over multiple episodes.

### Testing

To test the trained model, run the testing script:
```sh
python test.py
```
This will load the pre-trained model, test it on new data, and generate visualizations of the portfolio performance and trading signals.

## Results

The results of the training and testing phases, including the final portfolio values and cumulative returns, will be displayed in the console and visualized using Plotly.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## Acknowledgements

- [Yahoo Finance](https://finance.yahoo.com/) for providing historical stock data.
- [TA-Lib](https://mrjbq7.github.io/ta-lib/) for technical analysis functions.
- [PyTorch](https://pytorch.org/) for the deep learning framework.
