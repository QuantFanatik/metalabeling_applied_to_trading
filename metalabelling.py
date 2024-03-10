import ccxt
import numpy as np
import pandas as pd
import ta
import pandas_ta as pdta
from pandas import DataFrame
import finta
import math
import statistics
from binance.client import Client

import matplotlib.pyplot as plt
import seaborn as sns

import pandas_ta
import yfinance as yf
import tensorflow as tf
from scipy.optimize import minimize
from sklearn.svm import SVC
from sklearn.utils import class_weight
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.optimizers import Adam
from sklearn.calibration import CalibratedClassifierCV
from pykalman import KalmanFilter
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tensorflow.keras.losses import LogCosh
from keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np

global_timeframe = "15min"
ratio = "T"
symbol = "ustech100"
crypto =False

leverage = 1

path = "/Users/davidhuber/Desktop/Financial_data/Final/{}_{}.csv".format(symbol, global_timeframe)

data = pd.read_csv(path)

trading_length= 12

if crypto == False :
    data['Gmt time'] = data['Gmt time'].str.replace(".000","")
    data['Gmt time'] = pd.to_datetime(data['Gmt time'],format='%d.%m.%Y %H:%M:%S')
    data['timestamp'] = data['Gmt time']
    data['date'] = data['timestamp']
    data['close'] = pd.to_numeric(data['Close'])
    data['high'] = pd.to_numeric(data['High'])
    data['low'] = pd.to_numeric(data['Low'])
    data['open'] = pd.to_numeric(data['Open'])
    data['volume'] = pd.to_numeric(data['Volume'])
    data['Gmt time'] = pd.to_datetime(data['Gmt time'], unit='ms')
    data['hour'] = data['Gmt time'].dt.hour
    data['minute'] = data['Gmt time'].dt.minute
    data['hour+minute'] = data['hour'] + data['minute'] / 100
    data = data.drop(['Gmt time', 'Close', 'Low', 'Open', 'High'], axis=1)
    data = data.drop(['Gmt time.1'], axis=1)

elif crypto == True :
    data['date'] = pd.to_datetime(data['timestamp'])
    data['timestamp'] = pd.to_datetime(data['timestamp'])#, unit='ms')
    data['hour'] = data['timestamp'].dt.hour
    data['minute'] = data['timestamp'].dt.minute
    data['hour+minute'] = data['hour'] + data['minute'] / 100

def Slope(df, col_string, window):
    df_col = df[str(col_string)]
    slope = (df_col-df_col.shift(window))/df_col.shift(window)
    return slope

def Multiple_Slope(df, col_string, window, start_window):
    def Slope(df_2, col_string_2, window_1):
        df_col = df_2[str(col_string_2)]
        slope = (df_col - df_col.shift(window_1)) / df_col.shift(window_1)
        return slope

    while start_window < window:
        df[f'{col_string}_{window}_slope'] = Slope(df, col_string, start_window)
        start_window+=1
        return df[f'{col_string}_{window}_slope']


data['ema_20']= ta.wrapper.EMAIndicator(data['close'],20, fillna= True).ema_indicator()/data['close']
data['ema_50']= ta.wrapper.EMAIndicator(data['close'],50, fillna= True).ema_indicator()/data['close']
data['ema_100']= ta.wrapper.EMAIndicator(data['close'],100, fillna= True).ema_indicator()/data['close']
data['ema_200'] = ta.wrapper.EMAIndicator(data['close'],200, fillna= True).ema_indicator()/data['close']

data['VWAP_9'] = ta.wrapper.VolumeWeightedAveragePrice(data['high'], data['low'], data['close'], data['volume'], 9, fillna= True).volume_weighted_average_price()/data['close']
data['VWAP_12'] = ta.wrapper.VolumeWeightedAveragePrice(data['high'], data['low'], data['close'], data['volume'], 12, fillna= True).volume_weighted_average_price()/data['close']

data['rsi_k'] = ta.wrapper.StochRSIIndicator(data['close'], 9, fillna=True).stochrsi_k()
data['rsi_d'] = ta.wrapper.StochRSIIndicator(data['close'], 9, fillna=True).stochrsi_d()

data['atr_cl'] = ta.wrapper.AverageTrueRange(data['high'], data['low'],data['close'],14, fillna=True).average_true_range()/data['close']
data['atr'] = ta.wrapper.AverageTrueRange(data['high'], data['low'],data['close'],14, fillna=True).average_true_range()

data['std_50'] = data['close'].rolling(50).std()/data['close']
data['std_20'] = data['close'].rolling(20).std()/data['close']
data['std_5'] = data['close'].rolling(5).std()/data['close']

data['returns'] =(data['close'].shift(-5) - data['close'])/data['close']
data['returns1'] =(data['close'].shift(-4) - data['close'])/data['close']
data['returns2'] =(data['close'].shift(-3) - data['close'])/data['close']
data['returns3'] =(data['close'].shift(-2) - data['close'])/data['close']
data['returns4'] =(data['close'].shift(-1) - data['close'])/data['close']

data['min'] = data.low.rolling(20).min() / data.close
data['max'] = data.high.rolling(20).max() / data.close

data['min1'] = data.low.rolling(50).min() / data.close
data['max1'] = data.high.rolling(50).max() / data.close

data['min2'] = data.low.rolling(200).min() / data.close
data['max2'] = data.high.rolling(200).max() / data.close

data.merge(Multiple_Slope(data,"VWAP_9", 7,3))
data.merge(Multiple_Slope(data,"VWAP_12", 7,3))
data.merge(Multiple_Slope(data,"ema_20", 5,3))
data.merge(Multiple_Slope(data,"ema_50", 10,3))
data.merge(Multiple_Slope(data,"ema_100", 20,3))
data.merge(Multiple_Slope(data,"ema_200", 20,3))

data['stc'] = ta.wrapper.STCIndicator(data.close,50,23,10,3,3,True).stc()

data['adx'] = ta.wrapper.ADXIndicator(data.high, data.low, data.close, 14,True).adx()
data['adx_c'] = data.adx.shift(3)-data.adx / data.adx

data['adxp'] = ta.wrapper.ADXIndicator(data.high, data.low, data.close, 14,True).adx_pos()
data['adx_p_c'] = data.adxp.shift(3)-data.adxp / data.adxp

data['adxn'] = ta.wrapper.ADXIndicator(data.high, data.low, data.close, 14,True).adx_neg()
data['adx_n_c'] = data.adxn.shift(3)-data.adxn / data.adxn

data['macd'] = ta.wrapper.MACD(data['close'], 26, 12, 9, True).macd()
data['macd_diff'] = ta.wrapper.MACD(data['close'], 26, 12, 9, True).macd_diff()
data['macd_sign'] = ta.wrapper.MACD(data['close'], 26, 12, 9, True).macd_signal()




""" 
    signal processing using inequality constraints 
"""

#data['signal_rsi'] = np.where(data['rsi_k']<data['rsi_d'], np.where(data['rsi_k'].shift(1)>data['rsi_d'].shift(1), np.where(data['rsi_k']<0.5, 1, 0), 0), 0)
data['signal_ema'] = np.where(data['ema_20']>=data['ema_200'],np.where(data['ema_20'].shift(1)!=data['ema_200'].shift(1),1,0),0)

data['Signal'] = 1 * data['signal_ema'] * leverage

print(data)

"""
    separating in and out of sample
"""

half_df = round(len(data)*(3/4))
len_brute_df = round(len(data)*(3/4)*(1/5))

brute_force_df = data.iloc[:len_brute_df]
df=data.iloc[:half_df]
df_2 = data.iloc[half_df:]


"""        _________________________________________________________________________________________________________ 
backtesting :                                                                                                       |
|                                                                                                                   |
|   Let's backtest a little strategy, we will buy the asset when the slow period ema is bigger than the long period |
|   ema and we had a cross-over on the stochastic RSI.                                                              |
|                                                                                                                   |
|   To exit of the trade, we will use the Average True Range to set the stop loss and the take profit. also we will |
|   set a maximum trading period as an exit condition, to limit market exposure.                                    |
|                                                                                                                   |
|   To do this lets use and build a backtesting method:                                                             |
|___________________________________________________________________________________________________________________|
"""

class bt_engine():

    def __init__(self, df, atr_multiplier, risk_reward_ratio, fee, holding_period_threshold):
        self.df = df
        self.atr_multiplier = atr_multiplier
        self.risk_reward_ratio = risk_reward_ratio
        self.fee = fee
        self.holding_period_threshold = holding_period_threshold

    def net_pnl(self):
        in_a_trade = False
        dt = pd.DataFrame(columns=['date', 'net_pnl', 'price','signal'])
        net_pnl = 0
        target_profit_price = 0
        stop_loss_price = 0
        entry_price = 0
        holding_period = 0
        price = 0
        size = 0
        copy_df = self.df.copy()

        for index, row in copy_df.iterrows():
            if row['Signal'] > 0 and not in_a_trade:
                entry_price = row['close']
                date = row['date']
                price = row['close']
                size = row['Signal']
                in_a_trade = True
                target_profit_price = entry_price + self.atr_multiplier * row['atr'] * self.risk_reward_ratio
                stop_loss_price = entry_price - self.atr_multiplier * row['atr']
                net_pnl = 0
                holding_period = 0
                myrow = pd.DataFrame({'date': [date], 'net_pnl': [net_pnl], 'price': [price], 'signal': [size]})
                dt = pd.concat([dt, myrow], ignore_index=True)

            elif in_a_trade==True:
                if row['high'] >= target_profit_price and in_a_trade == True:
                    net_pnl = (size * (target_profit_price - entry_price) / entry_price) - 2 * self.fee
                    in_a_trade = False
                    date = row['date']
                    price = row['close']
                    signal = 0
                    myrow = pd.DataFrame({'date': [date], 'net_pnl': [net_pnl], 'price': [price], 'signal': [signal]})
                    dt = pd.concat([dt, myrow], ignore_index=True)

                elif row['low'] <= stop_loss_price and in_a_trade == True:
                    net_pnl = (size * (stop_loss_price - entry_price) / entry_price) - 2 * self.fee
                    in_a_trade = False
                    date = row['date']
                    price = row['close']
                    signal = 0
                    myrow = pd.DataFrame({'date': [date], 'net_pnl': [net_pnl], 'price': [price], 'signal': [signal]})
                    dt = pd.concat([dt, myrow], ignore_index=True)

                elif holding_period >= self.holding_period_threshold:
                    exit_price = row['close']
                    net_pnl = (size * (exit_price - entry_price) / entry_price) - 2 * self.fee
                    in_a_trade = False
                    date = row['date']
                    price = row['close']
                    signal = 0
                    myrow = pd.DataFrame({'date': [date], 'net_pnl': [net_pnl], 'price': [price], 'signal': [signal]})
                    dt = pd.concat([dt, myrow], ignore_index=True)


            holding_period += 1
        return dt

    def profit_factor(self):
        dt = bt_engine(self.df,self.atr_multiplier,self.risk_reward_ratio,self.fee,self.holding_period_threshold).net_pnl()
        profit_factor = sum([r for r in dt['net_pnl'] if r > 0]) / -sum([r for r in dt['net_pnl'] if r < 0])
        return profit_factor

    def plot_cumulative_returns(self):
        """
        Plots the cumulative returns of a trading strategy.

        Parameters:
        - trade_returns: A list of individual trade returns.
        """
        # Convert individual trade returns to growth factors
        trade_returns = bt_engine(self.df,self.atr_multiplier,self.risk_reward_ratio,self.fee,self.holding_period_threshold).net_pnl()
        growth_factors = [1 + r for r in trade_returns['net_pnl']]

        # Calculate cumulative returns
        cumulative_returns = np.cumprod(growth_factors)

        # Convert to pandas Series for easy plotting
        cumulative_returns_series = pd.Series(cumulative_returns)

        # Plotting
        plt.figure(figsize=(10, 6))  # Set figure size
        cumulative_returns_series.plot(title='Cumulative Returns of Trading Strategy')
        plt.xlabel('Trade Number')
        plt.ylabel('Cumulative Return')
        plt.grid(True)  # Add grid lines for better readability
        plt.show()

    def plot_cumulative_returns_with_price(self):
        """
        Plots the cumulative returns of a trading strategy alongside the normalized closing price.

        Parameters:
        - dt: The DataFrame containing trade dates and net PnL.
        - df: The DataFrame containing the closing prices and dates.
        """
        # Merge the original df with dt on 'date' to align trade PnL with the corresponding dates
        new_df = self.df.copy()
        dt = bt_engine(self.df,self.atr_multiplier,self.risk_reward_ratio,self.fee,self.holding_period_threshold).net_pnl()
        merged_df = new_df.merge(dt, on='date', how='left')

        # Replace NaN values in 'net_pnl' with 0 for days without trades
        merged_df['net_pnl'].fillna(0, inplace=True)

        # Calculate cumulative returns from 'net_pnl'
        merged_df['cumulative_returns'] = (1 + merged_df['net_pnl']).cumprod()

        # Normalize the closing price for comparison (assuming 'close' is in df)
        normalized_price = new_df['close'] / new_df['close'].iloc[0]

        # Setting the 'date' column as the DataFrame index for plotting
        merged_df.set_index('date', inplace=True)
        new_df.set_index('date', inplace=True)

        # Plotting
        plt.figure(figsize=(14, 7))
        plt.plot(merged_df.index, merged_df['cumulative_returns'], label='Cumulative Returns of Strategy')
        plt.plot(new_df.index, normalized_price, label='Normalized Closing Price', alpha=0.75)

        plt.title('Strategy Cumulative Returns vs. Normalized Closing Price')
        plt.xlabel('Date')
        plt.ylabel('Returns')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)  # Rotate date labels for better readability
        plt.tight_layout()  # Adjust layout to make room for the rotated date labels
        plt.show()
    """
    def nelder_mead(self):
        # Objective function for optimization
        def objective(params):
            atr_multiplier, risk_reward_ratio = params
            # Create a new instance with current params to calculate profit factor
            temp_engine = bt_engine(self.df, atr_multiplier, risk_reward_ratio, self.fee, self.holding_period_threshold)
            # Return the negative profit factor since minimize function seeks to minimize the objective
            return -temp_engine.profit_factor()

        # Initial guess for the parameters [atr_multiplier, risk_reward_ratio]
        initial_guess = [self.atr_multiplier, self.risk_reward_ratio]

        # Run the optimization using Nelder-Mead method
        result = minimize(objective, initial_guess, method='Nelder-Mead')

        # Extract the optimized parameters
        optimized_params = result.x
        optimized_profit_factor = -result.fun  # Convert back to positive as we minimized the negative profit factor

        # Update the instance variables to optimized values
        self.atr_multiplier, self.risk_reward_ratio = optimized_params

        # Return the optimization results
        return {
            'optimized_atr_multiplier': self.atr_multiplier,
            '\n optimized_risk_reward_ratio': self.risk_reward_ratio,
            '\n maximum_profit_factor': optimized_profit_factor
        }
"""
    def nelder_mead(self):
        def objective(params):
            atr_multiplier, risk_reward_ratio = params
            # Enforce parameter bounds directly within the objective function
            atr_multiplier = np.clip(atr_multiplier, 1, 100)
            risk_reward_ratio = np.clip(risk_reward_ratio, 0.1, 10)
            # Calculate and return the negative profit factor
            return -bt_engine(self.df,atr_multiplier,risk_reward_ratio,self.fee,self.holding_period_threshold).profit_factor()

        # Initial guess for the parameters
        initial_guess = [self.atr_multiplier, self.risk_reward_ratio]  # Example: Midpoints of your parameter ranges

        # Run the optimization using Nelder-Mead method
        result = minimize(objective, initial_guess, method='Nelder-Mead', options={'xatol': 0.00001, 'fatol': 0.00001})

        # Extract the optimized parameters
        optimized_atr_multiplier = np.clip(result.x[0], 1, 100)
        optimized_risk_reward_ratio = np.clip(result.x[1], 0.1, 10)
        optimized_profit_factor = -result.fun  # Convert back to positive as we minimized the negative profit factor

        # Return the optimization results
        return {
            'optimized_atr_multiplier': optimized_atr_multiplier,
            'optimized_risk_reward_ratio': optimized_risk_reward_ratio,
            'maximum_profit_factor': optimized_profit_factor
        }

    def brute_force(self, initial_atr, atr_multiplier_max, atr_multiplier_step_size, initial_Risk_Reward,
                    risk_reward_ratio_max, risk_reward_ratio_step_size, heat_map):
        t = pd.DataFrame(columns=['atr_mult', 'risk_reward', 'profit_factor'])

        def generate_heatmap(self, t):
            heatmap_data = t.pivot(index='atr_mult', columns='risk_reward', values='profit_factor')
            plt.figure(figsize=(10, 10))
            sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt=".2f")
            plt.title('Profit Factor Heatmap')
            plt.xlabel('Risk-Reward Ratio')
            plt.ylabel('ATR Multiplier')
            plt.show()

        while initial_atr < atr_multiplier_max:
            current_atr = initial_atr + atr_multiplier_step_size  # Use a different variable for the current iteration
            current_Risk_Reward = initial_Risk_Reward  # Reset this for each new atr_multiplier

            while current_Risk_Reward < risk_reward_ratio_max:
                current_Risk_Reward += risk_reward_ratio_step_size
                profit_factor = bt_engine(df=self.df, atr_multiplier=current_atr, risk_reward_ratio=current_Risk_Reward,
                                          fee=self.fee,
                                          holding_period_threshold=self.holding_period_threshold).profit_factor()
                myrow = pd.DataFrame(
                    {'atr_mult': [round(current_atr,5)], 'risk_reward': [round(current_Risk_Reward,5)], 'profit_factor': [profit_factor]})
                t = pd.concat([t, myrow], ignore_index=True)

            initial_atr += atr_multiplier_step_size  # This should likely be moved or adjusted to correct the logic

        max_profit_factor = max(t['profit_factor'])
        optimized_atr_multiplier = t.loc[t['profit_factor'] == max_profit_factor, 'atr_mult'].iloc[0]
        optimized_risk_reward_ratio = t.loc[t['profit_factor'] == max_profit_factor, 'risk_reward'].iloc[0]

        if heat_map:
            generate_heatmap(self,t)
        else:
            return {
                ' optimized_atr_multiplier': optimized_atr_multiplier,
                '\n optimized_risk_reward_ratio': optimized_risk_reward_ratio,
                '\n maximum_profit_factor': max_profit_factor
            }


brute = bt_engine(brute_force_df,atr_multiplier=5.5, risk_reward_ratio=6 , fee=0.05 / 100, holding_period_threshold=30)
brute.brute_force(initial_atr=2.0, atr_multiplier_max=30, atr_multiplier_step_size=2, initial_Risk_Reward=0.1, risk_reward_ratio_max=7, risk_reward_ratio_step_size=1, heat_map=True)

engine = bt_engine(df,atr_multiplier=1.5, risk_reward_ratio=9.1, fee=0.05 / 100, holding_period_threshold=30)
#engine.plot_cumulative_returns_with_price()



"""        _________________________________________________________________________________________________________ 
Meta-labelling :                                                                                                    |
|                                                                                                                   |
|   Here we will make a meta-labelling part, the goal will be the next one :                                        |
|   We will label our trade as 1 if the ouput is positive and 0 if the ouput is negative                            |
|                                                                                                                   |
|   The goal is to use machine learning in a way that it help us deciding if a signal is a good(1) or bad(0) one.   |
|                                                                                                                   |
|   LSTM neural network could be a good idea since they can catch non-linear relationship in the data.              | 
|                                                                                                                   |   
|   We need too be aware of some points :                                                                           |
|                                                                                                                   |      
|   -   The holding period need to be short so it will permit use having more observations to fit the model,        |
|       but not too short too, because we want our pnl to be impacted by real movement of the price and             |
|       not only by random noise.                                                                                   |
|   -   We will have to fit our model on a in sample data set, and then test it on a out of sample dataset          |   
|                                                                                                                   |   
|___________________________________________________________________________________________________________________|
"""

# labelling our trading strategy:

returns_dt = engine.net_pnl().copy()

returns_dt['labels'] = np.where(returns_dt['net_pnl'].shift(-1) > 0, 1, 0) # correct
dn = df.merge(returns_dt, on='date', how='left')

# Replace NaN values in 'net_pnl' with 0 for days without trades
dn.fillna(0, inplace=True)

dn_filtered = dn[dn['signal'] == 1]


# Prepare features, excluding the 'label' and possibly the 'signal' column
X = dn_filtered.drop(columns=['labels','signal', 'Signal','timestamp','date','net_pnl']).values
# Prepare labels, selecting only the 'label' column
y = dn_filtered['labels'].values


# Reshape X to be 3D [samples, time steps, features] for LSTM
X_l=X.reshape((X.shape[0], 1, X.shape[1]))

# Splitting the dataset into training and testing sets
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_l, y, test_size=0.05, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=60,random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

predictions = rf_classifier.predict(X_test)

# Calculate the accuracy and other performance metrics
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

print(f"Model Accuracy: {accuracy}")
print(report)

# Feature Scaling
scaler = StandardScaler()
X_train_lstm = scaler.fit_transform(X_train_l.reshape(-1, X_train_l.shape[-1])).reshape(X_train_l.shape)
X_test_lstm = scaler.transform(X_test_l.reshape(-1, X_test_l.shape[-1])).reshape(X_test_l.shape)

# Initialize EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor the validation loss
    patience=105,         # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restores model weights from the epoch with the best value of the monitored quantity
)

model = Sequential()
model.add(LSTM(70, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), return_sequences=True))
"""model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.5))"""
model.add(LSTM(70, return_sequences=False))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
# Compile the model (use your preferred loss function and optimizer)
model.compile(optimizer=Adam(learning_rate=0.00007), loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(X_train_lstm, y_train_l, epochs=5000, batch_size=100, validation_data=(X_test_lstm, y_test_l), verbose=1, callbacks=[early_stopping])

# Model evaluation
model.evaluate(X_test_lstm, y_test_l)


""" Out of sample testing """


returns_dt_2 = bt_engine(df_2, atr_multiplier=5.5, risk_reward_ratio=6, fee=0.05 / 100, holding_period_threshold=30).net_pnl()

dn_2 = returns_dt_2.merge(df_2, on='date', how='left')


# Replace NaN values in 'net_pnl' with 0 for days without trades
dn_2.fillna(0, inplace=True)



dn_2_filtered = dn_2

print(dn_2_filtered['net_pnl'])

# Prepare features, excluding the 'label' and possibly the 'signal' column
X_2 = dn_2_filtered.drop(columns=['signal', 'Signal','timestamp', 'date', 'net_pnl']).values

out_of_sample_predictions = rf_classifier.predict(X_2)

X_2=scaler.transform(X_2)
# Reshape X to be 3D [samples, time steps, features] for LSTM
X_2_lstm=X_2.reshape((X_2.shape[0], 1, X_2.shape[1]))


predictions = model.predict(X_2_lstm)

dn_2_filtered['RF_labels'] = out_of_sample_predictions

daily_returns = dn_2_filtered['net_pnl']#.shift(-1)  # Example: PnL already represents returns
# Calculate cumulative returns from daily returns


# Add LSTM predictions as a binary signal to the DataFrame
dn_2_filtered.loc[:, 'lstm_signal'] = np.where(predictions >= 0.5, 1, 0)
print(sum(dn_2_filtered['net_pnl']))

# Set 'date' as the DataFrame index
dn_2_filtered.set_index('date', inplace=True)

# Calculate cumulative returns for the original strategy


print(sum(dn_2_filtered['net_pnl']))

# Calculate PnL for LSTM-selected trades and cumulative returns for LSTM strategy
dn_2_filtered['LSTM_pnl'] = np.where(dn_2_filtered['lstm_signal'].shift(1) == 1, np.where( dn_2_filtered['signal'].shift(1) == 1,dn_2_filtered['net_pnl'],0), 0)


dn_2_filtered['RF_pnl'] = np.where(dn_2_filtered['RF_labels'].shift(1) == 1, np.where( dn_2_filtered['signal'].shift(1) == 1, dn_2_filtered['net_pnl'],0), 0)




print(df_2.close)

df_2 = df_2.merge(dn_2_filtered, on='date', how='left')

df_2.fillna(0, inplace = True)

print(df_2)

#df_2['price_returns'] = np.cumprod(((df_2['close_x'] - df_2['close_x'].shift(1)) / df_2['close_x'].shift(1)) + 1).copy()

init_ret_lstm = [1 + r for r in df_2['LSTM_pnl']]
#df_2['cum_ret_lstm'] = np.cumprod(init_ret_lstm).copy()

init_ret = [1 + r for r in df_2['net_pnl']]
#df_2['cum_ret_init'] = np.cumprod(init_ret).copy()

init_ret_RF = [1 + r for r in df_2['RF_pnl']]
#df_2['cum_ret_RF'] = np.cumprod(init_ret_RF).copy()
# Prepare new columns as a dictionary
new_columns = {
    'price_returns': np.cumprod(((df_2['close_x'] - df_2['close_x'].shift(1)) / df_2['close_x'].shift(1)) + 1),
    'cum_ret_lstm': np.cumprod(init_ret_lstm),
    'cum_ret_init': np.cumprod(init_ret),
    'cum_ret_RF': np.cumprod(init_ret_RF)
}

# Create a new DataFrame from the dictionary
new_columns_df = pd.DataFrame(new_columns)

# Concatenate the new columns to the original DataFrame
df_2 = pd.concat([df_2, new_columns_df], axis=1)

df_2.set_index('date', inplace=True)

# Plotting
fig, ax = plt.subplots(figsize=(14, 7))

# Original Strategy Cumulative Returns
color_original = 'tab:red'
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Returns', color=color_original)  # One label for both series
ax.plot(df_2.index, df_2['cum_ret_init'], label='Original Strategy', color=color_original)

# LSTM Modified Strategy Cumulative Returns
color_lstm = 'tab:blue'
ax.plot(df_2.index, df_2['cum_ret_lstm'], label='LSTM Modified Strategy', color=color_lstm, alpha=0.6)
color_rf ='tab:gray'
ax.plot(df_2.index, df_2['cum_ret_RF'], label='Random Forest strategy', color=color_rf, alpha=0.6)
color_price ='tab:green'
ax.plot(df_2.index, df_2['price_returns'], label='price', color=color_price, alpha=0.6)

# Legend
ax.legend(loc='upper left')  # Adjust the location of the legend if needed

# Title and layout
plt.title('Original vs. LSTM Modified Strategy Cumulative Returns')
fig.tight_layout()  # Adjust layout to make room for the legend and labels
plt.show()












