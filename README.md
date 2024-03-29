# Metalabeling Enhanced Trend Following Strategy

## Introduction
This project aims to enhance a trend-following strategy using metalabeling, specifically utilizing LSTM and Random Forest classifiers. Our primary focus will be on two markets: the Crypto market (Ethereum) and the US market (US Tech 100/NASDAQ 100), on the 15 minutes timeframe.

## About Metalabeling
Metalabeling is a sophisticated technique in machine learning for trading that involves applying a secondary model to enhance the decision-making process of a primary trading model. The primary purpose of metalabeling is to analyze the trading signals generated by the initial model and assess their potential profitability. By incorporating additional context and predictive analytics, metalabeling helps in distinguishing between signals that are likely to result in profitable outcomes and those that are not.

This extra layer of analysis contributes to a more nuanced and selective approach to executing trades. In this project, we use metalabeling to refine the signals from our trend-following strategy, employing LSTM and Random Forest classifiers to predict the success of each trade. This methodology not only aims to increase the overall profitability of the trading strategy but also seeks to mitigate risk by avoiding less promising trades.

## About Trend Following Signals
- Entry : We have opted for a straightforward trend-following strategy where the signal is based on the position of the 20-period exponential moving average (EMA) relative to the 200-period EMA. A long signal is generated when the 20-period EMA is above the 200-period EMA. Conversely, when the fast EMA is below the slow EMA, it generates a neutral signal, indicating no long position.

- Exit : The strategy's exit criteria are determined by the Average True Range (ATR), where we multiply the ATR by specific multipliers to set take profit and stop-loss levels. These multipliers are determined through a brute force method. An essential aspect of the strategy is the holding period; positions are held for a maximum number of days. This approach ensures we accumulate enough trades to provide a substantial dataset for training our metalabeling model, as more trades equate to more observations for the model.

## Metalabeling Models

We provide each model with a plethora of indicators and data to ensure they have comprehensive information to analyze. By supplying an extensive range of data and indicators, we enable the models to extract valuable insights from the data and price action of each underlying asset. Among the data provided are the returns of the last five trading candles, ATR (Average True Range), various EMAs (Exponential Moving Averages) with their slopes, and the minimum and maximum prices for the last 20/50/200 periods, among others.

##### LSTM (Long Short-Term Memory):
LSTM networks are a type of recurrent neural network (RNN) that are particularly well-suited for analyzing time series data. They excel at capturing long-term dependencies and non-linear behaviors in the data, which makes them an advanced and powerful tool for predicting the time-series data typically encountered in financial markets. Their ability to remember information for extended periods is what sets them apart from standard RNNs, making them highly effective for our metalabeling purposes where understanding past price movements is crucial.

##### Random Forest:
Random Forest is an ensemble learning method that operates by constructing a multitude of decision trees during the training phase and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Random forests correct for decision trees' habit of overfitting to their training set, providing a more generalized and robust prediction.


## Sample management

 - 3/4 for fiting the model, in this sample we will run the brute force routine on 1/5 of it.
 - 1/4 for the out-of-sample testing


## Brute force

This is an heatmap that represent the output of the brute force routine we used to optimize the parametters of the trading strategy.

<table>
  <tr>
    <td>
     <h4>US TECH 100, NASDAQ 100</h4>
      <img src="Heat_map_USTECH.png" width="100%" />
      <p></p>
      <p>We took an atr multiplier of : 1,5 </p>
      <p></p>
      <p>and an risk to reward ratio of : 9,1</p>
    </td>
    <td>
     <h4>Ethereum</h4>
      <img src="ETH_USDT_HEATMAP.png" width="100%" />
      <p></p>
      <p>We took an atr multiplier of : 5,5 </p>
      <p></p>
      <p>and an risk to reward ratio of : 6,1</p>
    </td>
  </tr>
</table>



## Results 

We can see the results for each metalabelling trading models. From the differents plot we can see that random forest performed better than the LSTM model, it can be becasue of many reasons: 
- Lack of hyperparemter optimisation
- Overfitting
- Underfitting
  

<table>
  <tr>
    <td>
     <h4>US TECH 100, NASDAQ 100</h4>
      <img src="Equity_curve_USTECH.png" width="100%" />
      <p></p>
    </td>
    <td>
     <h4>Ethereum</h4>
      <img src="Equity_curve_ETHUSDT.png" width="100%" />
    </td>
  </tr>
</table>


## Acknowledgements and Credits

### Data Sources

- Cryptocurrency data has been sourced from the Binance exchange platform, providing comprehensive market insights for various digital assets.
- Data pertaining to the US technology sector, specifically the US Tech 100 (NASDAQ 100), has been obtained from Dukascopy, which offers detailed financial information.

### Libraries

A heartfelt thank you to the developers and contributors of the various open-source libraries that have been instrumental in the realization of this project.




  






