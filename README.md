# Yuji-s-ForEx-ML-Pipeline (Hybrid)

This pipeline is an automated, multi-timeframe trading model utilizing Deep Learning, structured Machine Learning, and NLP Sentiment Analysis

**My Intention for this Project (The "Why")**
There are three isolation biases that often plague standard financial analysis: technical isolation, macro isolation, and sentiment isolation. These happen because technical indicators, macroeconomic shifts, and global news sentiments are analyzed separately, rather than as a whole-of-system approach.

Hence, I wanted to automate a tedious process that usually requires extensive hours. This pipeline works by ingesting live market data via API keys, assess financial news, and synthesizes these disparate data streams into a single, high-probability trading signal for the target currency pair variables, simultaneously.

**Tools & Libraries**
I imported 8 critical libraries/tools to execute this pipeline. To make sense of how and why the libraries/tools function, let's divide the pipeline into four critical phases.
Firstly, it begins with _Data Acquisition and Ingestion_, where (1) **yfinance** is utilized to retrieve real-time market data, while (2) **concurrent.futures** manages asynchronous parallel processing to minimize latency during bulk API requests. Nextly, in the _Data Engineering and Sentiment Analysis Phase_, (3) **pandas** and (4) **numpy** perform complex matrix transformations and pipeline construction, while (5) **Hugging Face transformers (FinBERT)** are deployed to extract quantitative sentiment scores from live financial news streams. Thirdly, the _Optimization_ phase leverages (6) **scikit-learn** to conduct probabilistic hyperparameter tuning, ensuring that the models are statistically calibrated for the specific market environment. And finally, the _Predictive Modeling_ phase implements a hybrid ensemble approach: (7) PyTorch powers multi-layered LSTMs for sequential time-series forecasting, while (8) **XGBoost** utilizes Gradient Boosted Decision Trees to capture non-linear trends within structured tabular data.

**The Mathematical Frameworks/Architecture of the Pipeline**
Coming from a Biology background, I view these frameworks not just as formulas, but as the essential foundations needed to keep this system stable, functional, and reliable. Thus, I integrated these three specific concepts to solve real bottlenecks I encoutered during the development stage.

1. **To filter market noise**: Having traded in the ForEx market for quite some time, the data can be incredibly noisy. I chose the LSTM architecture specifically for its Forget Gate mechanism, represented by this equation:
2. $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
3. One realization I gathered was the model could easily get overwhelmed by short-term price spikes (stochastic noise). Hence, this neural network gata functions to decide which historical price data is actually important to keep in its memory and which data is irrelevant to the long-term cycle, before making a prediction.
4. **To balance search efficiency**: While I did not test every possible settings available, I chose RandomizedSearchCV to tune the XGBoost model efficiently. This basically relies on a basic statistical principle:
5. $$P(\text{success}) = 1 - (1 - 0.05)^{n}$$
6. By setting $n=60$, the pipeline can achieve at least a 95% probability of finding a near-optimal result within the top 5% of the total search space, saving huge amounts of time.
7. **To penalize outlier data**: For the evaluation of the model, I chose the Mean Squared Error (MSE) as my primary compass, represented by this equation:
8. $$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
9. What it basically does is that it heavily penalizes large mistakes because the errors are squared. In effect, this trains the algorith to be conservative and stable, prioritizing safe predictions over wild guesses.

**The Practicality of the System**
Beyond the models, I focused on building a "workflow" that is efficient and responsible, as much as possible. I wanted to move towards a system that highly acknowledges both time and data integrity. Thus, I added three features to this pipeline that at least achieves that goal.
One implementation is the **ThreadPoolExecutor**. This handles data ingestion in parallel by fetching the data out of API request for the multiple currency pairs. Thus, immensely reducing the total wait time.
Second implementation is by building a "memory". For this to work, I utilized **joblib** and **torch.save** to create an automated "checkpointing" system. This essentially allows the pipeline to save its "brain", and when the new data arrives, the model can sustainably fine-tune itself on the new information rather than starting all over again.
And third and final implementation is the addition of a "safety switch" (the 'Hard Gate'): This basically just works by performing a continuous Walk-Forward Backtest. As a result, if the model's recent win rate falls below a 55% threshold, the system automatically restricts all execution to "NO TRADE." 
