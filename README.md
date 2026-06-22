# H-STDH: Forecasting Asymmetric Lead-Lag Anomalies (Implementation 1)

## 📌 Project Summary

Traditional quantitative models view the stock market as a flat list of prices, relying on linear math to find trading pairs. This research challenges that paradigm by building a bleeding-edge deep learning system that treats the S&P 500 as a dynamic, breathing network. Our goal is to uncover hidden **lead-lag anomalies**—where economic shocks in one sector systematically predict price movements in another days or hours later—using advanced representation learning. 

## 🎯 Motivation

Classical statistical arbitrage (like the Engle-Granger two-step method) assumes market relationships are linear and static. When market volatility shifts, these flat topological structures break down and strategies lose money. I am motivated to build a framework that actually understands the multi-scale, hierarchical nature of supply chains and institutional capital rotation, allowing the model to survive and adapt during regime shifts.

## 🧩 Novelty

* **Directed Hypergraphs:** Mapping complex, multi-industry supply chains where a single connection links whole clusters of supplier stocks to consumer stocks simultaneously (rather than simple pairs).
* **Dynamic Topology:** Allowing the neural network to completely "rewire" its understanding of the market network in real time as market volatility shifts.
* **Hyperbolic Space:** Embedding data into curved, non-Euclidean geometry (Poincam manifolds) to perfectly capture the naturally hierarchical tree structure of the financial markets without data distortion.

## 🧠 Methodology

* **Dataset:** Uses the [yfinance API](https://pypi.org/project/yfinance/) to stream continuous Hourly OHLCV (Open, High, Low, Close, Volume) data for a highly correlated mega-cap technology cluster. 
* **Architecture:** Implementation 1 utilizes a Baseline Graph Attention Network (GAT) built with PyTorch Geometric. It features a single-head spatial message passing layer connected to a linear prediction layer, optimized via MSE gradient descent.
* **Evaluation:** A custom **Walk-Forward Backtester**. It slides an 840-hour In-Sample (IS) training window and a 140-hour Out-of-Sample (OOS) testing window across the dataset. Crucially, a 5-hour buffer is placed betIen the train and test sets to eliminate overlapping feature leakage and look-ahead bias.
* **Metrics:** Annualized Out-of-Sample (OOS) Sharpe Ratio, derived from a simulated cross-sectional Long/Short portfolio execution.
* **Additional Methodology:** In-Sample Graph Topology Generation. To prevent structural data leakage, the Pearson correlation matrix used to draw the directed edges betIen stocks is calculated strictly within the isolated training window for each fold.

## 🌍 Impact

This research bridges a massive gap betIen classical quantitative finance and modern geometric deep learning. By proving that non-Euclidean graph architectures can capture hidden market inefficiencies that standard statistical models are blind to, I am laying the groundwork for a highly resilient, next-generation systematic trading framework.

## Future Work

With the data ingestion pipeline and baseline control successfully established, the remainder of the semester will follow our planned ablation study:
* **Implementation 2 (The Geometric Upgrade):** I will isolate the spatial variable by swapping the flat Euclidean network for a **Hyperbolic Manifold (Poincam ball)**. I will keep the standard pairwise edges to precisely measure the alpha generated purely by curving the embedding space to fit the market's hierarchy.
* **Implementation 3 (The Topological Upgrade):** The final synthesis. I will delete the pairwise edges and introduce **Directed Hyperedges**, allowing the Hyperbolic network to process multi-scale, industry-wide economic shocks simultaneously (H-STDH).

## Additional Sources
* **CHARMED:** [Predicting stock price movements using hyperbolic space representation learning with cross-attention and multimodal data fusion](https://www.researchgate.net/publication/405323632_Predicting_stock_price_movements_using_hyperbolic_space_representation_learning_with_cross-attention_and_multimodal_data_fusion) (Fukasawa et al.)
* **Hermes:** [A Multi-Scale Spatial-Temporal Hypergraph Network for Stock Time Series Forecasting](https://arxiv.org/abs/2509.23668) (Qiu et al.)
