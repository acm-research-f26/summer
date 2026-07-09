# H-STDH: Forecasting Asymmetric Lead-Lag Anomalies (Implementation 2)

## 📌 Project Summary

Traditional quantitative models view the stock market as a flat list of prices, relying on linear math to find trading pairs. This research challenges that paradigm by building a bleeding-edge deep learning system that treats the S&P 500 as a dynamic, breathing network. My goal is to uncover hidden **lead-lag anomalies**—where economic shocks in one sector systematically predict price movements in another days or hours later—using advanced representation learning. 

To mathematically isolate the results, this project is structured as a three-stage ablation study. **This branch represents Implementation 2: The Geometric Upgrade.** I have successfully isolated the spatial variable by upgrading the Euclidean baseline to a Hyperbolic Graph Attention Network (H-GAT), embedding the financial data into a curved manifold to better capture the market's hierarchical tree structure.

## 🎯 Motivation

Classical statistical arbitrage (like the Engle-Granger two-step method) assumes market relationships are linear and static. When market volatility shifts, these flat topological structures break down and strategies lose money. I am motivated to build a framework that actually understands the multi-scale, hierarchical nature of supply chains and institutional capital rotation, allowing the model to survive and adapt during regime shifts.

## 🧩 Novelty

* **Hyperbolic Space Embedding (Current Phase):** Embedding data into curved, non-Euclidean geometry (Poincaré manifolds) using the `geoopt` library to perfectly capture the naturally hierarchical tree structure of the financial markets without spatial crowding or data distortion.
* **Riemannian Optimization:** Utilizing Riemannian Adam to enforce constant negative curvature during backpropagation, preventing the neural network weights from collapsing back into flat space.
* **Directed Hypergraphs (Upcoming Phase):** Mapping complex, multi-industry supply chains where a single connection links whole clusters of supplier stocks to consumer stocks simultaneously.

## 🧠 Methodology

* **Dataset:** Uses the [yfinance API](https://pypi.org/project/yfinance/) to stream continuous Hourly OHLCV (Open, High, Low, Close, Volume) data for a highly correlated mega-cap technology cluster. 
* **Architecture:** Implementation 2 utilizes a **Hyperbolic Graph Attention Network (H-GAT)**. Input features are projected onto a Poincaré ball manifold via exponential maps. Hyperbolic distances are used to calculate spatial attention weights, and the network is strictly optimized via `geoopt.optim.RiemannianAdam`.
* **Evaluation:** A custom **Purged Walk-Forward Backtester**. It slides an 840-hour In-Sample (IS) training window and a 140-hour Out-of-Sample (OOS) testing window across the dataset. Crucially, a 5-hour buffer is placed between the train and test sets to eliminate overlapping feature leakage and look-ahead bias.
* **Metrics:** Annualized Out-of-Sample (OOS) Sharpe Ratio ($\sqrt{1764}$), derived from a simulated cross-sectional Long/Short portfolio execution.
* **Additional Methodology:** In-Sample Graph Topology Generation. To prevent structural data leakage, the Pearson correlation matrix used to draw the directed pairwise edges between stocks is calculated strictly within the isolated training window for each fold.

## 🌍 Impact

This research bridges a massive gap between classical quantitative finance and modern geometric deep learning. By proving that non-Euclidean graph architectures can capture hidden market inefficiencies that standard statistical models are blind to, I am laying the groundwork for a highly resilient, next-generation systematic trading framework.

## Future Work

With the Hyperbolic Geometry successfully stabilized in this implementation, the final sprint will execute the Topological Upgrade:
* **Implementation 3 (H-STDH Synthesis):** I will delete the standard pairwise edges and introduce **Directed Hyperedges**. This will allow the established hyperbolic network to process multi-scale, industry-wide economic shocks simultaneously, completing the final Hyperbolic Spatiotemporal Directed Hypergraph architecture.

## Additional Sources
* **CHARMED:** [Predicting stock price movements using hyperbolic space representation learning with cross-attention and multimodal data fusion](https://www.researchgate.net/publication/405323632_Predicting_stock_price_movements_using_hyperbolic_space_representation_learning_with_cross-attention_and_multimodal_data_fusion) (Fukasawa et al.)
* **Hermes:** [A Multi-Scale Spatial-Temporal Hypergraph Network for Stock Time Series Forecasting](https://arxiv.org/abs/2509.23668) (Qiu et al.)
