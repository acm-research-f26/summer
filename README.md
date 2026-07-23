# H-STDH: Forecasting Asymmetric Lead-Lag Anomalies (Implementation 3 - Final Synthesis)

## 📌 Project Summary

Traditional quantitative models treat the stock market as a flat list of prices, which fails to capture the complex, interconnected nature of modern supply chains. Our overarching research builds a deep learning system that treats the S&P 500 as a dynamic network to uncover asymmetric **lead-lag anomalies**. 

**This branch represents Implementation 3: The Topological Upgrade.** This is the final synthesis of our three-stage ablation study. We have successfully combined the curved geometry of the Poincaré manifold (Implementation 2) with a custom **Incidence Matrix**, creating a complete Hyperbolic Spatiotemporal Directed Hypergraph (H-STDH) to stabilize out-of-sample quantitative trading performance during extreme market shocks.

## 🎯 Motivation

In Implementation 2, we successfully upgraded the spatial geometry to a Hyperbolic manifold. However, Out-of-Sample testing revealed a mathematical vulnerability: during extreme macroeconomic volatility (e.g., Q1 2024 AI sector decoupling), standard pairwise edges pushed stock embeddings too close to the boundary of the Poincaré disk, causing distance metrics to explode and Sharpe ratios to degrade. I am motivated to solve this geometric instability by replacing 1-to-1 pairwise edges with overlapping sets (hyperedges), distributing stock-specific volatility across entire sector clusters to create a resilient, market-neutral forecasting engine.

## 🧩 Novelty

* **Incidence Matrix Topology:** Abandoning simple pairwise directed edges in favor of an incidence matrix, mapping overlapping hyperedges that group highly correlated stocks into localized "centers of mass."
* **Tangent Space Aggregation:** Projecting hyperbolic features back to the Euclidean tangent space via logarithmic maps (`logmap0`) before performing matrix multiplication, preventing exploding gradients during high-volatility regime shifts.
* **GPU-Accelerated Purged Backtesting:** Engineered a low-latency walk-forward backtester that pre-loads temporal tensors directly to the GPU hardware, slashing CPU-to-GPU I/O latency while strictly preventing look-ahead bias via a 5-hour purge buffer.

## 🧠 Methodology

* **Dataset:** Uses the [yfinance API](https://pypi.org/project/yfinance/) to stream continuous Hourly OHLCV (Open, High, Low, Close, Volume) data for a highly correlated mega-cap technology cluster. 
* **Architecture:** The final **Hyperbolic Hypergraph Network**. Input features are projected onto a Poincaré ball, pulled into the tangent space, multiplied against the dynamic incidence matrix to aggregate cluster-wide data, and pushed back to individual nodes for return prediction. Optimized via Riemannian Adam.
* **Evaluation:** A custom Purged Walk-Forward Engine. It slides an 840-hour In-Sample (IS) training window and a 140-hour Out-of-Sample (OOS) testing window across the dataset. 
* **Metrics:** Annualized Out-of-Sample (OOS) Sharpe Ratio, derived from a simulated cross-sectional Long/Short portfolio execution. 
* **Additional Methodology:** Dynamic Hyperedge Generation. To prevent structural data leakage, the rolling correlation matrix used to generate the incidence matrix is calculated strictly within the isolated training window for each fold.

## 🌍 Impact

This architecture bridges theoretical mathematical statistics with high-performance computer science. By proving that a non-Euclidean hypergraph can absorb and distribute macroeconomic shocks better than standard flat networks, this research provides a mathematically rigorous blueprint for next-generation quantitative trading and statistical arbitrage frameworks.

## Future Work

With the foundational deep learning model mathematically stabilized and outperforming the baseline control group, future developments will focus on the execution layer:
* **High-Frequency Adaptation:** Porting the mathematical inference layers into a lower-level, compiled environment (e.g., C++) to reduce execution latency.
* **Tick-Level Data Ingestion:** Scaling the temporal inputs from hourly OHLCV bars to full Order Book Imbalance (OBI) limit order book data.

## Additional Sources
* **CHARMED:** [Predicting stock price movements using hyperbolic space representation learning with cross-attention and multimodal data fusion](https://www.researchgate.net/publication/405323632_Predicting_stock_price_movements_using_hyperbolic_space_representation_learning_with_cross-attention_and_multimodal_data_fusion) (Fukasawa et al.)
* **Hermes:** [A Multi-Scale Spatial-Temporal Hypergraph Network for Stock Time Series Forecasting](https://arxiv.org/abs/2509.23668) (Qiu et al.)
