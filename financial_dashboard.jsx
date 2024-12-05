%%{init: {'theme': 'neutral'}}%%
graph TB
    subgraph Market_Overview[Market Overview]
        CurrentPrice[Current Price: $104.50]
        Volume[Volume: 1.3M]
        PredictedPrice[Predicted: $108.50]
    end

    subgraph Price_Trends[Price Analysis]
        Historical --> Current --> Predicted
        Historical[Historical Data<br/>$100.00 - $104.50]
        Current[Current Trend<br/>+4.5% Growth]
        Predicted[Forecast<br/>+3.8% Projected]
    end

    subgraph Market_Metrics[Key Metrics]
        Volatility[Volatility<br/>Medium]
        Momentum[Momentum<br/>Positive]
        Volume_Trend[Volume Trend<br/>Increasing]
    end

    Market_Overview --> Price_Trends
    Market_Overview --> Market_Metrics

style Market_Overview fill:#f5f5f5,stroke:#333,stroke-width:2px
style Price_Trends fill:#e6f3ff,stroke:#333,stroke-width:2px
style Market_Metrics fill:#f0f7f0,stroke:#333,stroke-width:2px