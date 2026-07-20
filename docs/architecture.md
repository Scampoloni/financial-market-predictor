# Architecture

```mermaid
flowchart TD
    M[Market collector] --> MF[Market feature builder]
    N[RSS / optional NewsAPI collector] --> NF[FinBERT + VADER feature builder]
    M --> CG[Candlestick chart generator]
    CG --> CF[EfficientNet embedding + PCA]
    MF --> T[Temporal model training]
    NF --> T
    CF --> T
    T --> A[Model and result artefacts]
    A --> S[Streamlit app]
    N --> R[Headline retrieval index]
    R --> S
```

The predictive research pipeline and the news Q&A interface are separate. The Q&A feature retrieves bundled headlines and only uses an optional external generator when a local environment key is configured. Public demo mode should leave such keys unset.

Data paths and model locations are centralised in `src/config.py`. `src/models/splits.py` holds the date-aware validation splitter. Notebooks are development records and can differ from the revised production-style modules.
