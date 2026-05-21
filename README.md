# Meridian Financial Customer Intelligence Platform

## Problem Statement

Meridian Financial needs a production-minded Customer Intelligence Platform to run smarter outreach campaigns and resolve customer complaints at scale. The system requires deploying one classical Machine Learning (ML) service and one Large Language Model (LLM) / Retrieval-Augmented Generation (RAG) service, both integrated behind a single production API spine.

The platform is divided into two primary lanes:

1. **Machine Learning Service**: Predicts campaign conversion (whether a contacted customer will subscribe to a term-deposit product) using structured customer data.
2. **LLM/RAG Service**: Answers operational and intelligence questions over free-text complaint narratives using cited evidence and validation.

---

## Datasets & Data Pipeline

For detailed documentation on the datasets, schemas, and download sources used in this platform, see [Dataset Reference](/docs/dataset_reference.md).


## Architecture Diagram 

```mermaid
graph TD
    subgraph S1 ["Structured Data (ML Lane)"]
        A[UCI Bank Marketing Data] --> B[Data Pipeline & Validation]
        B --> C[Feature Engineering]
        C --> D[Model Training & MLflow]
        D --> E[Relative Promotion Gate]
        E --> F[ML API Endpoint]
    end

    subgraph S2 ["Unstructured Data (LLM / RAG Lane)"]
        G[CFPB Consumer Complaints] --> H[Chunking & Embedding]
        H --> I["Vector DB index: FAISS/Chroma"]
        I --> J[Retrieval & Grounding Logic]
        J --> K[RAG API Endpoint]
    end

    subgraph S3 ["Production Spine (Unified API)"]
        F --> L[FastAPI Gateway]
        K --> L
        L --> M["/customer-intel (Integrated Endpoint)"]
        L --> N["/metrics (Monitoring & Drift)"]
    end

    classDef ml fill:#009688,stroke:#333,stroke-width:2px,color:#fff;
    classDef rag fill:#2196F3,stroke:#333,stroke-width:2px,color:#fff;
    classDef spine fill:#9C27B0,stroke:#333,stroke-width:2px,color:#fff;

    class A,B,C,D,E,F ml;
    class G,H,I,J,K rag;
    class L,M,N spine;

    %% Remove background and border from subgraphs
    style S1 fill:none,stroke:none;
    style S2 fill:none,stroke:none;
    style S3 fill:none,stroke:none;
```