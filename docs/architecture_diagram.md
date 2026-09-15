# DeepBreath Architecture Diagrams

These diagrams describe the research and application architecture of DeepBreath.
They are intended for portfolio documentation and do not imply clinical
deployment or diagnostic validation.

## Research Pipeline

This diagram shows the original respiratory audio machine learning workflow:
dataset ingestion, cycle-level preprocessing, Mel-spectrogram feature
generation, and CNN probability prediction.

```mermaid
flowchart TD
    A["ICBHI Respiratory Audio Dataset"] --> B["Audio Loading & Annotation Parsing"]
    B --> C["Respiratory Cycle Segmentation"]
    C --> D["Audio Augmentation"]
    D --> E["Mel Spectrogram Extraction"]
    E --> F["CNN Classification Model"]
    F --> G["Disease Probability Output"]

    subgraph ResearchLayer["Research / Training Layer"]
        A
        B
        C
        D
        E
        F
        G
    end
```

## Deployment Inference Service

This diagram shows the application-layer inference path exposed by FastAPI.
The service wraps the existing saved CNN model and preprocessing pipeline; it
does not alter the model architecture, weights, class mapping, or evaluation
results.

```mermaid
flowchart TD
    U["User"] -->|"WAV upload"| API["FastAPI API Layer"]
    API --> V["Input Validation"]
    V --> P["Inference Pipeline"]
    P --> M["Mel Spectrogram Generator"]
    M --> C["CNN Model"]
    C --> R["JSON Prediction Response"]

    subgraph DeploymentLayer["Deployment / API Layer"]
        API
        V
        P
        M
        C
        R
    end
```

## Current Layer Separation

DeepBreath separates the research pipeline from the deployable inference
service. This keeps the original MSc experiment auditable while allowing the
saved model to be served through a production-style API boundary.

```mermaid
flowchart LR
    subgraph Research["Research Artifacts"]
        N["Archived MSc Notebooks"]
        P["MSc Paper"]
        E["Legacy + Grouped Evaluation Reports"]
    end

    subgraph Package["Refactored Python Package"]
        D["data"]
        F["features"]
        M["models"]
        T["training"]
        I["inference"]
        R["reporting"]
    end

    subgraph Runtime["Application Runtime"]
        API["FastAPI Service"]
        Docker["Docker Image"]
        H5["Legacy H5 Model Artifact"]
    end

    Research --> Package
    Package --> Runtime
    H5 --> API
```

## Future LLM Reporting Extension

The LLM reporting layer is an optional extension for AI-assisted explanation.
It is separate from the `/predict` endpoint and should not be treated as
clinical functionality or diagnosis.

```mermaid
flowchart TD
    A["CNN Prediction"] --> C["LLM Explanation Layer"]
    B["Structured Patient-level Acoustic Features"] --> C
    C --> D["Patient-friendly Explanation"]
    C --> E["Clinician-style Explanation"]
    C --> F["Limitations + Research Disclaimer"]

    subgraph Extension["Optional / Modular Reporting Extension"]
        A
        B
        C
        D
        E
        F
    end
```

## Safety Boundary

- The FastAPI service returns model probabilities for research demonstration.
- The grouped evaluation is a more conservative generalisation estimate than
  the original sample-level MSc split.
- The system is not clinically validated and must not be used for diagnosis.
