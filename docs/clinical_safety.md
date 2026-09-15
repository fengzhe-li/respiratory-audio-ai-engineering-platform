# Clinical Safety Notes

DeepBreath is a research and portfolio prototype. It is not a medical device and must not be used as a standalone diagnostic system.

The LLM reporting layer should:

- avoid inventing vital signs, imaging results, medications, or lab values;
- clearly state that outputs are AI-assisted;
- present uncertainty and differential diagnoses;
- encourage professional medical review for concerning symptoms.

The LLM layer is configured through provider-neutral OpenAI-compatible
environment variables. Multiple providers were explored in the original research
notebooks, so public documentation should avoid implying that one provider is
required for the project architecture.
