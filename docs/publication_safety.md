# Publication Safety

This repository distinguishes between three artifact types:

- Original research artifacts: MSc paper and raw notebooks retained privately for provenance. The MSc paper PDF is **not** published: its cover page lists student numbers, so it is git-ignored (`legacy-paper/` and `docs/ML_Assignment_G26_DeepBreath_v5_27032026.pdf`) and kept local-only.
- Cleaned public-facing artifacts: notebooks in `notebooks/archive/research/` with outputs cleared, local paths replaced, and API keys redacted.
- Refactored engineering code: importable Python modules under `src/deepbreath/`.

Before publishing publicly:

- Do not commit real `.env` files or provider API keys.
- Do not commit the full ICBHI audio dataset unless its license and storage strategy are explicitly handled.
- The MSc paper PDF was reviewed and confirmed to list student numbers on its cover page; it must stay excluded from any public commit.
- Keep the original model filename `melspectogramTrained.h5` only as a legacy artifact reference.
