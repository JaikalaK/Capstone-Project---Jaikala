# Capstone Project

This repository is organized into 4 folders, one per part.

## Folder Structure
- `part1/`
  - `README.md`
  - `part1_eda.py`
  - `requirements.txt`
  - `output/`
- `part2/`
  - `README.md`
  - `part2_ml.py`
  - `requirements.txt`
  - `output/`
- `part3/`
  - `README.md`
  - `part3_ml.py`
  - `requirements.txt`
  - `output/`
  - `best_model.pkl`
- `part4/`
  - `README.md`
  - `part4_llm_feature.py`
  - `requirements.txt`
  - `output/`
  - `.env.example`

## Run Order
1. Part 1
   - `cd part1`
   - `python3 part1_eda.py`
2. Part 2
   - `cd ../part2`
   - `python3 part2_ml.py`
3. Part 3
   - `cd ../part3`
   - `python3 part3_ml.py`
4. Part 4
   - `cd ../part4`
  - `python3 part4_llm_feature.py`
  - If API key/dependencies are missing, it auto-falls back to mock mode.

## Environment Variables (Part 4)
Part 4 reads API settings from environment variables. Do not hardcode secrets.

Required variable names:
- `LLM_API_KEY`
- `LLM_API_URL`
- `LLM_MODEL`

Use `.env.example` as template for local setup. Do not commit real `.env` values.
