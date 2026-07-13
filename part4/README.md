# Part 4 - LLM-Powered Prediction Explanation

## Track chosen: C - Model Prediction Explanation Pipeline

## What this does
Loads the saved model from part 3, runs predictions on 3 hand-crafted inputs, and asks an LLM to explain each prediction as structured JSON.

## How to run
```bash
# with a real API key
export LLM_API_KEY="your_key_here"
python part4_llm_feature.py

# without an API key (uses a local mock response)
python part4_llm_feature.py --mock-llm
```

If `requests` is not installed or `LLM_API_KEY` is missing, the script now auto-falls back to mock mode and still produces output files.

## API key setup
The API key must be set as an environment variable. Never hardcode it in the script.

Use this exact pattern in code:
`os.environ['LLM_API_KEY']`

Create a `.env` file (excluded from git via `.gitignore`):
```
LLM_API_KEY=your_key_here
LLM_API_URL=https://openrouter.ai/api/v1/chat/completions
LLM_MODEL=openai/gpt-4o-mini
```

You can start from `part4/.env.example` and copy it to `.env`.

Then export them before running:
```bash
export LLM_API_KEY=your_key_here
```

## call_llm function
```python
def call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=512):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens":  max_tokens
    }
    headers = {
        "Authorization": "Bearer " + api_key,
        "Content-Type":  "application/json"
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        print("LLM call failed, status:", resp.status_code)
        return None
    return resp.json()["choices"][0]["message"]["content"]
```

## System prompt (exact text)
```
You are a model-prediction explanation assistant.
Return ONLY valid JSON matching exactly this schema:
{
  "prediction_label": "string",
  "confidence_level": "low|medium|high",
  "top_reason": "string",
  "second_reason": "string",
  "next_step": "string",
  "requires_human_review": true|false
}
Rules:
- Do not include markdown, code fences, or extra keys.
- Keep reasons concise and grounded in provided features and probability.
- If probability >= 0.9 use confidence_level="high".
- If probability between 0.7 and 0.9 use confidence_level="medium".
- Else use confidence_level="low".
```

## User prompt template
```
Feature values:
{feature_json}

Predicted class: {pred_class}
Predicted probability for class 1: {prob}

Explain this prediction as JSON only.
```

## Why temperature=0
I used temperature=0 for the main runs because I want consistent, predictable JSON output. At temperature=0 the model always picks the highest probability token, so the output is basically deterministic. At higher temperatures like 0.7 it samples from a wider distribution, so you get different wording or slightly different field values each run. For structured tasks like JSON extraction you want the model to be boring and consistent.

## JSON validation
After each LLM call:
1. Strip whitespace from the response
2. Parse with `json.loads()` - catch `JSONDecodeError` if it fails
3. Validate against the schema with `jsonschema.validate()` - catch `ValidationError`
4. If either step fails, return the fallback dict with all fields set to None

Note: if `jsonschema` is not installed, the script uses an internal fallback validator that checks required keys and basic field types.

The schema requires 6 fields: `prediction_label`, `confidence_level`, `top_reason`, `second_reason`, `next_step`, `requires_human_review`.

## PII guardrail
Before every LLM call I check the input for emails and phone numbers using regex:
```python
def has_pii(text):
    email = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    phone = r"\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"
    return bool(re.search(email, text) or re.search(phone, text))
```
If it returns True, the call is blocked and I print "Input blocked: PII detected."

Test results:
- `"My email is alice@example.com. Explain this."` → **blocked**
- `"Explain this prediction in JSON only."` → **passed**

## Results (3 inputs)

| Input | Predicted class | Prob | LLM output | Valid JSON | Guardrail |
|-------|-----------------|------|------------|------------|-----------|
| Input 1: carat=0.30, Ideal cut | 0 | 0.0008 | prediction_label=below_median_price, confidence=low | pass | passed |
| Input 2: carat=1.20, Good cut  | 1 | 1.0000 | prediction_label=above_median_price, confidence=high | pass | passed |
| Input 3: carat=1.80, Fair cut  | 1 | 1.0000 | prediction_label=above_median_price, confidence=high | pass | passed |

## Temperature comparison (0.0 vs 0.7)

| Input | Temp=0 next_step | Temp=0.7 next_step | Difference |
|-------|------------------|--------------------|------------|
| Input 1 | Use standard pricing workflow. | Route to standard catalog with QA check. | wording varies |
| Input 2 | Use standard pricing workflow. | Escalate to premium pricing analyst. | wording varies |
| Input 3 | Use standard pricing workflow. | Escalate to premium pricing analyst. | wording varies |

At temp=0 the `next_step` field is identical every time. At 0.7 it varies because the model is sampling from a wider token distribution. Both pass schema validation since the field just needs to be a string.
