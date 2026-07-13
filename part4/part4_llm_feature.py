# Part 4 - LLM-powered prediction explanation (Track C)
# Loads best_model.pkl from part 3 and explains predictions using an LLM API
#
# To run with a real API key:
#   export LLM_API_KEY="your_key_here"
#   python part4_llm_feature.py
#
# To test without an API key:
#   python part4_llm_feature.py --mock-llm

import os
import re
import json
import argparse
import joblib
import pandas as pd
try:
    import requests
except ImportError:
    requests = None

try:
    from jsonschema import validate, ValidationError
    HAS_JSONSCHEMA = True
except ImportError:
    validate = None
    ValidationError = Exception
    HAS_JSONSCHEMA = False

os.chdir(os.path.dirname(__file__) or ".")

OUT = "output"
os.makedirs(OUT, exist_ok=True)

PART3_MODEL = "../part3/best_model.pkl"

parser = argparse.ArgumentParser()
parser.add_argument("--mock-llm", action="store_true")
ARGS = parser.parse_args()

USE_MOCK = ARGS.mock_llm
if not USE_MOCK and requests is None:
    print("'requests' package not installed. Falling back to --mock-llm mode.")
    USE_MOCK = True
if not USE_MOCK and ("LLM_API_KEY" not in os.environ or not os.environ["LLM_API_KEY"].strip()):
    print("LLM_API_KEY not set. Falling back to --mock-llm mode.")
    USE_MOCK = True

# the 6 fields we expect back from the LLM
SCHEMA = {
    "type": "object",
    "properties": {
        "prediction_label":   {"type": "string"},
        "confidence_level":   {"type": "string", "enum": ["low", "medium", "high"]},
        "top_reason":         {"type": "string"},
        "second_reason":      {"type": "string"},
        "next_step":          {"type": "string"},
        "requires_human_review": {"type": "boolean"}
    },
    "required": ["prediction_label", "confidence_level", "top_reason",
                 "second_reason", "next_step", "requires_human_review"],
    "additionalProperties": False
}

FALLBACK = {
    "prediction_label":      None,
    "confidence_level":      None,
    "top_reason":            None,
    "second_reason":         None,
    "next_step":             None,
    "requires_human_review": None
}

SYSTEM_PROMPT = """You are a model-prediction explanation assistant.
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
- Else use confidence_level="low"."""


def has_pii(text):
    email = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    phone = r"\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"
    return bool(re.search(email, text) or re.search(phone, text))


def call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=512):
    if USE_MOCK:
        return mock_response(user_prompt, temperature)
    api_key = os.environ["LLM_API_KEY"]

    url   = os.environ.get("LLM_API_URL", "https://openrouter.ai/api/v1/chat/completions")
    model = os.environ.get("LLM_MODEL",   "openai/gpt-4o-mini")

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


def mock_response(user_prompt, temperature):
    # simple local mock so the script works without an API key
    m_prob  = re.search(r"Predicted probability.*?:\s*([0-9.]+)", user_prompt)
    m_class = re.search(r"Predicted class:\s*([01])", user_prompt)
    prob  = float(m_prob.group(1))  if m_prob  else 0.5
    pred  = int(m_class.group(1))   if m_class else int(prob >= 0.5)

    if prob >= 0.9:
        conf = "high"
    elif prob >= 0.7:
        conf = "medium"
    else:
        conf = "low"

    next_step = "Use standard pricing workflow."
    if temperature >= 0.7 and pred == 1:
        next_step = "Escalate to premium pricing analyst."
    elif temperature >= 0.7:
        next_step = "Route to standard catalog with QA check."

    return json.dumps({
        "prediction_label":      "above_median_price" if pred == 1 else "below_median_price",
        "confidence_level":      conf,
        "top_reason":            "Stone dimensions are the main driver of price.",
        "second_reason":         "Quality grades (cut, clarity) add secondary signal.",
        "next_step":             next_step,
        "requires_human_review": pred == 1 and prob < 0.8
    })


def safe_call(system_prompt, user_prompt, temperature):
    if has_pii(user_prompt):
        print("Input blocked: PII detected.")
        return None, "blocked"
    raw = call_llm(system_prompt, user_prompt, temperature=temperature)
    return raw, "passed" if raw is not None else "failed"


def parse_and_validate(raw):
    if raw is None:
        return FALLBACK.copy(), False, "no_response"
    try:
        parsed = json.loads(raw.strip())
    except json.JSONDecodeError as e:
        return FALLBACK.copy(), False, "json_error: " + str(e)

    if HAS_JSONSCHEMA:
        try:
            validate(instance=parsed, schema=SCHEMA)
            return parsed, True, None
        except ValidationError as e:
            return FALLBACK.copy(), False, "schema_error: " + str(e)

    # Fallback validation when jsonschema is not installed.
    required = {
        "prediction_label",
        "confidence_level",
        "top_reason",
        "second_reason",
        "next_step",
        "requires_human_review",
    }
    if not isinstance(parsed, dict):
        return FALLBACK.copy(), False, "schema_error: output is not an object"
    if set(parsed.keys()) != required:
        return FALLBACK.copy(), False, "schema_error: keys mismatch"
    if parsed["confidence_level"] not in {"low", "medium", "high"}:
        return FALLBACK.copy(), False, "schema_error: confidence_level invalid"
    if not isinstance(parsed["requires_human_review"], bool):
        return FALLBACK.copy(), False, "schema_error: requires_human_review must be boolean"
    for key in ["prediction_label", "top_reason", "second_reason", "next_step"]:
        if not isinstance(parsed[key], str):
            return FALLBACK.copy(), False, f"schema_error: {key} must be string"

    return parsed, True, None


# ---------------------------------------------------------------
print("PART 4 - LLM PREDICTION EXPLANATION  (mock:", ARGS.mock_llm, ")")
print("Runtime mode:", "mock" if USE_MOCK else "real-api")

if not os.path.exists(PART3_MODEL):
    raise FileNotFoundError("Missing part3/best_model.pkl. Run part3 script first.")

model = joblib.load(PART3_MODEL)
print("Loaded best_model.pkl")

# simple connectivity test
print("\nConnectivity test:")
raw_test, status = safe_call(SYSTEM_PROMPT, "Reply with only the word: hello", temperature=0.0)
print("Status:", status, "  Response:", raw_test)

# guardrail demo
print("\nGuardrail test:")
_, s1 = safe_call(SYSTEM_PROMPT, "My email is alice@example.com. Explain this.", 0.0)
_, s2 = safe_call(SYSTEM_PROMPT, "Explain this prediction in JSON only.", 0.0)
print("PII input:", s1, "(expected: blocked)")
print("Clean input:", s2, "(expected: passed)")

# 3 feature inputs to explain
inputs = [
    {"carat": 0.30, "cut": 4.0, "color": 6.0, "clarity": 7.0,
     "depth": 61.5, "table": 55.0, "x": 4.29, "y": 4.31, "z": 2.64},
    {"carat": 1.20, "cut": 3.0, "color": 4.0, "clarity": 3.0,
     "depth": 62.0, "table": 58.0, "x": 6.80, "y": 6.75, "z": 4.20},
    {"carat": 1.80, "cut": 1.0, "color": 1.0, "clarity": 1.0,
     "depth": 63.5, "table": 61.0, "x": 7.65, "y": 7.60, "z": 4.85}
]

result_rows = []
ab_rows     = []

print("\nRunning on 3 inputs:")
for i, feats in enumerate(inputs, 1):
    row = pd.DataFrame([feats])
    pred_class = int(model.predict(row)[0])
    prob       = float(model.predict_proba(row)[0, 1])

    user_prompt = (
        "Feature values:\n" + json.dumps(feats, indent=2) +
        "\n\nPredicted class: " + str(pred_class) +
        "\nPredicted probability for class 1: " + f"{prob:.4f}" +
        "\n\nExplain this prediction as JSON only."
    )

    raw_t0,  g0  = safe_call(SYSTEM_PROMPT, user_prompt, 0.0)
    raw_t07, g07 = safe_call(SYSTEM_PROMPT, user_prompt, 0.7)

    parsed, valid, err = parse_and_validate(raw_t0)

    print(f"\nInput {i}  class={pred_class}  P(1)={prob:.4f}")
    print("  Raw (temp=0):", raw_t0)
    print("  Valid:", valid, " Error:", err)

    result_rows.append({
        "input": i,
        "features": json.dumps(feats),
        "pred_class": pred_class,
        "prob": prob,
        "raw_temp0": raw_t0,
        "valid_temp0": valid,
        "guardrail": g0
    })

    # check key difference between temp 0 and 0.7
    p0,  v0,  _ = parse_and_validate(raw_t0)
    p07, v07, _ = parse_and_validate(raw_t07)
    key_diff = "no difference"
    if raw_t0 and raw_t07 and raw_t0.strip() != raw_t07.strip():
        key_diff = "content varies"

    ab_rows.append({
        "input": i,
        "output_temp0":  raw_t0,
        "output_temp07": raw_t07,
        "valid_temp0":   v0,
        "valid_temp07":  v07,
        "key_diff":      key_diff,
        "next_step_t0":  p0.get("next_step"),
        "next_step_t07": p07.get("next_step")
    })

pd.DataFrame(result_rows).to_csv(os.path.join(OUT, "part4_trackc_results.csv"), index=False)
pd.DataFrame(ab_rows).to_csv(os.path.join(OUT, "part4_temperature_ab.csv"), index=False)

print("\nDone. Outputs in", OUT)
