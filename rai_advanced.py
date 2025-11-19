import os
import json
import re
from statistics import mean, pstdev

from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

# ------------------------------------------------------------------------------------
#  FLASK APP
# ------------------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# ------------------------------------------------------------------------------------
#  MODEL INITIALIZATION
# ------------------------------------------------------------------------------------
GENERATION_MODEL = None
ANALYSIS_MODEL = None
MODEL_INIT_ERROR = None


def init_models():
    """
    Initialize Gemini models safely.
    """
    global GENERATION_MODEL, ANALYSIS_MODEL, MODEL_INIT_ERROR

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        MODEL_INIT_ERROR = "GEMINI_API_KEY environment variable missing."
        return

    try:
        genai.configure(api_key=api_key)

        GENERATION_MODEL = genai.GenerativeModel(
            "gemini-2.0-flash",
            system_instruction="Be safe, factual, short, neutral, and refuse harmful content.",
        )

        ANALYSIS_MODEL = genai.GenerativeModel(
            "gemini-2.5-flash",
            system_instruction="Always output STRICT JSON only.",
        )

    except Exception as e:
        MODEL_INIT_ERROR = f"Model init failed: {e}"


init_models()


def _ensure_ready():
    if MODEL_INIT_ERROR:
        raise RuntimeError(MODEL_INIT_ERROR)
    if GENERATION_MODEL is None or ANALYSIS_MODEL is None:
        raise RuntimeError("Models not ready.")


# ------------------------------------------------------------------------------------
#  SAFETY LAYER: TEXT EXTRACTION (FIXES: finish_reason=2, missing parts)
# ------------------------------------------------------------------------------------
def safe_extract_text(response):
    """
    Safely extract text from Gemini responses even when safety-blocked.
    """
    if response is None:
        return "[Error: Empty response]"

    # Check finish reason
    try:
        if hasattr(response, "candidates") and response.candidates:
            finish = response.candidates[0].finish_reason
            if finish == 2:
                return "[Model blocked response due to safety rules.]"
    except Exception:
        pass

    # Try normal extraction
    try:
        if hasattr(response, "text") and response.text:
            return response.text
    except Exception:
        pass

    # Try to manually extract text parts
    try:
        parts = response.candidates[0].content.parts
        return " ".join(p.text for p in parts if hasattr(p, "text"))
    except Exception:
        return "[Error: No valid text returned]"


# ------------------------------------------------------------------------------------
#  SAFETY LAYER: JSON EXTRACTION (FIXES MALFORMED JSON)
# ------------------------------------------------------------------------------------
def safe_json_extract(raw):
    """
    Convert Gemini output into valid JSON.
    Handles:
    - code fences
    - leading text
    - trailing commas
    - broken braces
    """
    if not raw:
        return {"error": "Empty model response", "raw": raw}

    cleaned = raw.strip()

    # Remove code fences
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    # Keep only the JSON braces content
    if "{" in cleaned and "}" in cleaned:
        cleaned = cleaned[cleaned.find("{"):]
        cleaned = cleaned[: cleaned.rfind("}") + 1]

    # Fix trailing commas
    cleaned = cleaned.replace(",}", "}").replace(",]", "]")

    # Try parsing
    try:
        return json.loads(cleaned)
    except Exception:
        return {"error": "Invalid JSON", "raw": raw}


# ------------------------------------------------------------------------------------
#  MODEL HELPERS
# ------------------------------------------------------------------------------------
def call_model_text(prompt, model=None, temperature=0.4, max_output_tokens=768):
    _ensure_ready()
    if model is None:
        model = GENERATION_MODEL
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )
        return safe_extract_text(response)
    except Exception as e:
        return f"[Error: {e}]"


def call_model_json(prompt, model=None, temperature=0.2, max_output_tokens=512):
    _ensure_ready()
    if model is None:
        model = ANALYSIS_MODEL
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                response_mime_type="application/json",
            ),
        )
        raw = safe_extract_text(response)
        return safe_json_extract(raw)
    except Exception as e:
        return {"error": f"Model call failed: {e}"}


# ------------------------------------------------------------------------------------
#  UTILS
# ------------------------------------------------------------------------------------
def tokenize_prompt(prompt):
    if not prompt:
        return []
    tokens = re.split(r"[\\s,.;:!?()\\[\\]\"'\\-]+", prompt)
    return [t.strip() for t in tokens if t.strip()]


def ablate_prompt(prompt, changes):
    if not changes:
        return prompt
    terms = [c.strip() for c in changes.split(",") if c.strip()]
    pattern = "|".join([re.escape(t) for t in terms])
    return re.sub(pattern, "", prompt, flags=re.IGNORECASE).strip()


def swap_terms(prompt, mapping_text):
    if not mapping_text:
        return prompt
    mappings = {}
    for part in mapping_text.split(","):
        if "->" in part:
            left, right = [x.strip() for x in part.split("->")]
            if left and right:
                mappings[left] = right

    new = prompt
    for left, right in mappings.items():
        new = re.sub(re.escape(left), right, new, flags=re.IGNORECASE)
    return new.strip()


def summarize_change(original, new):
    prompt = f"""
Return JSON:
{{
  "semantic_change_score": number,
  "change_summary": "string"
}}
ORIGINAL:
{original}
NEW:
{new}
"""
    data = call_model_json(prompt)
    return (
        float(data.get("semantic_change_score", 0)),
        data.get("change_summary", ""),
    )


# ------------------------------------------------------------------------------------
#  FEATURE BUILDERS
# ------------------------------------------------------------------------------------
def build_analysis(prompt, output):
    schema = """
Return JSON:
{
 "heatmap_data": [{"word": "", "impact_score": 0}],
 "connections": [{"prompt_word": "", "impact_score": 0, "influenced_output_words": []}],
 "notes": ""
}
"""
    final_prompt = schema + f"\nPROMPT:\n{prompt}\nOUTPUT:\n{output}"
    data = call_model_json(final_prompt)

    data.setdefault("heatmap_data", [])
    data.setdefault("connections", [])
    data.setdefault("notes", "")

    return data


def build_simple(prompt, output, label):
    data = call_model_json(f"{label}:\nPrompt:\n{prompt}\nOutput:\n{output}")
    return data


# ------------------------------------------------------------------------------------
#  ROUTES
# ------------------------------------------------------------------------------------
@app.route("/health")
def health():
    if MODEL_INIT_ERROR:
        return jsonify({"status": "error", "detail": MODEL_INIT_ERROR}), 500
    return jsonify({"status": "ok"})


@app.route("/api/generate", methods=["POST"])
def api_generate():
    b = request.get_json()
    prompt = (b.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    out = call_model_text(prompt)
    return jsonify({"output": out})


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    b = request.get_json()
    prompt = (b.get("prompt") or "").strip()
    output = (b.get("output") or "").strip()
    return jsonify(build_analysis(prompt, output))


@app.route("/api/run_experiment", methods=["POST"])
def api_experiment():
    b = request.get_json()
    mode = (b.get("mode") or "").strip()
    original = (b.get("original_prompt") or "").strip()
    output = (b.get("original_output") or "").strip()
    changes = (b.get("changes_text") or "").strip()

    if mode == "ablation":
        new_prompt = ablate_prompt(original, changes)
    else:
        new_prompt = swap_terms(original, changes)

    new_output = call_model_text(new_prompt)
    score, summary = summarize_change(output, new_output)

    return jsonify({
        "new_prompt": new_prompt,
        "new_output": new_output,
        "change_data": {
            "semantic_change_score": score,
            "change_summary": summary,
        }
    })


@app.route("/api/<path:tool>", methods=["POST"])
def api_tool(tool):
    b = request.get_json()
    prompt = (b.get("prompt") or "").strip()
    output = (b.get("output") or "").strip()

    label_map = {
        "causal_heatmap": "CAUSAL HEATMAP",
        "fragility_map": "FRAGILITY MAP",
        "paraphrase_robustness": "PARAPHRASE ROBUSTNESS",
        "bias_probe": "FAIRNESS PROBE",
        "prompt_genome": "PROMPT GENOME",
        "temperature_sensitivity": "TEMPERATURE SENSITIVITY",
        "reliability_certificate": "RELIABILITY CERTIFICATE",
    }

    if tool not in label_map:
        return jsonify({"error": f"Unknown tool {tool}"}), 400

    result = build_simple(prompt, output, label_map[tool])
    return jsonify(result)


# ------------------------------------------------------------------------------------
#  LOCAL DEV
# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)), debug=False)
