import os
import json
import math
import re
from statistics import mean, pstdev

from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

# Vercel looks for a WSGI app called `app` in this file
app = Flask(__name__)
CORS(app)

# ----------------------------- 1. GEMINI CONFIG -----------------------------

GENERATION_MODEL = None
ANALYSIS_MODEL = None
MODEL_INIT_ERROR = None


def init_models():
    """
    Initialize Gemini models once at startup.
    """
    global GENERATION_MODEL, ANALYSIS_MODEL, MODEL_INIT_ERROR

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        MODEL_INIT_ERROR = "GOOGLE_API_KEY environment variable not set."
        return

    try:
        genai.configure(api_key=api_key)

        GENERATION_MODEL = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=(
                "You are a careful, concise response model. "
                "For safety and fairness-related prompts, you must default to the safest, "
                "most neutral, non-harmful answer possible."
            ),
        )

        ANALYSIS_MODEL = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=(
                "You are a careful, structured analysis model that always outputs valid JSON "
                "matching the requested schema."
            ),
        )

    except Exception as e:
        MODEL_INIT_ERROR = f"Error initializing models: {e}"


init_models()


def _ensure_models_ready():
    if MODEL_INIT_ERROR:
        raise RuntimeError(MODEL_INIT_ERROR)
    if GENERATION_MODEL is None or ANALYSIS_MODEL is None:
        raise RuntimeError("Models not ready. Check API key and initialization.")


# ----------------------------- 2. JSON SAFETY LAYER -----------------------------

def safe_json_extract(raw_text: str):
    """
    Safely extract valid JSON from Gemini output.

    Handles:
    - ```json ... ``` fences
    - trailing commas
    - leading explanations
    - extra whitespace
    - partial JSON
    """
    if not raw_text:
        return {"error": "Empty model response", "raw": raw_text}

    raw = raw_text.strip()

    # Remove code fences
    raw = raw.replace("```json", "").replace("```", "").strip()

    # Keep only JSON braces
    if "{" in raw and "}" in raw:
        raw = raw[raw.find("{"):]
        raw = raw[: raw.rfind("}") + 1]

    # Remove common Gemini trailing commas
    raw = raw.replace(",}", "}").replace(",]", "]")

    try:
        return json.loads(raw)
    except Exception:
        return {"error": "Invalid JSON returned from model", "raw": raw_text}


# ----------------------------- 3. SMALL UTILS ---------------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def safe_mean(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return mean(values) if values else 0.0


def safe_pstdev(values):
    values = [v for v in values if isinstance(v, (int, float))]
    if len(values) <= 1:
        return 0.0
    return pstdev(values)


def tokenize_prompt(prompt: str):
    if not prompt:
        return []
    tokens = re.split(r"[\s,\.;:!?()\[\]\"\'\-]+", prompt)
    return [t for t in tokens if t.strip()]


# ----------------------------- 4. MODEL CALL HELPERS -----------------------------

def call_model_json(prompt: str, model=None, temperature: float = 0.2, max_output_tokens: int = 512):
    _ensure_models_ready()
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

        return safe_json_extract(response.text)

    except Exception as e:
        return {"error": f"Model call failed: {e}"}


def call_model_text(prompt: str, model=None, temperature: float = 0.4, max_output_tokens: int = 768):
    _ensure_models_ready()
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

        return response.text or ""
    except Exception as e:
        return f"[Error: {e}]"


# ----------------------------- 5. CORE ANALYSIS -----------------------------

def build_prompt_for_analysis(prompt: str, output: str) -> str:
    system = """
You must output ONLY a single JSON object with:
{
  "heatmap_data": [{ "word": "...", "impact_score": number }],
  "connections": [{ "prompt_word": "...", "impact_score": number, "influenced_output_words": [] }],
  "notes": "..."
}
"""
    return system + "\n\n" + json.dumps({"prompt": prompt, "model_output": output}, ensure_ascii=False)


def analyze_prompt_influence(prompt: str, output: str):
    data = call_model_json(build_prompt_for_analysis(prompt, output))
    data.setdefault("heatmap_data", [])
    data.setdefault("connections", [])
    data.setdefault("notes", "")
    return data


# ----------------------------- 6. EXPERIMENT FUNCTIONS -----------------------------

def ablate_prompt(original_prompt: str, changes_text: str) -> str:
    if not changes_text:
        return original_prompt
    terms = [t.strip().lower() for t in changes_text.split(",") if t.strip()]
    if not terms:
        return original_prompt
    pattern = r"|".join([re.escape(t) for t in terms])
    return re.sub(pattern, "", original_prompt, flags=re.IGNORECASE).strip()


def swap_terms_in_prompt(original_prompt: str, mapping_text: str) -> str:
    if not mapping_text:
        return original_prompt
    mappings = {}
    for part in mapping_text.split(","):
        if "->" in part:
            left, right = [x.strip() for x in part.split("->")]
            if left and right:
                mappings[left] = right
    new_prompt = original_prompt
    for left, right in mappings.items():
        new_prompt = re.sub(re.escape(left), right, new_prompt, flags=re.IGNORECASE)
    return new_prompt.strip()


def summarize_change(original_output: str, new_output: str):
    prompt = f"""
Return JSON:
{{"semantic_change_score": number, "change_summary": "string"}}

ORIGINAL:
{original_output}

NEW:
{new_output}
"""
    data = call_model_json(prompt)
    return float(data.get("semantic_change_score", 0.0)), data.get("change_summary", "")


# ----------------------------- 7. ADVANCED ANALYSIS -----------------------------

def build_causal_heatmap(prompt, output):
    data = call_model_json(f"CAUSAL HEATMAP:\nPrompt:\n{prompt}\nOutput:\n{output}")
    data.setdefault("causal_scores", [])
    data.setdefault("notes", "")
    return data


def build_fragility_map(prompt, output):
    data = call_model_json(f"FRAGILITY MAP:\nPrompt:\n{prompt}\nOutput:\n{output}")
    data.setdefault("fragility_scores", [])
    data.setdefault("notes", "")
    return data


def build_paraphrase_robustness(prompt, output):
    data = call_model_json(f"PARAPHRASE ROBUSTNESS:\nPrompt:\n{prompt}\nOutput:\n{output}")
    data.setdefault("paraphrase_cases", [])
    data.setdefault("notes", "")
    return data


def build_bias_probe(prompt, output):
    data = call_model_json(f"FAIRNESS PROBE:\nPrompt:\n{prompt}\nOutput:\n{output}")
    data.setdefault("fairness_findings", [])
    data.setdefault("overall_risk_summary", "")
    return data


def build_prompt_genome(prompt, output):
    data = call_model_json(f"PROMPT GENOME:\nPrompt:\n{prompt}\nOutput:\n{output}")
    data.setdefault("blocks", [])
    data.setdefault("notes", "")
    return data


def build_temperature_sensitivity(prompt):
    data = call_model_json(f"TEMPERATURE SENSITIVITY:\nPrompt:\n{prompt}")
    data.setdefault("temperature_cases", [])
    data.setdefault("global_comment", "")
    return data


def build_reliability_certificate(prompt, output):
    data = call_model_json(f"RELIABILITY CERTIFICATE:\nPrompt:\n{prompt}\nOutput:\n{output}")
    data.setdefault("grade", "C")
    data.setdefault("headline", "")
    data.setdefault("key_strengths", [])
    data.setdefault("key_risks", [])
    data.setdefault("suggested_improvements", [])
    return data


# ----------------------------- 8. ROUTES -------------------------------------

@app.route("/health")
def health():
    if MODEL_INIT_ERROR:
        return jsonify({"status": "error", "detail": MODEL_INIT_ERROR}), 500
    return jsonify({"status": "ok"})


@app.route("/api/generate", methods=["POST"])
def api_generate():
    body = request.get_json(force=True)
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400
    text = call_model_text(prompt)
    return jsonify({"output": text})


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    body = request.get_json(force=True)
    prompt = (body.get("prompt") or "").strip()
    output = (body.get("output") or "").strip()
    if not prompt or not output:
        return jsonify({"error": "Both prompt and output are required."}), 400
    return jsonify(analyze_prompt_influence(prompt, output))


@app.route("/api/run_experiment", methods=["POST"])
def api_run_experiment():
    body = request.get_json(force=True)
    mode = (body.get("mode") or "").strip().lower()
    original_prompt = (body.get("original_prompt") or "").strip()
    original_output = (body.get("original_output") or "").strip()
    changes_text = (body.get("changes_text") or "").strip()

    if mode not in {"ablation", "swap"}:
        return jsonify({"error": "mode must be 'ablation' or 'swap'"}), 400

    new_prompt = (
        ablate_prompt(original_prompt, changes_text)
        if mode == "ablation"
        else swap_terms_in_prompt(original_prompt, changes_text)
    )

    new_output = call_model_text(new_prompt)
    score, summary = summarize_change(original_output, new_output)

    return jsonify({
        "new_prompt": new_prompt,
        "new_output": new_output,
        "change_data": {
            "semantic_change_score": score,
            "change_summary": summary,
        },
    })


@app.route("/api/causal_heatmap", methods=["POST"])
def api_causal_heatmap():
    body = request.get_json(force=True)
    return jsonify(build_causal_heatmap(body.get("prompt", ""), body.get("output", "")))


@app.route("/api/fragility_map", methods=["POST"])
def api_fragility_map():
    body = request.get_json(force=True)
    return jsonify(build_fragility_map(body.get("prompt", ""), body.get("output", "")))


@app.route("/api/paraphrase_robustness", methods=["POST"])
def api_paraphrase_robustness():
    body = request.get_json(force=True)
    return jsonify(build_paraphrase_robustness(body.get("prompt", ""), body.get("output", "")))


@app.route("/api/bias_probe", methods=["POST"])
def api_bias_probe():
    body = request.get_json(force=True)
    return jsonify(build_bias_probe(body.get("prompt", ""), body.get("output", "")))


@app.route("/api/prompt_genome", methods=["POST"])
def api_prompt_genome():
    body = request.get_json(force=True)
    return jsonify(build_prompt_genome(body.get("prompt", ""), body.get("output", "")))


@app.route("/api/temperature_sensitivity", methods=["POST"])
def api_temperature_sensitivity():
    body = request.get_json(force=True)
    return jsonify(build_temperature_sensitivity(body.get("prompt", "")))


@app.route("/api/reliability_certificate", methods=["POST"])
def api_reliability_certificate():
    body = request.get_json(force=True)
    return jsonify(build_reliability_certificate(body.get("prompt", ""), body.get("output", "")))


# ----------------------------- 9. LOCAL RUNNER ---------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=False)
