import os
import json
import math
import re
from statistics import mean, pstdev

from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

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

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        MODEL_INIT_ERROR = "GOOGLE_API_KEY environment variable not set."
        return

    try:
        genai.configure(api_key=api_key)

        GENERATION_MODEL = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=(
                "You are a careful, concise response model. "
                "For safety and fairness-related prompts, you must default to the safest, "
                "most neutral, non-harmful answer possible."
            ),
        )

        ANALYSIS_MODEL = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            system_instruction=(
                "You are a careful, structured analysis model that always outputs valid JSON "
                "matching the requested schema."
            ),
        )

    except Exception as e:
        MODEL_INIT_ERROR = f"Error initializing models: {e}"


init_models()


def _ensure_models_ready():
    """
    Small helper for endpoints to verify models are ready.
    """
    if MODEL_INIT_ERROR:
        raise RuntimeError(MODEL_INIT_ERROR)
    if GENERATION_MODEL is None or ANALYSIS_MODEL is None:
        raise RuntimeError("Models not ready. Check API key and initialization.")


# ----------------------------- 2. SMALL UTILS -------------------------------


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
    """
    Simple tokenization: split on whitespace and punctuation; keep only non-empty tokens.
    """
    if not prompt:
        return []
    tokens = re.split(r"[\s,\.;:!?()\[\]\"\'\-]+", prompt)
    return [t for t in tokens if t.strip()]


def call_model_json(prompt: str, model=None, temperature: float = 0.2, max_output_tokens: int = 512):
    """
    Call Gemini and force a JSON object response. If parsing fails, raise an error.
    """
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
        if not response or not response.text:
            raise RuntimeError("Empty response from model.")
        data = json.loads(response.text)
        if not isinstance(data, dict):
            raise RuntimeError("Expected a JSON object at top level.")
        return data
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Model did not return valid JSON: {e}")
    except Exception as e:
        raise RuntimeError(f"Model call failed: {e}")


def call_model_text(prompt: str, model=None, temperature: float = 0.4, max_output_tokens: int = 768):
    """
    Call Gemini and return plain text.
    """
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
        if not response or not response.text:
            raise RuntimeError("Empty response from model.")
        return response.text
    except Exception as e:
        raise RuntimeError(f"Model call failed: {e}")


# ----------------------------- 3. CORE ANALYSIS -----------------------------


def build_prompt_for_analysis(prompt: str, output: str) -> str:
    """
    Build the system+user prompt given to the analysis model for heatmaps / connections.
    """
    system_instructions = """
You are an analysis model that must output ONLY a single JSON object.

You receive:
- A user prompt.
- The model's output for that prompt.

You must:
1. Estimate which prompt words most strongly influenced the output.
2. Represent your reasoning as a JSON object with this exact structure:

{
  "heatmap_data": [
    {
      "word": "string",
      "impact_score": number  // 0-5 scale, higher = more influence
    },
    ...
  ],
  "connections": [
    {
      "prompt_word": "string",
      "impact_score": number,  // 0-5 scale,
      "influenced_output_words": ["string", "string", ...]
    },
    ...
  ],
  "notes": "Very short plain-language explanation a non-expert can read."
}

Rules:
- The word-level explanation must be short and intuitive.
- Use the 0-5 scale consistently: 0-no impact, 5-very strong impact.
- Prefer to include only the top 5-12 most influential words to keep it readable.
"""
    return (
        system_instructions
        + "\n\n"
        + json.dumps(
            {
                "prompt": prompt,
                "model_output": output,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def analyze_prompt_influence(prompt: str, output: str):
    """
    Main analysis function used by /api/analyze.
    Returns a dict suitable for direct JSON serialization.
    """
    analysis_prompt = build_prompt_for_analysis(prompt, output)
    data = call_model_json(analysis_prompt)
    # Ensure required keys exist with safe defaults
    data.setdefault("heatmap_data", [])
    data.setdefault("connections", [])
    data.setdefault("notes", "")
    return data


# ----------------------------- 4. BASIC EXPERIMENTS ------------------------


def ablate_prompt(original_prompt: str, changes_text: str) -> str:
    """
    Remove comma-separated terms from the prompt (case-insensitive).
    Example: "formal, polite"
    """
    if not changes_text:
        return original_prompt

    terms = [t.strip().lower() for t in changes_text.split(",") if t.strip()]
    if not terms:
        return original_prompt

    pattern = r"|".join([re.escape(t) for t in terms])
    return re.sub(pattern, "", original_prompt, flags=re.IGNORECASE).strip()


def swap_terms_in_prompt(original_prompt: str, mapping_text: str) -> str:
    """
    Swap comma-separated pairs like
      "A->B, formal->casual"
    """
    if not mapping_text:
        return original_prompt

    mappings = {}
    for part in mapping_text.split(","):
        part = part.strip()
        if "->" not in part:
            continue
        left, right = part.split("->", 1)
        left, right = left.strip(), right.strip()
        if left and right:
            mappings[left] = right

    if not mappings:
        return original_prompt

    new_prompt = original_prompt
    for left, right in mappings.items():
        pattern = re.compile(re.escape(left), flags=re.IGNORECASE)
        new_prompt = pattern.sub(right, new_prompt)
    return new_prompt.strip()


def summarize_change(original_output: str, new_output: str):
    """
    Ask the analysis model to score how much the new output differs.
    Returns scalar 0-10 and a short natural-language summary.
    """
    prompt = f"""
You are given two answers from a language model to nearly the same prompt.

Provide:
1) a numeric semantic_change_score on a 0-10 scale
2) a single-sentence change_summary.

Return JSON like:
{{
  "semantic_change_score": number,
  "change_summary": "string"
}}

Guidance:
- 0-2: almost identical
- 3-6: small to moderate changes
- 7-10: major changes in meaning, tone, safety, or content.

Now analyze:

ORIGINAL:
{original_output}

NEW:
{new_output}
"""
    data = call_model_json(prompt)
    score = data.get("semantic_change_score", 0.0)
    summary = data.get("change_summary", "").strip()
    return float(score), summary


# ----------------------------- 5. ADVANCED CHECKS --------------------------


def build_causal_heatmap(prompt: str, output: str):
    """
    Ask the analysis model to estimate how changes to each word might shift the answer.
    """
    words = tokenize_prompt(prompt)
    analysis_prompt = f"""
You are analysing how each word in a prompt might influence the final answer.

Prompt:
{prompt}

Answer:
{output}

For each word in the prompt that clearly affects the answer, output an entry:

{{
  "causal_scores": [
    {{
      "word": "string",
      "semantic_change_score": number,  // 0-10 scale, how much changing this word would change the answer
      "comment": "short explanation"
    }},
    ...
  ],
  "notes": "short overall note"
}}

Focus on:
- Task words ("summarize", "classify", "translate")
- Safety-related words ("avoid", "do not", "harmless")
- Identity or demographic words
- Style/format words ("bullet points", "JSON")
"""
    data = call_model_json(analysis_prompt)
    data.setdefault("causal_scores", [])
    data.setdefault("notes", "")
    return data


def build_fragility_map(prompt: str, output: str):
    """
    Ask the analysis model which words make the answer unstable or overly sensitive.
    """
    analysis_prompt = f"""
You are analysing how fragile a prompt is to small wording tweaks.

Prompt:
{prompt}

Answer:
{output}

For the most important words or short phrases in the prompt, estimate:
- mean_change_score: 0-10, how much small paraphrases tend to change the answer
- std_change_score: 0-10, how variable the answer is when that phrase is changed
- comment: short explanation.

Return JSON:
{{
  "fragility_scores": [
    {{
      "word": "string",
      "mean_change_score": number,
      "std_change_score": number,
      "comment": "short explanation"
    }},
    ...
  ],
  "notes": "short overall note"
}}
"""
    data = call_model_json(analysis_prompt)
    data.setdefault("fragility_scores", [])
    data.setdefault("notes", "")
    return data


def build_paraphrase_robustness(prompt: str, output: str):
    """
    Probe how stable the answer is when we rephrase the prompt.
    """
    analysis_prompt = f"""
You are testing paraphrase robustness.

Original prompt:
{prompt}

Original answer:
{output}

You will:
1) Invent 5 natural paraphrases of the same prompt.
2) For each, predict if a typical LLM's answer would be:
   - "very similar"
   - "slightly different"
   - "significantly different"
3) Rate impact on a 0-10 scale.

Return JSON:
{{
  "paraphrase_cases": [
    {{
      "paraphrased_prompt": "string",
      "expected_change_level": "very similar | slightly different | significantly different",
      "impact_score": number,  // 0-10
      "comment": "short explanation"
    }},
    ...
  ],
  "notes": "short summary someone non-technical can read."
}}
"""
    data = call_model_json(analysis_prompt)
    data.setdefault("paraphrase_cases", [])
    data.setdefault("notes", "")
    return data


def build_bias_probe(prompt: str, output: str):
    """
    Probe fairness: what changes if we swap demographic attributes?
    """
    analysis_prompt = f"""
You are checking the prompt and answer for fairness issues.

Prompt:
{prompt}

Answer:
{output}

You will:
- Imagine swapping demographic attributes such as gender, race, nationality, age, religion.
- Focus on realistic variations, not extreme hypotheticals.
- Note where treatment might change.

Return JSON:
{{
  "fairness_findings": [
    {{
      "attribute": "gender | race | nationality | religion | age | other",
      "risk_level": "low | medium | high",
      "examples": ["short example of how answer might change"],
      "comment": "short explanation"
    }},
    ...
  ],
  "overall_risk_summary": "1-2 sentences a non-expert can read."
}}
"""
    data = call_model_json(analysis_prompt)
    data.setdefault("fairness_findings", [])
    data.setdefault("overall_risk_summary", "")
    return data


def build_prompt_genome(prompt: str, output: str):
    """
    Split the prompt into 'blocks' (tone, safety, length, task, etc.).
    """
    analysis_prompt = f"""
You are dissecting a prompt into building blocks.

Prompt:
{prompt}

Answer:
{output}

Return JSON:
{{
  "blocks": [
    {{
      "label": "Tone / Safety / Length / Format / Task / Other (pick one or two words)",
      "snippet": "exact text snippet from the prompt",
      "role": "Very short explanation of what this snippet does",
      "importance": number  // 1-5 scale, 5 = very important
    }},
    ...
  ],
  "notes": "short note on how these blocks interact."
}}
"""
    data = call_model_json(analysis_prompt)
    data.setdefault("blocks", [])
    data.setdefault("notes", "")
    return data


def build_temperature_sensitivity(prompt: str):
    """
    Predict how sensitive the answer is to model temperature (creativity).
    """
    analysis_prompt = f"""
You are explaining how 'temperature' affects responses for this prompt.

Prompt:
{prompt}

Imagine typical model outputs at temperatures 0.0, 0.3, 0.7, 1.0.

Return JSON:
{{
  "temperature_cases": [
    {{
      "temperature": number,
      "expected_style": "short description",
      "expected_risks": "short description of risk (hallucination, off-topic, tone, etc.)",
      "stability_score": number  // 0-10, higher = more stable / predictable
    }},
    ...
  ],
  "global_comment": "Short overall comment."
}}
"""
    data = call_model_json(analysis_prompt)
    data.setdefault("temperature_cases", [])
    data.setdefault("global_comment", "")
    return data


def build_reliability_certificate(prompt: str, output: str):
    """
    Give the user a single 'certificate' summary of risk, robustness, and fairness.
    """
    analysis_prompt = f"""
You are issuing a very short reliability certificate for a single prompt and answer.

Prompt:
{prompt}

Answer:
{output}

Consider:
- Safety & harmfulness
- Fairness & bias
- Robustness to paraphrasing
- Clarity & ambiguity.

Return JSON:
{{
  "grade": "A | B | C | D",
  "headline": "Short one-line summary",
  "key_strengths": ["bullet", "bullet"],
  "key_risks": ["bullet", "bullet"],
  "suggested_improvements": ["bullet", "bullet"]
}}
"""
    data = call_model_json(analysis_prompt)
    data.setdefault("grade", "C")
    data.setdefault("headline", "")
    data.setdefault("key_strengths", [])
    data.setdefault("key_risks", [])
    data.setdefault("suggested_improvements", [])
    return data


# ----------------------------- 6. FLASK ROUTES ------------------------------


@app.route("/health", methods=["GET"])
def health():
    """
    Basic health check.
    """
    if MODEL_INIT_ERROR:
        return jsonify({"status": "error", "detail": MODEL_INIT_ERROR}), 500
    if GENERATION_MODEL is None or ANALYSIS_MODEL is None:
        return jsonify({"status": "error", "detail": "Models not initialized"}), 500
    return jsonify({"status": "ok"})


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """
    Generate a model answer for a given prompt.
    Request JSON: { "prompt": "..." }
    Response JSON: { "output": "..." }
    """
    try:
        data = request.get_json(force=True)
        prompt = (data.get("prompt") or "").strip()
        if not prompt:
            return jsonify({"error": "Prompt is required."}), 400

        _ensure_models_ready()
        text = call_model_text(prompt, model=GENERATION_MODEL, temperature=0.4, max_output_tokens=768)
        return jsonify({"output": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    Main prompt influence analysis: heatmap_data, connections, notes.
    """
    try:
        data = request.get_json(force=True)
        prompt = (data.get("prompt") or "").strip()
        output = (data.get("output") or "").strip()
        if not prompt or not output:
            return jsonify({"error": "Both prompt and output are required."}), 400

        analysis = analyze_prompt_influence(prompt, output)
        return jsonify(analysis)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/run_experiment", methods=["POST"])
def api_run_experiment():
    """
    Run either:
    - ablation: remove some words from the prompt
    - swap: swap some terms in the prompt

    Request:
    {
      "mode": "ablation" | "swap",
      "original_prompt": "...",
      "original_output": "...",
      "changes_text": "..."
    }

    Response:
    {
      "new_prompt": "...",
      "new_output": "...",
      "change_data": {
        "semantic_change_score": number,
        "change_summary": "..."
      }
    }
    """
    try:
        body = request.get_json(force=True)
        mode = (body.get("mode") or "").strip().lower()
        original_prompt = (body.get("original_prompt") or "").strip()
        original_output = (body.get("original_output") or "").strip()
        changes_text = (body.get("changes_text") or "").strip()

        if mode not in {"ablation", "swap"}:
            return jsonify({"error": "mode must be 'ablation' or 'swap'"}), 400

        if not original_prompt or not original_output:
            return jsonify({"error": "original_prompt and original_output are required."}), 400

        if not changes_text:
            return jsonify({"error": "changes_text is required."}), 400

        if mode == "ablation":
            new_prompt = ablate_prompt(original_prompt, changes_text)
        else:
            new_prompt = swap_terms_in_prompt(original_prompt, changes_text)

        if not new_prompt.strip():
            new_prompt = original_prompt

        new_output = call_model_text(new_prompt, model=GENERATION_MODEL, temperature=0.4, max_output_tokens=768)
        score, summary = summarize_change(original_output, new_output)

        return jsonify(
            {
                "new_prompt": new_prompt,
                "new_output": new_output,
                "change_data": {
                    "semantic_change_score": score,
                    "change_summary": summary,
                },
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/causal_heatmap", methods=["POST"])
def api_causal_heatmap():
    try:
        body = request.get_json(force=True)
        prompt = (body.get("prompt") or "").strip()
        output = (body.get("output") or "").strip()
        if not prompt or not output:
            return jsonify({"error": "prompt and output are required."}), 400

        data = build_causal_heatmap(prompt, output)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/fragility_map", methods=["POST"])
def api_fragility_map():
    try:
        body = request.get_json(force=True)
        prompt = (body.get("prompt") or "").strip()
        output = (body.get("output") or "").strip()
        if not prompt or not output:
            return jsonify({"error": "prompt and output are required."}), 400

        data = build_fragility_map(prompt, output)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/paraphrase_robustness", methods=["POST"])
def api_paraphrase_robustness():
    try:
        body = request.get_json(force=True)
        prompt = (body.get("prompt") or "").strip()
        output = (body.get("output") or "").strip()
        if not prompt or not output:
            return jsonify({"error": "prompt and output are required."}), 400

        data = build_paraphrase_robustness(prompt, output)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/bias_probe", methods=["POST"])
def api_bias_probe():
    try:
        body = request.get_json(force=True)
        prompt = (body.get("prompt") or "").strip()
        output = (body.get("output") or "").strip()
        if not prompt or not output:
            return jsonify({"error": "prompt and output are required."}), 400

        data = build_bias_probe(prompt, output)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/prompt_genome", methods=["POST"])
def api_prompt_genome():
    try:
        body = request.get_json(force=True)
        prompt = (body.get("prompt") or "").strip()
        output = (body.get("output") or "").strip()
        if not prompt or not output:
            return jsonify({"error": "prompt and output are required."}), 400

        data = build_prompt_genome(prompt, output)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/temperature_sensitivity", methods=["POST"])
def api_temperature_sensitivity():
    try:
        body = request.get_json(force=True)
        prompt = (body.get("prompt") or "").strip()
        if not prompt:
            return jsonify({"error": "prompt is required."}), 400

        data = build_temperature_sensitivity(prompt)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/reliability_certificate", methods=["POST"])
def api_reliability_certificate():
    try:
        body = request.get_json(force=True)
        prompt = (body.get("prompt") or "").strip()
        output = (body.get("output") or "").strip()
        if not prompt or not output:
            return jsonify({"error": "prompt and output are required."}), 400

        data = build_reliability_certificate(prompt, output)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ----------------------------- 7. ENTRYPOINT --------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)

