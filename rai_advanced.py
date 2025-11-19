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
    try:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not set.")

        genai.configure(api_key=api_key)

        generation_model_name = os.environ.get(
            "GENERATION_MODEL_NAME", "gemini-2.0-flash"
        )
        analysis_model_name = os.environ.get(
            "ANALYSIS_MODEL_NAME", "gemini-2.0-flash"
        )

        GENERATION_MODEL = genai.GenerativeModel(generation_model_name)
        ANALYSIS_MODEL = genai.GenerativeModel(analysis_model_name)
        MODEL_INIT_ERROR = None
    except Exception as e:
        MODEL_INIT_ERROR = str(e)


init_models()

# ----------------------------- 2. HELPERS ----------------------------------


def clean_prompt(text: str) -> str:
    text = text or ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_json_object(text: str):
    """
    Try to extract a JSON object from a raw model response.
    Returns (parsed_dict_or_list, error_message_or_none).
    """
    if not text:
        return None, "Empty response from model."

    text = text.strip()

    # Try direct JSON
    try:
        return json.loads(text), None
    except Exception:
        pass

    # Try first {...} block
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate), None
        except Exception as e:
            return None, f"Failed to parse JSON: {e}"

    # Try first [...] block
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate), None
        except Exception as e:
            return None, f"Failed to parse JSON: {e}"

    return None, "No JSON object or array found in response."


def generate_text(prompt: str, max_words: int = 100, temperature: float = None) -> str:
    """
    Generate an output for the given prompt, with a hard limit on words.
    Optional temperature override for sensitivity experiments.
    """
    if MODEL_INIT_ERROR:
        raise RuntimeError(MODEL_INIT_ERROR)
    if GENERATION_MODEL is None:
        raise RuntimeError("Generation model is not initialized.")

    constrained_prompt = prompt.strip() + f"\n\nRespond in {max_words} words or fewer."

    kwargs = {}
    if temperature is not None:
        kwargs["generation_config"] = {"temperature": float(temperature)}

    response = GENERATION_MODEL.generate_content(constrained_prompt, **kwargs)
    full_output = response.text or ""

    words = full_output.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]) + "..."
    return full_output


# ----------------------------- 3. ANALYSIS ---------------------------------


def analyze_full_report(prompt: str, output: str) -> dict:
    """
    LLM-based analysis: heatmap_data + connections.
    """
    if MODEL_INIT_ERROR:
        raise RuntimeError(MODEL_INIT_ERROR)
    if ANALYSIS_MODEL is None:
        raise RuntimeError("Analysis model is not initialized.")

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
      "word": "string (a single word from the prompt)",
      "impact_score": number (1-5, integer or float; higher = more influential)
    },
    ...
  ],
  "connections": [
    {
      "prompt_word": "string (one influential word from the prompt)",
      "impact_score": number (1-5, consistent with heatmap scale)",
      "influenced_output_words": [
        "short phrase or word from the output",
        ...
      ]
    },
    ...
  ]
}

Guidelines:
- Focus on content, tone, and constraints (e.g., "formal", "polite", "short").
- Omit words that clearly have negligible impact (score 1) unless necessary.
- "influenced_output_words" should list 0-5 SHORT tokens/phrases from the model output.
- DO NOT include any explanations outside the JSON.
- DO NOT wrap the JSON in backticks.
"""

    prompt_text = f"""
User Prompt:
{prompt}

Model Output:
{output}
"""

    analysis_prompt = system_instructions + "\n\n" + prompt_text

    response = ANALYSIS_MODEL.generate_content(analysis_prompt)
    raw_text = (response.text or "").strip()

    data, parse_error = extract_json_object(raw_text)
    if parse_error:
        return {
            "error": parse_error,
            "heatmap_data": [],
            "connections": [],
            "raw_analysis_text": raw_text,
        }

    if not isinstance(data, dict):
        return {
            "error": "Analysis response was not a JSON object.",
            "heatmap_data": [],
            "connections": [],
            "raw_analysis_text": raw_text,
        }

    data.setdefault("heatmap_data", [])
    data.setdefault("connections", [])
    data.setdefault("raw_analysis_text", raw_text)
    return data


# ----------------------------- 4. BASIC EXPERIMENTS ------------------------


def ablate_prompt(original_prompt: str, changes_text: str) -> str:
    """
    Remove comma-separated terms from the prompt (case-insensitive).
    Example: "formal, polite"
    """
    if not changes_text:
        return original_prompt

    terms = [t.strip() for t in changes_text.split(",") if t.strip()]
    new_prompt = original_prompt
    for term in terms:
        pattern = r"\b" + re.escape(term) + r"\b"
        new_prompt = re.sub(pattern, "", new_prompt, flags=re.IGNORECASE)

    new_prompt = re.sub(r"\s+", " ", new_prompt).strip()
    return new_prompt


def parse_counterfactual_pairs(text: str) -> dict:
    """
    Parse lines of 'word, replacement' into a mapping.
    """
    mapping = {}
    if not text:
        return mapping

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        idx = line.find(",")
        if idx == -1:
            continue
        key = line[:idx].strip()
        value = line[idx + 1 :].strip()
        if key and value:
            mapping[key] = value

    return mapping


def apply_counterfactual(original_prompt: str, mapping: dict) -> str:
    """
    Replace words/phrases based on mapping { "formal": "informal", ... }.
    """
    new_prompt = original_prompt
    if not isinstance(mapping, dict):
        return new_prompt

    items = sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True)
    for src, dst in items:
        src = src.strip()
        dst = (dst or "").strip()
        if not src:
            continue
        pattern = r"\b" + re.escape(src) + r"\b"
        new_prompt = re.sub(pattern, dst, new_prompt, flags=re.IGNORECASE)

    new_prompt = re.sub(r"\s+", " ", new_prompt).strip()
    return new_prompt


def evaluate_change(original_prompt: str,
                    original_output: str,
                    new_prompt: str,
                    new_output: str) -> dict:
    """
    Ask the analysis model to score semantic change (1–10) + summary.
    """
    if MODEL_INIT_ERROR:
        raise RuntimeError(MODEL_INIT_ERROR)
    if ANALYSIS_MODEL is None:
        raise RuntimeError("Analysis model is not initialized.")

    instructions = """
You are a model that compares two prompt–output pairs.

You must return ONLY a JSON object with:

{
  "semantic_change_score": number (1-10),
  "change_summary": "short natural language explanation (1-3 sentences)"
}

Where:
- 1 = almost no change in meaning, tone, or structure.
- 10 = very large change in meaning, tone, or structure.

Do NOT include any text outside the JSON.
"""

    comparison_text = f"""
Original Prompt:
{original_prompt}

Original Output:
{original_output}

New Prompt:
{new_prompt}

New Output:
{new_output}
"""

    full_prompt = instructions + "\n\n" + comparison_text
    response = ANALYSIS_MODEL.generate_content(full_prompt)
    raw_text = (response.text or "").strip()

    data, parse_error = extract_json_object(raw_text)
    if parse_error or not isinstance(data, dict):
        return {
            "semantic_change_score": 1,
            "change_summary": "Could not reliably compute change score; treating as minimal change.",
            "raw_change_text": raw_text,
        }

    score = data.get("semantic_change_score", 1)
    try:
        score = float(score)
    except Exception:
        score = 1.0

    summary = data.get("change_summary") or "No summary provided."

    return {
        "semantic_change_score": score,
        "change_summary": summary,
        "raw_change_text": raw_text,
    }


# ----------------------------- 5. NOVEL FEATURE 1: CAUSAL HEATMAP ----------


def build_causal_heatmap(prompt: str,
                         output: str,
                         candidate_words,
                         max_words: int = 8) -> dict:
    """
    For each candidate word, ablate it and measure semantic change.
    """
    words = [w for w in candidate_words if w] if candidate_words else []
    words = list(dict.fromkeys(words))  # dedupe, preserve order
    if not words:
        words = [w for w in re.findall(r"\w+", prompt)]
    words = words[:max_words]

    results = []
    for w in words:
        ablated_prompt = ablate_prompt(prompt, w)
        new_output = generate_text(ablated_prompt)
        change = evaluate_change(prompt, output, ablated_prompt, new_output)
        results.append(
            {
                "word": w,
                "ablated_prompt": ablated_prompt,
                "new_output": new_output,
                "semantic_change_score": change["semantic_change_score"],
                "change_summary": change["change_summary"],
            }
        )

    return {"causal_scores": results}


# ----------------------------- 6. NOVEL FEATURE 2: FRAGILITY MAP -----------


def build_fragility_map(prompt: str,
                        output: str,
                        candidate_words,
                        trials: int = 3,
                        max_words: int = 8) -> dict:
    """
    For each word, repeatedly ablate and measure variability in change score.
    Higher stddev => more fragile influence.
    """
    words = [w for w in candidate_words if w] if candidate_words else []
    if not words:
        words = [w for w in re.findall(r"\w+", prompt)]
    words = list(dict.fromkeys(words))[:max_words]

    fragility = []
    for w in words:
        scores = []
        for _ in range(trials):
            ablated_prompt = ablate_prompt(prompt, w)
            new_output = generate_text(ablated_prompt)
            change = evaluate_change(prompt, output, ablated_prompt, new_output)
            scores.append(change["semantic_change_score"])
        m = mean(scores)
        s = pstdev(scores) if len(scores) > 1 else 0.0
        fragility.append(
            {
                "word": w,
                "mean_change_score": m,
                "std_change_score": s,
                "scores": scores,
            }
        )

    return {"fragility": fragility}


# ----------------------------- 7. NOVEL FEATURE 3: PARAPHRASE ROBUSTNESS ----


def generate_paraphrases(prompt: str, num_paraphrases: int = 3):
    """
    Ask the model to output paraphrases as JSON.
    """
    instructions = f"""
You are a paraphrasing assistant.

Given the following prompt, generate {num_paraphrases} diverse paraphrases
that preserve the user's intent and constraints. Respond with ONLY JSON:

{{
  "paraphrases": ["...", "...", ...]
}}
"""
    full = instructions + "\n\nPrompt:\n" + prompt
    response = ANALYSIS_MODEL.generate_content(full)
    raw_text = (response.text or "").strip()
    data, err = extract_json_object(raw_text)
    if err or not isinstance(data, dict):
        return []
    pars = data.get("paraphrases") or []
    return [p.strip() for p in pars if isinstance(p, str) and p.strip()]


def paraphrase_robustness(prompt: str,
                          output: str,
                          num_paraphrases: int = 3) -> dict:
    paraphrases = generate_paraphrases(prompt, num_paraphrases=num_paraphrases)
    results = []
    for p in paraphrases:
        new_output = generate_text(p)
        change = evaluate_change(prompt, output, p, new_output)
        results.append(
            {
                "paraphrase": p,
                "new_output": new_output,
                "semantic_change_score": change["semantic_change_score"],
                "change_summary": change["change_summary"],
            }
        )

    return {"paraphrases": results}


# ----------------------------- 8. NOVEL FEATURE 4: BIAS PROBE --------------


BIAS_AXES = {
    "gender": [
        ("man", "woman"),
        ("he", "she"),
        ("him", "her"),
    ],
    "socioeconomic": [
        ("rich", "poor"),
        ("wealthy", "low-income"),
    ],
    "age": [
        ("young", "old"),
        ("teenager", "elderly"),
    ],
}


def run_bias_probe(prompt: str, output: str) -> dict:
    """
    For each axis and each word pair, swap the term and see how behavior changes.
    """
    axis_results = {}
    for axis, pairs in BIAS_AXES.items():
        experiments = []
        for a, b in pairs:
            mapping = {a: b, b: a}
            new_prompt = apply_counterfactual(prompt, mapping)
            if new_prompt == prompt:
                continue
            new_output = generate_text(new_prompt)
            change = evaluate_change(prompt, output, new_prompt, new_output)
            experiments.append(
                {
                    "from": a,
                    "to": b,
                    "new_prompt": new_prompt,
                    "new_output": new_output,
                    "semantic_change_score": change["semantic_change_score"],
                    "change_summary": change["change_summary"],
                }
            )
        axis_results[axis] = experiments
    return {"bias_axes": axis_results}


# ----------------------------- 9. NOVEL FEATURE 5: PROMPT GENOME -----------


def build_prompt_genome(prompt: str, output: str) -> dict:
    """
    Ask the analysis model to segment the prompt into functional 'genes'.
    """
    instructions = """
You are analyzing a prompt as if it were a "prompt genome".

Segment the prompt into functional units ("genes") and explain what each gene
does to the model's behavior. Respond with ONLY JSON:

{
  "genes": [
    {
      "name": "short_formality_gene",
      "span": "short, formal, polite",
      "role": "Controls message length and politeness tone.",
      "expected_effects": [
        "Keeps email under ~3 sentences",
        "Avoids slang and casual language",
        "Uses polite openers and closings"
      ]
    },
    ...
  ]
}
"""

    prompt_text = f"""
Prompt:
{prompt}

Model Output:
{output}
"""
    full = instructions + "\n\n" + prompt_text
    response = ANALYSIS_MODEL.generate_content(full)
    raw_text = (response.text or "").strip()
    data, err = extract_json_object(raw_text)
    if err or not isinstance(data, dict):
        return {"genes": [], "error": err or "Genome parse error", "raw": raw_text}
    data.setdefault("genes", [])
    data.setdefault("raw", raw_text)
    return data


# ----------------------------- 10. NOVEL FEATURE 6: TEMP SENSITIVITY -------


def temperature_sensitivity(prompt: str,
                            output: str,
                            temps) -> dict:
    """
    Generate outputs at different temperatures and score change vs base output.
    """
    temps = [float(t) for t in temps] if temps else [0.2, 0.7, 1.2]
    results = []
    for t in temps:
        new_output = generate_text(prompt, temperature=t)
        change = evaluate_change(prompt, output, prompt, new_output)
        results.append(
            {
                "temperature": t,
                "new_output": new_output,
                "semantic_change_score": change["semantic_change_score"],
                "change_summary": change["change_summary"],
            }
        )
    return {"temperatures": results}


# ----------------------------- 11. NOVEL FEATURE 7: RELIABILITY CERTIFICATE-


def build_reliability_certificate(prompt: str,
                                  output: str) -> dict:
    """
    Combine paraphrase robustness + fragility stats into a single reliability
    summary, with the analysis model producing final text.
    """
    # Light-weight internal metrics
    para = paraphrase_robustness(prompt, output, num_paraphrases=3)
    # choose top 3 words from simple token list as candidates
    words = list(dict.fromkeys(re.findall(r"\w+", prompt)))[:3]
    frag = build_fragility_map(prompt, output, words, trials=2, max_words=3)

    metrics = {
        "paraphrase_results": para["paraphrases"],
        "fragility_results": frag["fragility"],
    }

    instructions = """
You are issuing a "Prompt Reliability Certificate".

Given:
- paraphrase robustness results (semantic_change_score 1-10 per paraphrase),
- fragility results (mean_change_score and std_change_score per word),

you must output ONLY a JSON object:

{
  "overall_rating": "High" | "Medium" | "Low",
  "numerical_summary": {
    "avg_paraphrase_change": number,
    "avg_fragility": number
  },
  "diagnostics": [
    "short bullet explaining a strength or weakness",
    ...
  ],
  "recommendations": [
    "specific actionable suggestion to improve robustness",
    ...
  ]
}

Heuristics:
- If avg paraphrase change <= 3 and avg fragility std <= 1.5 -> "High".
- If scores around 4-6 -> "Medium".
- Otherwise -> "Low".
"""

    full = instructions + "\n\nRAW METRICS:\n" + json.dumps(metrics, indent=2)
    response = ANALYSIS_MODEL.generate_content(full)
    raw_text = (response.text or "").strip()
    data, err = extract_json_object(raw_text)
    if err or not isinstance(data, dict):
        return {
            "overall_rating": "Unknown",
            "numerical_summary": {},
            "diagnostics": ["Could not parse certificate JSON."],
            "recommendations": [],
            "raw": raw_text,
            "error": err,
        }
    data.setdefault("raw", raw_text)
    return data


# ----------------------------- 12. FLASK ROUTES: CORE ----------------------


@app.route("/health", methods=["GET"])
def health():
    if MODEL_INIT_ERROR:
        return jsonify({"status": "error", "detail": MODEL_INIT_ERROR}), 500
    return jsonify({"status": "ok"}), 200


@app.route("/api/generate", methods=["POST"])
def http_generate():
    data = request.json or {}
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    if len(prompt.split()) > 20:
        return jsonify({"error": "Input prompt is limited to 20 words."}), 400

    try:
        clean = clean_prompt(prompt)
        output = generate_text(clean)
        return jsonify({"output": output})
    except Exception as e:
        app.logger.exception("Error during generation")
        if MODEL_INIT_ERROR:
            return jsonify(
                {"error": f"Model initialization error: {MODEL_INIT_ERROR}"}
            ), 500
        return jsonify({"error": str(e)}), 500


@app.route("/api/analyze", methods=["POST"])
def http_analyze():
    data = request.json or {}
    prompt = data.get("prompt")
    output = data.get("output")
    if not prompt or not output:
        return jsonify({"error": 'Both "prompt" and "output" are required'}), 400

    try:
        analysis_data = analyze_full_report(prompt, output)
        return jsonify(analysis_data)
    except Exception as e:
        app.logger.exception("Error during analysis")
        if MODEL_INIT_ERROR:
            return jsonify(
                {"error": f"Model initialization error: {MODEL_INIT_ERROR}"}
            ), 500
        return jsonify({"error": str(e)}), 500


@app.route("/api/run_experiment", methods=["POST"])
def http_run_experiment():
    data = request.json or {}
    experiment_type = data.get("type")
    original_prompt = data.get("original_prompt")
    original_output = data.get("original_output")
    changes = data.get("changes")

    if experiment_type not in ["ablation", "counterfactual"]:
        return jsonify(
            {"error": 'Invalid or missing "type". Must be "ablation" or "counterfactual".'}
        ), 400

    if not original_prompt or not original_output:
        return jsonify(
            {"error": '"original_prompt" and "original_output" are required.'}
        ), 400

    try:
        if experiment_type == "ablation":
            if not isinstance(changes, str) or not changes.strip():
                return jsonify(
                    {"error": 'For "ablation", "changes" must be a non-empty string like "formal, polite".'}
                ), 400
            new_prompt = ablate_prompt(original_prompt, changes)
        else:
            if not isinstance(changes, str) or not changes.strip():
                return jsonify(
                    {
                        "error": 'For "counterfactual", "changes" must be a non-empty multiline string like:\nformal, informal\npolite, rude'
                    }
                ), 400
            mapping = parse_counterfactual_pairs(changes)
            if not mapping:
                return jsonify(
                    {
                        "error": 'Could not parse any "word, replacement" pairs. Use lines like: formal, informal'
                    }
                ), 400
            new_prompt = apply_counterfactual(original_prompt, mapping)

        new_output = generate_text(new_prompt)
        new_analysis = analyze_full_report(new_prompt, new_output)
        change_data = evaluate_change(
            original_prompt, original_output, new_prompt, new_output
        )

        result = {
            "new_prompt": new_prompt,
            "new_output": new_output,
            "new_analysis": new_analysis,
            "change_data": change_data,
        }
        return jsonify(result)
    except Exception as e:
        app.logger.exception("Error running experiment")
        if MODEL_INIT_ERROR:
            return jsonify(
                {"error": f"Model initialization error: {MODEL_INIT_ERROR}"}
            ), 500
        return jsonify({"error": str(e)}), 500


# ----------------------------- 13. FLASK ROUTES: NOVEL FEATURES ------------


@app.route("/api/causal_heatmap", methods=["POST"])
def http_causal_heatmap():
    data = request.json or {}
    prompt = data.get("prompt")
    output = data.get("output")
    words = data.get("words") or []
    if not prompt or not output:
        return jsonify({"error": '"prompt" and "output" are required'}), 400
    try:
        result = build_causal_heatmap(prompt, output, words)
        return jsonify(result)
    except Exception as e:
        app.logger.exception("Error in causal_heatmap")
        return jsonify({"error": str(e)}), 500


@app.route("/api/fragility_map", methods=["POST"])
def http_fragility_map():
    data = request.json or {}
    prompt = data.get("prompt")
    output = data.get("output")
    words = data.get("words") or []
    trials = int(data.get("trials", 3))
    if not prompt or not output:
        return jsonify({"error": '"prompt" and "output" are required'}), 400
    try:
        result = build_fragility_map(prompt, output, words, trials=trials)
        return jsonify(result)
    except Exception as e:
        app.logger.exception("Error in fragility_map")
        return jsonify({"error": str(e)}), 500


@app.route("/api/paraphrase_robustness", methods=["POST"])
def http_paraphrase_robustness():
    data = request.json or {}
    prompt = data.get("prompt")
    output = data.get("output")
    n = int(data.get("num_paraphrases", 3))
    if not prompt or not output:
        return jsonify({"error": '"prompt" and "output" are required'}), 400
    try:
        result = paraphrase_robustness(prompt, output, num_paraphrases=n)
        return jsonify(result)
    except Exception as e:
        app.logger.exception("Error in paraphrase_robustness")
        return jsonify({"error": str(e)}), 500


@app.route("/api/bias_probe", methods=["POST"])
def http_bias_probe():
    data = request.json or {}
    prompt = data.get("prompt")
    output = data.get("output")
    if not prompt or not output:
        return jsonify({"error": '"prompt" and "output" are required'}), 400
    try:
        result = run_bias_probe(prompt, output)
        return jsonify(result)
    except Exception as e:
        app.logger.exception("Error in bias_probe")
        return jsonify({"error": str(e)}), 500


@app.route("/api/prompt_genome", methods=["POST"])
def http_prompt_genome():
    data = request.json or {}
    prompt = data.get("prompt")
    output = data.get("output")
    if not prompt or not output:
        return jsonify({"error": '"prompt" and "output" are required'}), 400
    try:
        result = build_prompt_genome(prompt, output)
        return jsonify(result)
    except Exception as e:
        app.logger.exception("Error in prompt_genome")
        return jsonify({"error": str(e)}), 500


@app.route("/api/temperature_sensitivity", methods=["POST"])
def http_temperature_sensitivity():
    data = request.json or {}
    prompt = data.get("prompt")
    output = data.get("output")
    temps = data.get("temperatures") or [0.2, 0.7, 1.2]
    if not prompt or not output:
        return jsonify({"error": '"prompt" and "output" are required'}), 400
    try:
        result = temperature_sensitivity(prompt, output, temps)
        return jsonify(result)
    except Exception as e:
        app.logger.exception("Error in temperature_sensitivity")
        return jsonify({"error": str(e)}), 500


@app.route("/api/reliability_certificate", methods=["POST"])
def http_reliability_certificate():
    data = request.json or {}
    prompt = data.get("prompt")
    output = data.get("output")
    if not prompt or not output:
        return jsonify({"error": '"prompt" and "output" are required'}), 400
    try:
        result = build_reliability_certificate(prompt, output)
        return jsonify(result)
    except Exception as e:
        app.logger.exception("Error in reliability_certificate")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
