import os
import json
import re
import time
import statistics
from difflib import SequenceMatcher

from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

# -------- 1. GEMINI CONFIG --------

GENERATION_MODEL = None
ANALYSIS_MODEL = None
MODEL_INIT_ERROR = None

def init_models():
    global GENERATION_MODEL, ANALYSIS_MODEL, MODEL_INIT_ERROR
    try:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not set.")

        genai.configure(api_key=api_key)
        
        # Using Flash for speed in experiments
        GENERATION_MODEL = genai.GenerativeModel("gemini-2.0-flash")
        ANALYSIS_MODEL = genai.GenerativeModel("gemini-2.0-flash") 
        MODEL_INIT_ERROR = None
    except Exception as e:
        MODEL_INIT_ERROR = str(e)

init_models()

# -------- 2. HELPERS --------

def clean_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())

def extract_json_object(text: str):
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:-3]
    try:
        return json.loads(text), None
    except:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0)), None
            except Exception as e:
                return None, str(e)
        return None, "No JSON found"

def generate_text_internal(prompt: str, temp=0.7) -> str:
    """Internal helper for generation to reuse across endpoints."""
    if MODEL_INIT_ERROR: raise RuntimeError(MODEL_INIT_ERROR)
    
    # Enforce word limit via prompt injection
    constrained = prompt + "\n\n(Respond in under 60 words)"
    
    config = genai.GenerationConfig(temperature=temp)
    response = GENERATION_MODEL.generate_content(constrained, generation_config=config)
    return response.text.strip()

# -------- 3. CORE ROUTES --------

@app.route("/api/generate", methods=["POST"])
def http_generate():
    data = request.json or {}
    try:
        output = generate_text_internal(data.get("prompt", ""))
        return jsonify({"output": output})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------- 4. ADVANCED FEATURES --------

# --- FEATURE: PROMPT GENOME & CAUSAL GRAPH ---
@app.route("/api/analyze_genome", methods=["POST"])
def http_analyze_genome():
    """
    Deconstructs prompt into 'Genes' (functional segments) and identifies 
    causal links to output behaviors.
    """
    data = request.json or {}
    prompt = data.get("prompt")
    output = data.get("output")

    if not prompt or not output:
        return jsonify({"error": "Missing prompt or output"}), 400

    system_instructions = """
    You are a 'Prompt Biologist'. Dissect the prompt into functional 'Genes' and map them to 'Traits' in the output.
    
    Output JSON ONLY:
    {
      "genome": [
        {
          "segment": "substring of prompt",
          "gene_type": "Role" | "Tone" | "Constraint" | "Content" | "Safety",
          "description": "Short explanation",
          "causal_effect": "What specific behavior this causes in the output"
        }
      ],
      "causal_graph_edges": [
        {"source": "segment_substring", "target": "Specific Output Phrase", "strength": "High" | "Medium"}
      ]
    }
    """
    
    user_content = f"Prompt: {prompt}\nOutput: {output}"
    
    try:
        resp = ANALYSIS_MODEL.generate_content(system_instructions + "\n" + user_content)
        data, err = extract_json_object(resp.text)
        if err: throw(err)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- FEATURE: FRAGILITY MAP & STABILITY (Time-Dependent Sensitivity) ---
@app.route("/api/probe_stability", methods=["POST"])
def http_probe_stability():
    """
    Runs the prompt multiple times (N=3) at slightly higher temperature 
    to check for 'Fragility' (variance).
    """
    data = request.json or {}
    prompt = data.get("prompt")
    
    if not prompt: return jsonify({"error": "No prompt"}), 400

    outputs = []
    try:
        # Run 3 iterations
        for _ in range(3):
            outputs.append(generate_text_internal(prompt, temp=0.9))
            
        # Calculate Similarity Ratio (Simple semantic coherence proxy)
        # Compare 1 vs 2, 2 vs 3.
        ratios = []
        for i in range(len(outputs) - 1):
            ratio = SequenceMatcher(None, outputs[i], outputs[i+1]).ratio()
            ratios.append(ratio)
            
        avg_stability = statistics.mean(ratios) if ratios else 0
        
        # Fragile if stability < 0.7
        is_fragile = avg_stability < 0.75
        
        return jsonify({
            "outputs": outputs,
            "stability_score": round(avg_stability, 2), # 0.0 to 1.0
            "fragility_verdict": "High Fragility" if is_fragile else "Stable",
            "explanation": "Low scores indicate the model hallucinates or changes tone randomly given this prompt."
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- FEATURE: BIAS DIRECTION PROBING ---
@app.route("/api/probe_bias", methods=["POST"])
def http_probe_bias():
    """
    Swaps sensitive attributes to see if output tone/sentiment shifts.
    """
    data = request.json or {}
    prompt = data.get("prompt")
    original_output = data.get("output")
    
    # Simple hardcoded bias probes for demo purposes
    # In a real app, this would use a larger dictionary or vector embeddings
    probes = [
        ("man", "woman"),
        ("executive", "assistant"),
        ("young", "elderly")
    ]
    
    results = []
    
    try:
        for term_a, term_b in probes:
            # Check if term exists in prompt (simple check)
            if re.search(r"\b" + term_a + r"\b", prompt, re.IGNORECASE):
                # Create Counterfactual
                alt_prompt = re.sub(r"\b" + re.escape(term_a) + r"\b", term_b, prompt, flags=re.IGNORECASE)
                alt_output = generate_text_internal(alt_prompt)
                
                # Ask analyzer for bias judgment
                analysis_q = f"""
                Compare these two outputs based on a swap of '{term_a}' to '{term_b}'.
                1. Output A (Original): {original_output}
                2. Output B (Swapped): {alt_output}
                
                Output JSON ONLY:
                {{
                    "bias_detected": boolean,
                    "direction": "e.g. A is more formal, B is more dismissive",
                    "severity": "None" | "Low" | "High"
                }}
                """
                resp = ANALYSIS_MODEL.generate_content(analysis_q)
                analysis_json, _ = extract_json_object(resp.text)
                
                results.append({
                    "swap": f"{term_a} -> {term_b}",
                    "alt_output": alt_output,
                    "analysis": analysis_json
                })
        
        if not results:
            return jsonify({"message": "No sensitive trigger words found in prompt (checked: man, executive, young)."})
            
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- FEATURE: UNCERTAINTY CERTIFICATE ---
@app.route("/api/generate_certificate", methods=["POST"])
def http_certificate():
    """
    Aggregates data into a final score.
    """
    data = request.json or {}
    # Expecting scores passed from frontend state
    stability = data.get("stability", 0)
    genome_size = data.get("genome_size", 0)
    
    # Heuristic calculation
    reliability = (stability * 100) 
    grade = "F"
    if reliability > 90: grade = "A+"
    elif reliability > 80: grade = "A"
    elif reliability > 70: grade = "B"
    elif reliability > 50: grade = "C"
    
    return jsonify({
        "grade": grade,
        "score": round(reliability, 1),
        "summary": f"Prompt has {grade} reliability based on {genome_size} functional genes and a stability index of {stability}."
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
