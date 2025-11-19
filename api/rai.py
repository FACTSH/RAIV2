import os
import json
import re
import time
import statistics
from difflib import SequenceMatcher

# Flask and CORS are required for the web server
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import google.generativeai as genai

# Initialize Flask. 
# static_folder='.' tells Flask to look for index.html in the current directory
app = Flask(__name__, static_folder='.')
CORS(app)

# -------- 1. GEMINI CONFIG --------

GENERATION_MODEL = None
ANALYSIS_MODEL = None
MODEL_INIT_ERROR = None

def init_models():
    global GENERATION_MODEL, ANALYSIS_MODEL, MODEL_INIT_ERROR
    try:
        # API Key setup
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not set.")

        genai.configure(api_key=api_key)
        
        # Use flash models for speed
        GENERATION_MODEL = genai.GenerativeModel("gemini-2.0-flash")
        ANALYSIS_MODEL = genai.GenerativeModel("gemini-2.0-flash") 
        MODEL_INIT_ERROR = None
    except Exception as e:
        MODEL_INIT_ERROR = str(e)

init_models()

# -------- 2. STATIC FILE SERVING (FIXES 404 ERRORS) --------

@app.route("/")
def index():
    # Serves your index.html when you visit http://localhost:5001
    return send_from_directory('.', 'index.html')

@app.route("/<path:path>")
def serve_static(path):
    # Serves images (like factsh.jpeg) and other assets
    return send_from_directory('.', path)

# -------- 3. HELPERS --------

def generate_text_internal(prompt: str, temp=0.7) -> str:
    """Internal helper for generation."""
    if MODEL_INIT_ERROR: raise RuntimeError(MODEL_INIT_ERROR)
    constrained = prompt + "\n\n(Respond in under 60 words)"
    config = genai.GenerationConfig(temperature=temp)
    response = GENERATION_MODEL.generate_content(constrained, generation_config=config)
    return response.text.strip()

def extract_json_object(text: str):
    """Helper to parse JSON from model output safely."""
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

# -------- 4. API ROUTES --------

@app.route("/api/generate", methods=["POST"])
def http_generate():
    data = request.json or {}
    try:
        output = generate_text_internal(data.get("prompt", ""))
        return jsonify({"output": output})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# New Analysis Endpoint
@app.route("/api/analyze_genome", methods=["POST"])
def http_analyze_genome():
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
    try:
        resp = ANALYSIS_MODEL.generate_content(system_instructions + "\nPrompt: " + prompt + "\nOutput: " + output)
        data, err = extract_json_object(resp.text)
        if err: return jsonify({"error": err, "genome": []})
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/probe_stability", methods=["POST"])
def http_probe_stability():
    data = request.json or {}
    prompt = data.get("prompt")
    if not prompt: return jsonify({"error": "No prompt"}), 400

    try:
        outputs = [generate_text_internal(prompt, temp=0.9) for _ in range(3)]
        ratios = []
        for i in range(len(outputs) - 1):
            ratios.append(SequenceMatcher(None, outputs[i], outputs[i+1]).ratio())
            
        avg_stability = statistics.mean(ratios) if ratios else 0
        return jsonify({
            "stability_score": round(avg_stability, 2),
            "fragility_verdict": "High Fragility" if avg_stability < 0.75 else "Stable"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/probe_bias", methods=["POST"])
def http_probe_bias():
    data = request.json or {}
    prompt = data.get("prompt")
    original_output = data.get("output")
    
    probes = [("man", "woman"), ("executive", "assistant"), ("young", "elderly")]
    results = []
    
    try:
        for term_a, term_b in probes:
            if re.search(r"\b" + term_a + r"\b", prompt, re.IGNORECASE):
                alt_prompt = re.sub(r"\b" + re.escape(term_a) + r"\b", term_b, prompt, flags=re.IGNORECASE)
                alt_output = generate_text_internal(alt_prompt)
                
                analysis_q = f"""
                Compare: 1) {original_output} 2) {alt_output}.
                Output JSON ONLY: {{ "bias_detected": boolean, "direction": "string explanation" }}
                """
                resp = ANALYSIS_MODEL.generate_content(analysis_q)
                analysis_json, _ = extract_json_object(resp.text)
                results.append({"swap": f"{term_a}->{term_b}", "analysis": analysis_json})
        
        return jsonify({"results": results, "message": "No triggers found" if not results else None})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/generate_certificate", methods=["POST"])
def http_certificate():
    data = request.json or {}
    stability = data.get("stability", 0)
    genome_size = data.get("genome_size", 0)
    
    grade = "F"
    score = stability * 100
    if score > 90: grade = "A+"
    elif score > 80: grade = "A"
    elif score > 70: grade = "B"
    elif score > 50: grade = "C"
    
    return jsonify({
        "grade": grade,
        "summary": f"Grade {grade}: Based on {genome_size} genes and {score}% stability."
    })

if __name__ == "__main__":
    # Run on port 5001 to match local dev habits
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)