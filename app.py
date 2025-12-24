import os
import webbrowser
import threading
from flask import Flask, render_template, request, jsonify
from threading import Timer
from huggingface_hub import hf_hub_download

# --- IMPORT ANALYZER CLASS (But don't init it yet) ---
from analyzer import RosterAnalyzer

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_DIR = os.path.join(os.getcwd(), "models")
MODEL_FILENAME = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
REPO_ID = "bartowski/Llama-3.2-1B-Instruct-GGUF"

# Global Variables
analyzer = None
is_ready = False
setup_started = False

def background_setup():
    """Downloads model and initializes AI in the background"""
    global analyzer, is_ready
    
    # 1. Download if missing
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        try:
            hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME, local_dir=MODEL_DIR)
        except Exception as e:
            print(f"Download Error: {e}")

    # 2. Load the AI into RAM
    print("Loading AI...")
    analyzer = RosterAnalyzer()
    is_ready = True
    print("System Ready.")

@app.route('/')
def index():
    # If AI isn't ready, show the Loading Page
    if not is_ready:
        return render_template('loading.html')
    
    # If AI is ready, show the Main App
    return render_template('index.html')

@app.route('/start-setup')
def start_setup():
    global setup_started
    if not setup_started and not is_ready:
        setup_started = True
        threading.Thread(target=background_setup, daemon=True).start()
    return "Started"

@app.route('/check-status')
def check_status():
    return jsonify(ready=is_ready)

@app.route('/analyze', methods=['POST'])
def analyze():
    # Helper route for the form submission
    if not is_ready: return render_template('loading.html')
    
    try:
        file = request.files['file']
        date = request.form['date']
        shift = request.form['shift']
        
        if file.filename == '': return render_template('index.html', error="No file selected")
        
        filepath = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(filepath)
        
        results = analyzer.get_shift_results(filepath, date, shift)
        return render_template('index.html', results=results)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    # Open Browser
    Timer(1, lambda: webbrowser.open_new("http://127.0.0.1:5000/")).start()
    app.run(port=5000, debug=False)