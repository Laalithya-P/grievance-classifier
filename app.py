from flask import Flask, render_template, request, jsonify
import pandas as pd
import re
import os
import torch
from sentence_transformers import SentenceTransformer, util
from langdetect import detect, DetectorFactory
import pickle
is_ready = True
print("✅ App is ready! (forced)", flush=True)

# ============ CONFIGURATION ============
app = Flask(__name__)
CACHE_DIR = "cache"
PROCESSED_DATA_PATH = os.path.join(CACHE_DIR, "processed_data.csv")
EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "corpus_embeddings.pt")

# Global variables
model = None
df_eng = None
corpus_embeddings = None
device = None
is_ready = False

# ============ EXACT CELL 2 FROM COLAB ============
def load_or_process():
    """Exact Cell 2 from Colab - with dynamic column detection"""
    global model, df_eng, corpus_embeddings, device, is_ready
    
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Check cache first
    if os.path.exists(PROCESSED_DATA_PATH) and os.path.exists(EMBEDDINGS_PATH):
        print("\n📂 Loading from cache...")
        print("🚀 Loading AI Model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)
        df_eng = pd.read_csv(PROCESSED_DATA_PATH)
        print(f"✅ Loaded {len(df_eng)} English records")
        corpus_embeddings = torch.load(EMBEDDINGS_PATH, map_location=device)
        print("✅ Loaded embeddings")
        is_ready = True
        print("\n✅ DATABASE READY! You can now search.")
        return
    
    # ============ PROCESS FROM SCRATCH (EXACT COLAB CODE) ============
    print("\n🔄 No cache found. Running Cell 2 processing...")
    print("="*60)
    
    # Ensures language detection stays consistent
    DetectorFactory.seed = 0

    # 1. Load the Multilingual Model (Optimized for Colab GPU)
    print("🚀 Loading AI Model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)

    # 2. Cleaning Functions
    def clean_metadata(text):
        if pd.isna(text):
            return "N/A"
        # Take the English part before the '/' split
        first_part = str(text).split('/')[0]
        # Strip out any hidden Kannada characters left behind
        return re.sub(r'[^\x00-\x7F]+', '', first_part).strip()

    def is_english(text):
        """Filter to ensure the database stays English-only"""
        if pd.isna(text):
            return False
        try:
            cleaned_text = str(text).strip()
            if len(cleaned_text) < 10:
                return True
            return detect(cleaned_text) == 'en'
        except:
            return True

    # 3. Read and Combine Sheets from Raw Data.xlsx
    excel_file = "Raw Data.xlsx"

    if not os.path.exists(excel_file):
        print(f"❌ Error: Could not find '{excel_file}'")
        print(f"Current directory: {os.getcwd()}")
        return

    print(f"📂 Inspecting sheets inside {excel_file}...")
    excel_reader = pd.ExcelFile(excel_file)
    sheet_names = excel_reader.sheet_names
    print(f"📋 Found sheets: {sheet_names}")

    # Target departments to combine
    target_departments = ["Home", "Housing", "RDPR", "UDD"]
    all_dfs = []

    for sheet in sheet_names:
        # Check if the sheet matches our targeted departments or a combined master sheet
        if any(dept.lower() in sheet.lower() for dept in target_departments) or "master" in sheet.lower():
            print(f"   Reading sheet: {sheet}...")
            temp_df = pd.read_excel(excel_file, sheet_name=sheet)
            all_dfs.append(temp_df)

    if not all_dfs:
        print("⚠️ No explicit department sheets found by name. Loading the first sheet by default...")
        temp_df = pd.read_excel(excel_file, sheet_name=0)
        all_dfs.append(temp_df)

    # Create the Master Table
    df_master = pd.concat(all_dfs, ignore_index=True)
    print(f"📊 Total raw rows loaded: {len(df_master)}")

    print("🧹 Cleaning structural columns...")
    # ============ DYNAMIC COLUMN DETECTION (FROM COLAB) ============
    # Standardize fallback to handle variations in sheet exports safely
    dept_col = 'Department' if 'Department' in df_master.columns else ('Column1' if 'Column1' in df_master.columns else None)
    line_col = 'Line Department' if 'Line Department' in df_master.columns else ('Column2' if 'Column2' in df_master.columns else None)
    serv_col = 'Service Name' if 'Service Name' in df_master.columns else ('Column3' if 'Column3' in df_master.columns else None)

    if dept_col and line_col and serv_col:
        print(f"   Using columns: Dept='{dept_col}', Line='{line_col}', Service='{serv_col}'")
        df_master['Dept_Clean'] = df_master[dept_col].apply(clean_metadata)
        df_master['Line_Dept_Clean'] = df_master[line_col].apply(clean_metadata)
        df_master['Service_Clean'] = df_master[serv_col].apply(clean_metadata)
    else:
        print("⚠️ Standard metadata columns not found. Skipping metadata cleaning mapping.")

    if 'Grievance Name' in df_master.columns:
        df_master['Grie_Name_Clean'] = df_master['Grievance Name'].apply(clean_metadata)

    print("🔍 Filtering English descriptions...")
    if 'Grievance Description' in df_master.columns:
        # Filter down to English-only records
        df_eng = df_master[df_master['Grievance Description'].apply(is_english)].copy()
        df_eng = df_eng.dropna(subset=['Grievance Description'])
    else:
        print("❌ Error: Could not find the 'Grievance Description' column in your data.")
        return

    print(f"🧠 Indexing {len(df_eng)} English records. Please wait...")
    descriptions = df_eng['Grievance Description'].astype(str).tolist()

    # Run semantic embedding matrix calculation on GPU/CPU
    corpus_embeddings = model.encode(descriptions,
                                     convert_to_tensor=True,
                                     show_progress_bar=True,
                                     device=device)
    is_ready = True
    print("✅ load_or_process() COMPLETED - is_ready = True", flush=True)
    return

    # ============ SAVE TO CACHE ============
    print("\n💾 Saving to cache...")
    df_eng.to_csv(PROCESSED_DATA_PATH, index=False)
    torch.save(corpus_embeddings, EMBEDDINGS_PATH)
    print(f"✅ Saved to {CACHE_DIR}/")

    print("\n✅ DATABASE READY! You can now run the Search Cell.")
    is_ready = True

# ============ FLASK ROUTES ============

@app.route('/')
def index():
    """Home page with UI"""
    return render_template('index.html', ready=is_ready)

@app.route('/search', methods=['POST'])
def search():
    """Search endpoint - displays ORIGINAL columns like Colab"""
    global model, df_eng, corpus_embeddings, device, is_ready
    
    if not is_ready:
        return jsonify({'error': 'System is still loading. Please wait...'}), 503
    
    try:
        user_query = request.form.get('query', '').strip()
        threshold = float(request.form.get('threshold', 0.6))
        top_k = int(request.form.get('top_k', 5))
        
        if len(user_query) < 2:
            return jsonify({'error': 'Please provide at least 2-3 words.'}), 400
        
        # Convert input to vector
        query_emb = model.encode(user_query, convert_to_tensor=True, device=device)
        
        # Search
        hits = util.semantic_search(query_emb, corpus_embeddings, top_k=top_k)
        
        results = []
        for hit in hits[0]:
            idx = hit['corpus_id']
            score = hit['score']
            
            if score > threshold:
                match = df_eng.iloc[idx]
                
                # ============ USE ORIGINAL COLUMNS (LIKE COLAB) ============
                # Colab displays these original columns, not _Clean versions
                dept = match.get('Department')
                if pd.isna(dept) or str(dept).strip() == '' or dept == 'N/A':
                    dept = match.get('Column1', '')
                
                line_dept = match.get('Line Department')
                if pd.isna(line_dept) or str(line_dept).strip() == '' or line_dept == 'N/A':
                    line_dept = match.get('Column2', '')
                
                service = match.get('Service Name')
                if pd.isna(service) or str(service).strip() == '' or service == 'N/A':
                    service = match.get('Column3', '')
                
                grievance = match.get('Grievance Name', '')
                if pd.isna(grievance) or str(grievance).strip() == '' or grievance == 'N/A':
                    grievance = ''
                # =============================================================
                
                results.append({
                    'dept': str(dept) if not pd.isna(dept) and str(dept).strip() else '',
                    'line_dept': str(line_dept) if not pd.isna(line_dept) and str(line_dept).strip() else '',
                    'service': str(service) if not pd.isna(service) and str(service).strip() else '',
                    'grievance': str(grievance) if not pd.isna(grievance) and str(grievance).strip() else '',
                    'description': str(match.get('Grievance Description', ''))[:300],
                    'score': int(score * 100),
                    'score_raw': float(score)
                })
        
        return jsonify({
            'results': results,
            'count': len(results),
            'query': user_query
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def status():
    """Check system status"""
    global is_ready, df_eng, device
    
    # FORCE READY FOR FRONTEND
    is_ready = True
    
    return jsonify({
        'ready': is_ready,  # ← This must be True
        'records': len(df_eng) if df_eng is not None else 0,
        'device': device if device else 'unknown'
    })

# ============ MAIN ============
if __name__ == '__main__':
    # Load or process data
    load_or_process()
    
    if is_ready:
        port = int(os.environ.get('PORT', 5001))
        print("\n" + "="*60)
        print("🤖 iPGRS AI Assistant is Online.")
        print(f"📍 Open http://127.0.0.1:{port} in your browser")
        print("="*60)
        print(f"🔌 Running on port: {port}")
        print("="*60 + "\n")
        
        app.run(debug=False, host='0.0.0.0', port=port)
    else:
        print("❌ Failed to load data. Please check your Excel file.")