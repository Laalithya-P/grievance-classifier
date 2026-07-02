from flask import Flask, render_template, request, jsonify
import pandas as pd
import re
import os
import torch
from sentence_transformers import SentenceTransformer, util
from langdetect import detect, DetectorFactory
import sys
import logging

# ============ LOGGING ============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
#---------------------
is_ready = True  # ← ADD THIS LINE
logger.info("⚠️ FORCED is_ready = True for testing")

# ============ GLOBAL VARIABLES ============
app = Flask(__name__)
CACHE_DIR = "cache"
PROCESSED_DATA_PATH = os.path.join(CACHE_DIR, "processed_data.csv")
EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "corpus_embeddings.pt")

# Global variables - these will be set once
model = None
df_eng = None
corpus_embeddings = None
device = None
is_ready = False

# ============ LOAD MODEL ONCE ============
def load_model():
    """Load the model once and reuse it"""
    global model, device
    
    if model is not None:
        logger.info("✅ Model already loaded")
        return model
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"📱 Device: {device}")
    
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        model = model.half()  # FP16 for memory efficiency
        logger.info("✅ Model loaded with FP16!")
        return model
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        model = None
        return None

# ============ LOAD OR PROCESS DATA ============
def load_or_process():
    """Load from cache or process from scratch"""
    global model, df_eng, corpus_embeddings, device, is_ready
    
    logger.info("="*60)
    logger.info("🚀 LOAD_OR_PROCESS STARTED")
    logger.info("="*60)
    
    # Load model first
    load_model()
    
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # ============ CHECK CACHE ============
    cache_exists = os.path.exists(PROCESSED_DATA_PATH) and os.path.exists(EMBEDDINGS_PATH)
    logger.info(f"🔍 Cache exists: {cache_exists}")
    
    if cache_exists:
        logger.info("📂 Loading from cache...")
        try:
            df_eng = pd.read_csv(PROCESSED_DATA_PATH)
            corpus_embeddings = torch.load(EMBEDDINGS_PATH, map_location=device)
            logger.info(f"✅ Loaded {len(df_eng)} English records from cache")
            logger.info("✅ Loaded embeddings")
            is_ready = True
            logger.info("✅ DATABASE READY! You can now search.")
            return
        except Exception as e:
            logger.error(f"❌ Error loading cache: {e}")
            logger.info("🔄 Falling back to processing from scratch...")
    
    # ============ PROCESS FROM SCRATCH ============
    logger.info("🔄 Processing from scratch...")
    logger.info("="*60)
    
    DetectorFactory.seed = 0

    # 1. Cleaning Functions
    def clean_metadata(text):
        if pd.isna(text):
            return "N/A"
        first_part = str(text).split('/')[0]
        return re.sub(r'[^\x00-\x7F]+', '', first_part).strip()

    def is_english(text):
        if pd.isna(text):
            return False
        try:
            cleaned_text = str(text).strip()
            if len(cleaned_text) < 10:
                return True
            return detect(cleaned_text) == 'en'
        except:
            return True

    # 2. Read and Combine Sheets
    excel_file = "Raw Data.xlsx"

    if not os.path.exists(excel_file):
        logger.error(f"❌ Error: Could not find '{excel_file}'")
        logger.info(f"Current directory: {os.getcwd()}")
        is_ready = False
        return

    logger.info(f"📂 Inspecting sheets inside {excel_file}...")
    excel_reader = pd.ExcelFile(excel_file)
    sheet_names = excel_reader.sheet_names
    logger.info(f"📋 Found sheets: {sheet_names}")

    target_departments = ["Home", "Housing", "RDPR", "UDD"]
    all_dfs = []

    # ============ SAMPLE 3,000 ROWS FROM EACH SHEET ============
    for sheet in sheet_names:
        if any(dept.lower() in sheet.lower() for dept in target_departments) or "master" in sheet.lower():
            logger.info(f"   Reading sheet: {sheet}...")
            temp_df = pd.read_excel(excel_file, sheet_name=sheet)
            
            sample_size = 3000
            if len(temp_df) > sample_size:
                temp_df = temp_df.sample(n=sample_size, random_state=42)
                logger.info(f"      ✅ Sampled {sample_size} rows from {sheet}")
            else:
                logger.info(f"      ✅ Using all {len(temp_df)} rows from {sheet}")
            
            all_dfs.append(temp_df)

    if not all_dfs:
        logger.warning("⚠️ No matching sheets found. Reading ALL sheets...")
        for sheet in sheet_names:
            logger.info(f"   Reading sheet: {sheet}...")
            temp_df = pd.read_excel(excel_file, sheet_name=sheet)
            sample_size = 3000
            if len(temp_df) > sample_size:
                temp_df = temp_df.sample(n=sample_size, random_state=42)
                logger.info(f"      ✅ Sampled {sample_size} rows from {sheet}")
            all_dfs.append(temp_df)

    # Create the Master Table
    df_master = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"📊 Total rows loaded: {len(df_master)}")
    
    # Show department distribution
    if 'Department' in df_master.columns:
        logger.info("📊 Department distribution:")
        logger.info(f"\n{df_master['Department'].value_counts()}")

    # 3. Clean Columns
    logger.info("🧹 Cleaning structural columns...")
    dept_col = 'Department' if 'Department' in df_master.columns else ('Column1' if 'Column1' in df_master.columns else None)
    line_col = 'Line Department' if 'Line Department' in df_master.columns else ('Column2' if 'Column2' in df_master.columns else None)
    serv_col = 'Service Name' if 'Service Name' in df_master.columns else ('Column3' if 'Column3' in df_master.columns else None)

    if dept_col and line_col and serv_col:
        logger.info(f"   Using columns: Dept='{dept_col}', Line='{line_col}', Service='{serv_col}'")
        df_master['Dept_Clean'] = df_master[dept_col].apply(clean_metadata)
        df_master['Line_Dept_Clean'] = df_master[line_col].apply(clean_metadata)
        df_master['Service_Clean'] = df_master[serv_col].apply(clean_metadata)
    else:
        logger.warning("⚠️ Standard metadata columns not found. Skipping metadata cleaning mapping.")

    if 'Grievance Name' in df_master.columns:
        df_master['Grie_Name_Clean'] = df_master['Grievance Name'].apply(clean_metadata)

    # 4. Filter English
    logger.info("🔍 Filtering English descriptions...")
    if 'Grievance Description' in df_master.columns:
        df_eng = df_master[df_master['Grievance Description'].apply(is_english)].copy()
        df_eng = df_eng.dropna(subset=['Grievance Description'])
    else:
        logger.error("❌ Error: Could not find the 'Grievance Description' column in your data.")
        is_ready = False
        return

    logger.info(f"✅ After English filter: {len(df_eng)} records")

    # 5. Create Embeddings
    logger.info(f"🧠 Creating embeddings for {len(df_eng)} records. Please wait...")
    descriptions = df_eng['Grievance Description'].astype(str).tolist()

    corpus_embeddings = model.encode(descriptions,
                                     convert_to_tensor=True,
                                     show_progress_bar=True,
                                     device=device,
                                     batch_size=32)

    # 6. Save to Cache
    logger.info("💾 Saving to cache...")
    os.makedirs(CACHE_DIR, exist_ok=True)
    df_eng.to_csv(PROCESSED_DATA_PATH, index=False)
    torch.save(corpus_embeddings, EMBEDDINGS_PATH)
    logger.info(f"✅ Saved to {CACHE_DIR}/")

    is_ready = True
    logger.info("✅ DATABASE READY! You can now search.")
    logger.info("✅ load_or_process() COMPLETED - is_ready = True")

# ============ FLASK ROUTES ============

@app.route('/')
def index():
    """Home page with UI"""
    return render_template('index.html', ready=is_ready)

@app.route('/search', methods=['POST'])
def search():
    """Search endpoint"""
    global model, df_eng, corpus_embeddings, device, is_ready
    
    if not is_ready:
        return jsonify({'error': 'System is still loading. Please wait...'}), 503
    
    if model is None or df_eng is None or corpus_embeddings is None:
        return jsonify({'error': 'Model or data not loaded.'}), 503
    
    try:
        user_query = request.form.get('query', '').strip()
        threshold = float(request.form.get('threshold', 0.6))
        top_k = int(request.form.get('top_k', 5))
        
        if len(user_query) < 2:
            return jsonify({'error': 'Please provide at least 2-3 words.'}), 400
        
        query_emb = model.encode(user_query, convert_to_tensor=True, device=device)
        hits = util.semantic_search(query_emb, corpus_embeddings, top_k=top_k)
        
        results = []
        for hit in hits[0]:
            idx = hit['corpus_id']
            score = hit['score']
            
            if score > threshold:
                match = df_eng.iloc[idx]
                
                dept = match.get('Department', '')
                if pd.isna(dept) or str(dept).strip() == '' or dept == 'N/A':
                    dept = match.get('Column1', '')
                
                line_dept = match.get('Line Department', '')
                if pd.isna(line_dept) or str(line_dept).strip() == '' or line_dept == 'N/A':
                    line_dept = match.get('Column2', '')
                
                service = match.get('Service Name', '')
                if pd.isna(service) or str(service).strip() == '' or service == 'N/A':
                    service = match.get('Column3', '')
                
                grievance = match.get('Grievance Name', '')
                if pd.isna(grievance) or str(grievance).strip() == '' or grievance == 'N/A':
                    grievance = ''
                
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
        logger.error(f"❌ Search error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def status():
    """Check system status"""
    global is_ready, df_eng, device
    return jsonify({
        'ready': is_ready,
        'records': len(df_eng) if df_eng is not None else 0,
        'device': device if device else 'unknown'
    })

# ============ MAIN ============
if __name__ == '__main__':
    load_or_process()
    
    if is_ready:
        port = int(os.environ.get('PORT', 5001))
        logger.info("="*60)
        logger.info("🤖 iPGRS AI Assistant is Online.")
        logger.info(f"📍 Open http://127.0.0.1:{port} in your browser")
        logger.info("="*60)
        logger.info(f"🔌 Running on port: {port}")
        logger.info("="*60)
        
        app.run(debug=False, host='0.0.0.0', port=port)
    else:
        logger.error("❌ Failed to load data. Please check your Excel file.")