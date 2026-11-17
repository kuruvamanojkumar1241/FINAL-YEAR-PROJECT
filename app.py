import os
import io
import csv
import random
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory, Response
from config import Config
from utils.db import get_db_connection, init_db
from auth import auth_bp
from groq_ai.analysis import get_breath_analysis_explanation
from utils.heatmap import generate_saliency_heatmap

app = Flask(__name__)
app.config.from_object(Config)

# Ensure database and upload folders exist
init_db()
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Register Blueprints
app.register_blueprint(auth_bp)

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    conn = get_db_connection()
    dataset_stats = conn.execute('SELECT category, count FROM dataset_stats').fetchall()
    model_stats = conn.execute('SELECT * FROM models ORDER BY created_at DESC LIMIT 5').fetchall()
    recent_predictions = conn.execute('SELECT * FROM predictions ORDER BY created_at DESC LIMIT 5').fetchall()
    conn.close()
    
    if not dataset_stats:
        dataset_stats = [
            {'category': 'Benign', 'count': 120},
            {'category': 'Malignant', 'count': 85},
            {'category': 'Normal', 'count': 150}
        ]
    
    return render_template('dashboard.html', 
                           dataset_stats=dataset_stats, 
                           model_stats=model_stats,
                           recent_predictions=recent_predictions)

@app.route('/dataset')
def view_dataset():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    view_all = request.args.get('view_all')
    categories = ['Bengin', 'Malignant', 'Normal']
    dataset_path = Config.DATASET_PATH
    
    data = {}
    for cat in categories:
        if view_all and view_all != cat:
            continue
            
        path = os.path.join(dataset_path, cat + ' cases')
        if os.path.exists(path):
            files = os.listdir(path)
            if view_all == cat:
                data[cat] = files
            else:
                data[cat] = files[:10]
        else:
            data[cat] = []
            
    return render_template('dataset.html', data=data, view_all=view_all)

@app.route('/dataset/<path:filename>')
def serve_dataset(filename):
    return send_from_directory(Config.DATASET_PATH, filename)

@app.route('/train', methods=['GET', 'POST'])
def train_models():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    if request.method == 'POST':
        from models.train import train_ensemble
        try:
            train_ensemble()
            flash('Ensemble models trained and optimized successfully!', 'success')
        except Exception as e:
            flash(f'Error during training: {str(e)}', 'error')
        return redirect(url_for('train_models'))
        
    return render_template('train.html')

@app.route('/test', methods=['GET', 'POST'])
def test_image():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Clinical patient metadata
            patient_name = request.form.get('patient_name', '').strip() or 'Anonymous Patient'
            patient_age = request.form.get('patient_age', type=int) or 45
            patient_gender = request.form.get('patient_gender', 'Not Specified')
            smoking_status = request.form.get('smoking_status', 'Non-smoker')
            symptoms_list = request.form.getlist('symptoms')
            symptoms_str = ', '.join(symptoms_list) if symptoms_list else 'Asymptomatic / None reported'
            
            # Run prediction
            from models.inference import predict_lung_cancer
            result, confidence, voc_percentage, probs, risk_level = predict_lung_cancer(
                filepath, smoking_status=smoking_status, symptoms=symptoms_str
            )
            
            # Generate Explainable AI (XAI) attention heatmap
            heatmap_filename = generate_saliency_heatmap(filepath)
            
            # Clinical explanation from Groq AI or medical knowledge engine
            explanation = get_breath_analysis_explanation(
                result=result,
                confidence=confidence,
                voc_percentage=voc_percentage,
                patient_name=patient_name,
                smoking_status=smoking_status,
                symptoms=symptoms_str,
                risk_level=risk_level
            )
            
            advice_map = {
                'Benign': 'No malignant transformation detected. Recommend annual low-dose surveillance CT and routine respiratory wellness.',
                'Malignant': 'Urgent oncology referral required. Immediate High-Resolution CT (HRCT), tissue biopsy, and staging assessment are critical.',
                'Normal': 'Biomarker levels and cellular architecture are within standard physiological limits. Maintain active lifestyle and risk prevention.'
            }
            advice = advice_map.get(result, 'Consult pulmonary specialist for comprehensive evaluation.')
            
            # Save into Database
            conn = get_db_connection()
            conn.execute('''
                INSERT INTO predictions (
                    filename, result, confidence, voc_percentage, explanation, advice,
                    patient_name, patient_age, patient_gender, smoking_status, symptoms,
                    prob_benign, prob_malignant, prob_normal, heatmap_filename, risk_level
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                filename, result, confidence, voc_percentage, explanation, advice,
                patient_name, patient_age, patient_gender, smoking_status, symptoms_str,
                probs.get('Benign', 0.0), probs.get('Malignant', 0.0), probs.get('Normal', 0.0),
                heatmap_filename, risk_level
            ))
            conn.commit()
            conn.close()
            
            return redirect(url_for('prediction_result', filename=filename))
            
    return render_template('test.html')

@app.route('/result/<filename>')
def prediction_result(filename):
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    conn = get_db_connection()
    result_data = conn.execute('SELECT * FROM predictions WHERE filename = ? ORDER BY created_at DESC', (filename,)).fetchone()
    conn.close()
    
    if not result_data:
        flash('Result not found', 'error')
        return redirect(url_for('test_image'))
    
    return render_template('result.html', result=result_data)

@app.route('/report/<filename>')
def diagnostic_report(filename):
    """
    Renders an official, printable A4 Medical Diagnostic Report for doctors & patients.
    """
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    conn = get_db_connection()
    result_data = conn.execute('SELECT * FROM predictions WHERE filename = ? ORDER BY created_at DESC', (filename,)).fetchone()
    conn.close()
    
    if not result_data:
        flash('Diagnostic report not found', 'error')
        return redirect(url_for('history'))
        
    return render_template('report.html', result=result_data)

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    status_filter = request.args.get('status', '').strip()
    search_query = request.args.get('q', '').strip()
    
    conn = get_db_connection()
    query = 'SELECT * FROM predictions WHERE 1=1'
    params = []
    
    if status_filter in ['Benign', 'Malignant', 'Normal']:
        query += ' AND result = ?'
        params.append(status_filter)
        
    if search_query:
        query += ' AND (patient_name LIKE ? OR filename LIKE ?)'
        wildcard = f"%{search_query}%"
        params.extend([wildcard, wildcard])
        
    query += ' ORDER BY created_at DESC'
    predictions = conn.execute(query, params).fetchall()
    conn.close()
    
    return render_template('history.html', 
                           predictions=predictions, 
                           current_status=status_filter, 
                           search_query=search_query)

@app.route('/export_history')
def export_history():
    """
    Exports diagnostic history to a downloadable CSV spreadsheet.
    """
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
        
    conn = get_db_connection()
    records = conn.execute('SELECT * FROM predictions ORDER BY created_at DESC').fetchall()
    conn.close()
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        'ID', 'Date', 'Patient Name', 'Age', 'Gender', 'Smoking Status',
        'Symptoms', 'Result', 'Confidence (%)', 'VOC (%)', 'Risk Level',
        'Benign %', 'Malignant %', 'Normal %', 'Filename'
    ])
    
    for r in records:
        writer.writerow([
            r['id'], r['created_at'],
            r['patient_name'] or 'Anonymous',
            r['patient_age'] or 'N/A',
            r['patient_gender'] or 'N/A',
            r['smoking_status'] or 'Unknown',
            r['symptoms'] or 'None',
            r['result'],
            r['confidence'],
            r['voc_percentage'],
            r['risk_level'] or 'Moderate',
            r['prob_benign'] if 'prob_benign' in r.keys() else '',
            r['prob_malignant'] if 'prob_malignant' in r.keys() else '',
            r['prob_normal'] if 'prob_normal' in r.keys() else '',
            r['filename']
        ])
        
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=lung_cancer_diagnostic_history.csv"}
    )

@app.route('/future-scope')
def future_scope():
    """
    Renders the Next-Generation Innovations & Future Scope interactive showcase.
    """
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    return render_template('future_scope.html')

if __name__ == '__main__':
    app.run(debug=False)
