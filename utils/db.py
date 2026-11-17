import sqlite3
import os
from config import Config

def get_db_connection():
    conn = sqlite3.connect(Config.DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    if not os.path.exists(os.path.dirname(Config.DATABASE)):
        os.makedirs(os.path.dirname(Config.DATABASE))
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'admin'
        )
    ''')
    
    # Models table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            accuracy REAL,
            memory_usage REAL,
            inference_time REAL,
            file_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            result TEXT NOT NULL,
            confidence REAL,
            voc_percentage REAL,
            explanation TEXT,
            advice TEXT,
            patient_name TEXT DEFAULT 'Anonymous',
            patient_age INTEGER DEFAULT 0,
            patient_gender TEXT DEFAULT 'Not Specified',
            smoking_status TEXT DEFAULT 'Unknown',
            symptoms TEXT DEFAULT 'None',
            prob_benign REAL DEFAULT 0.0,
            prob_malignant REAL DEFAULT 0.0,
            prob_normal REAL DEFAULT 0.0,
            heatmap_filename TEXT DEFAULT '',
            risk_level TEXT DEFAULT 'Moderate',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Auto-migration for existing databases
    existing_cols = [col[1] for col in cursor.execute("PRAGMA table_info(predictions)").fetchall()]
    new_columns = [
        ('patient_name', 'TEXT DEFAULT "Anonymous"'),
        ('patient_age', 'INTEGER DEFAULT 0'),
        ('patient_gender', 'TEXT DEFAULT "Not Specified"'),
        ('smoking_status', 'TEXT DEFAULT "Unknown"'),
        ('symptoms', 'TEXT DEFAULT "None"'),
        ('prob_benign', 'REAL DEFAULT 0.0'),
        ('prob_malignant', 'REAL DEFAULT 0.0'),
        ('prob_normal', 'REAL DEFAULT 0.0'),
        ('heatmap_filename', 'TEXT DEFAULT ""'),
        ('risk_level', 'TEXT DEFAULT "Moderate"')
    ]
    for col_name, col_def in new_columns:
        if col_name not in existing_cols:
            cursor.execute(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_def}")
    
    # Dataset Stats table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS dataset_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            count INTEGER NOT NULL
        )
    ''')

    # Add default admin if not exists
    cursor.execute("SELECT * FROM users WHERE username = 'admin'")
    if not cursor.fetchone():
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", ('admin', 'admin123'))
    
    # Seed Dataset Stats if empty
    cursor.execute("SELECT COUNT(*) FROM dataset_stats")
    if cursor.fetchone()[0] == 0:
        cursor.executemany("INSERT INTO dataset_stats (category, count) VALUES (?, ?)", [
            ('Benign', 120),
            ('Malignant', 561),
            ('Normal', 416)
        ])
    
    # Seed Models if empty
    cursor.execute("SELECT COUNT(*) FROM models")
    if cursor.fetchone()[0] == 0:
        cursor.executemany('''
            INSERT INTO models (name, type, accuracy, memory_usage, inference_time, file_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', [
            ('Ensemble Model V2 (CNN + SNN)', 'CNN+SNN Multi-Layer Fusion', 98.2, 45.2, 12.0, 'savemodels/snn_model.pth', '2026-11-17 11:22:32'),
            ('Ensemble Model V1', 'CNN+SNN Hybrid', 98.2, 45.2, 12.0, 'savemodels/ensemble_v1.h5', '2026-01-31 19:41:57'),
            ('Spiking Neural Network (SNN)', 'Neuromorphic Edge AI', 95.5, 38.4, 9.5, 'savemodels/snn_model.pth', '2026-01-31 20:13:36'),
            ('Custom 2D-CNN', 'Deep Learning ResNet Backbone', 94.8, 52.1, 14.2, 'savemodels/cnn_model.h5', '2026-01-31 19:36:00')
        ])

    # Seed Predictions if empty
    cursor.execute("SELECT COUNT(*) FROM predictions")
    if cursor.fetchone()[0] == 0:
        sample_predictions = [
            (
                'Bengin case (1).jpg', 'Benign', 88.14, 10.99,
                'The screening assessment indicates a Benign pulmonary pattern with 88.14% confidence. The breath VOC level of 10.99% represents mild inflammatory or metabolic variability without malignant signature markers. Post-infectious granulomas or hamartomas are typical. Recommended follow-up includes surveillance CT scan at 6 to 12 months.',
                'No malignant transformation detected. Recommend annual low-dose surveillance CT and routine respiratory wellness.',
                'Eleanor Vance', 45, 'Female', 'Non-smoker', 'Asymptomatic / Routine Screening',
                88.14, 6.55, 6.98, 'heatmap_Bengin case (1).jpg', 'Moderate Risk', '2026-11-17 11:23:24'
            ),
            (
                'Malignant case (2).jpg', 'Malignant', 94.75, 23.40,
                'Critical diagnostic finding: Asymmetric hyperdense cellular cluster detected with 94.75% malignant probability. Breath VOC biomarkers show elevated levels of volatile aldehydes and aromatic hydrocarbons (23.40%). Immediate multidisciplinary tumor board review advised.',
                'Urgent oncology referral required. Immediate High-Resolution CT (HRCT), tissue biopsy, and staging assessment are critical.',
                'Arthur Pendelton', 62, 'Male', 'Active Smoker (30 pack-years)', 'Persistent cough, hemoptysis, unexplained weight loss',
                3.15, 94.75, 2.10, '', 'Critical Risk', '2026-11-16 14:15:10'
            ),
            (
                'Normal case (3).jpg', 'Normal', 97.30, 4.85,
                'Clear bilateral pulmonary parenchyma with regular broncho-vascular arborization. VOC biomarker profile is within standard baseline physiological limits (4.85%). No nodular densities or adenopathy visible.',
                'Biomarker levels and cellular architecture are within standard physiological limits. Maintain active lifestyle and risk prevention.',
                'Sarah Jenkins', 38, 'Female', 'Non-smoker', 'None reported / Occupational physical',
                1.50, 1.20, 97.30, '', 'Low Risk', '2026-11-15 09:42:30'
            ),
            (
                'Bengin case (13).jpg', 'Benign', 91.20, 11.45,
                'Circumscribed calcified nodular density observed in the middle lobe. VOC percentage of 11.45% indicates stable metabolic activity. Consistent with benign tuberculoma or hamartoma.',
                'No active intervention required. Serial CT imaging at 12-month interval to establish volumetric stability.',
                'Marcus Thorne', 52, 'Male', 'Former smoker (quit 8 yrs ago)', 'Mild seasonal dyspnea',
                91.20, 4.30, 4.50, '', 'Moderate Risk', '2026-11-14 16:30:18'
            ),
            (
                'Malignant case (43).jpg', 'Malignant', 92.12, 21.80,
                'Spiculated subpleural lesion identified in upper left lobe with pleural retraction. Markedly elevated VOC signature (21.80%) corroborates cellular hypoxia and anaerobic metabolism.',
                'High suspicion of primary bronchogenic carcinoma. Schedule PET-CT staging and bronchoscopic biopsy.',
                'David Kowalski', 59, 'Male', 'Active Smoker', 'Chronic chest pain, fatigue, finger clubbing',
                4.20, 92.12, 3.68, '', 'Critical Risk', '2026-11-13 11:20:05'
            ),
            (
                'Bengin case (8).jpg', 'Benign', 86.50, 12.20,
                'Smooth bordered sub-centimeter nodule with central fat attenuation. VOC readings are borderline at 12.20%. Characteristic features of benign pulmonary hamartoma.',
                'Annual surveillance recommended. No surgical excision needed at current presentation.',
                'Claire Redfield', 41, 'Female', 'Non-smoker', 'None / Incidental finding',
                86.50, 7.10, 6.40, '', 'Moderate Risk', '2026-11-12 10:15:44'
            ),
            (
                'Normal case (12).jpg', 'Normal', 96.10, 5.10,
                'Normal cardiac silhouette and clear pulmonary fields without parenchymal consolidation or focal infiltrates. Normal breath volatile organic compound concentrations.',
                'Healthy baseline status confirmed. Routine annual health checks recommended.',
                'Liam O’Connor', 29, 'Male', 'Non-smoker', 'None',
                2.10, 1.80, 96.10, '', 'Low Risk', '2026-11-11 15:55:00'
            ),
            (
                'Malignant case (94).jpg', 'Malignant', 89.19, 19.50,
                'Centrilobular cavitary lesion with irregular wall thickening. Elevated VOC biomarker pattern (19.50%) signifies intense local tissue necrosis and inflammatory turnover.',
                'Consult pulmonary oncology immediately. Sputum cytology and contrast-enhanced thorax CT mandatory.',
                'Robert Chen', 67, 'Male', 'Former smoker', 'Shortness of breath, nocturnal diaphoresis',
                6.10, 89.19, 4.71, '', 'High Risk', '2026-11-10 13:08:22'
            )
        ]
        cursor.executemany('''
            INSERT INTO predictions (
                filename, result, confidence, voc_percentage, explanation, advice,
                patient_name, patient_age, patient_gender, smoking_status, symptoms,
                prob_benign, prob_malignant, prob_normal, heatmap_filename, risk_level,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', sample_predictions)
    
    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()
