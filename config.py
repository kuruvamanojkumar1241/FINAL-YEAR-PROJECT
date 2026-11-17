import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'lung-cancer-ai-secret-key')
    if os.environ.get('VERCEL'):
        DATABASE = '/tmp/lung_ai.db'
        UPLOAD_FOLDER = '/tmp/uploads'
    else:
        DATABASE = os.path.join(BASE_DIR, 'database', 'lung_ai.db')
        UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'savemodels')
    DATASET_PATH = os.path.join(BASE_DIR, 'dataset')
    GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
