from flask import Flask, request, jsonify
from model import load_model, predict_sales
import os
import shutil
import psutil
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
model_path = os.getenv('MODEL_PATH', 'model/lgb_model.pkl')

def load_model_with_status():
    try:
        model = load_model(model_path)
        return model, True
    except Exception as e:
        return None, False

def check_memory_usage():
    memory = psutil.virtual_memory()
    # Ensure there's at least 1GB of available memory
    return memory.available > 1 * 1024 * 1024 * 1024

def check_disk_space():
    total, used, free = shutil.disk_usage("/")
    # Ensure there's at least 1GB of free space
    return free > 1 * 1024 * 1024 * 1024

def check_environment_variables():
    # Check for any critical environment variables
    required_vars = ["MODEL_PATH", "DATA_PATH"]
    return all(var in os.environ for var in required_vars)

# Try loading the model initially and set the status
model, model_loaded_successfully = load_model_with_status()

@app.route('/predict', methods=['POST'])
def predict():
    global model
    data = request.get_json()
    date = data['date']
    store = data['store']
    item = data['item']
    sales = predict_sales(date, store, item, model)
    return jsonify({'sales': sales[0]})

@app.route('/status', methods=['GET'])
def status():
    status_checks = {
        "model_file_exists": os.path.exists(model_path),
        "model_loaded_successfully": model_loaded_successfully,
        "disk_space_ok": check_disk_space(),
        "memory_usage_ok": check_memory_usage(),
        "environment_variables_ok": check_environment_variables()
    }
    
    overall_status = all(status_checks.values())
    
    return jsonify({
        'status': 'success' if overall_status else 'failure',
        'checks': status_checks
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
