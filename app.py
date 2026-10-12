# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:40:41 2020

@author: win10
"""

# 1. Library imports
import uvicorn
from fastapi import FastAPI
from BankNotes import BankNote
import numpy as np
import pickle
import pandas as pd
import os
import signal
import subprocess
import time


def free_port(port):
    """Kill any process already listening on the given port (macOS/Linux, uses lsof)."""
    result = subprocess.run(['lsof', '-ti', f'tcp:{port}', '-sTCP:LISTEN'],
                            capture_output=True, text=True)
    pids = [int(p) for p in result.stdout.split() if int(p) != os.getpid()]
    for pid in pids:
        print(f'Port {port} in use by PID {pid}, killing it')
        os.kill(pid, signal.SIGTERM)
    if pids:
        time.sleep(1)

# 2. Create the app object
app = FastAPI()
pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)


# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome T': f'{name}'}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_banknote(data:BankNote):
    data = data.model_dump()
    # The model was trained on a DataFrame, so pass one with matching column names
    features = pd.DataFrame([data], columns=['variance', 'skewness', 'curtosis', 'entropy'])
    prediction = classifier.predict(features)
    if(prediction[0]>0.5):
        prediction="Fake note"
    else:
        prediction="Its a Bank note"
    return {
        'prediction': prediction
    }

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    free_port(8002)
    uvicorn.run(app, host='127.0.0.1', port=8002)
    
#uvicorn app:app --reload