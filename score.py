import json
import numpy as np
import os
from sklearn.externals import joblib
from azureml.core.model import Model
import joblib, pickle
def init():
    global model
    #model_path = Model.get_model_path('best_run')
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.joblib')
    model = joblib.load(model_path)
def run(data):
    try:
       # data = np.array(json.loads(data))
        data = json.loads(data)
        strn = "extracted json\n"
        data = np.array(data["data"])
        strn += "converted data to np array\n"
        result = model.predict(data)
        strn += "sent the data to the model for prediction\n"
        print(strn)
        # You can return any data type, as long as it is JSON serializable.
        return result.tolist()
    except Exception as err:
        return strn+"run method error: "+str(err)
        
    
