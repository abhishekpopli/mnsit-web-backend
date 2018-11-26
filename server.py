from flask import Flask, request, jsonify
from flask_cors import CORS

from keras.models import load_model
import numpy as np

from test import testArr

app = Flask(__name__)
CORS(app)

###############################

model = load_model('mnistCNN.h5')

def recogniseDigit(arr):
    img = np.array(arr)
    image = img.reshape((28,28))
    im2arr = image.reshape((28,28,1))
    result = model.predict_classes(np.array([im2arr]))[0]
    return result

##################################


@app.route("/recognise", methods=['POST'])
def recognise():
    data = request.get_json().get('data', None)
    
    statusCode = 200

    if data is not None:
        result = str(int(recogniseDigit(data)))
    else:
        result = "Input array is not correct"
        statusCode = 400
    
    dataToSend = {"result": result}
    print(f'>>> Sending data {dataToSend}')
    return jsonify(dataToSend), statusCode


print(recogniseDigit(testArr))

if __name__== '__main__':
    app.run()