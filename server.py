from flask import Flask, request
import model

app = Flask(__name__)


@app.route('/get_predict', methods=['GET'])
def get_predict():
    data = request.args['data']
    data = eval(data)
    ai = model.advanced_interface()
    result = ai.predict(data)
    return str(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8090)
