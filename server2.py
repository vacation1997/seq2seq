from werkzeug.wrappers import Request, Response
from werkzeug.serving import run_simple
from jsonrpc import JSONRPCResponseManager, dispatcher

import json
import numpy as np

import model


@dispatcher.add_method
def mnist_predict(data):
    # data = np.array(data)  # list
    ai = model.advanced_interface()
    label = ai.predict(data)
    return json.dumps({"state": 0, "result": label})


@Request.application
def application(request):
    # 手工注册rpc服务
    dispatcher['predict'] = mnist_predict
    response = JSONRPCResponseManager.handle(request.data, dispatcher)
    return Response(response.json, mimetype='application/json')


if __name__ == '__main__':
    run_simple('localhost', 4567, application)
