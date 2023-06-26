#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import pandas as pd
import sys

from sklearn import linear_model

from scripts.utils import list_recursive, log

def local_0(args):
    log(args['input'], args['state'])

    file = args['input']['data'][0]
    df = pd.read_csv(os.path.join(args["state"]["baseDirectory"], file))


    x_headers = args['input']['x_headers']
    y_headers = args['input']['y_headers']
    
    predictions = args['input']['predictions']

    X = df[x_headers]
    y = df[y_headers]

    regr = linear_model.LinearRegression()
    regr.fit(X,y)

    local_predictions = []

    for p in args['input']['predictions']:
        local_predictions.append(regr.predict(p))

    output = {
        'output': { 
            "local_X": X.to_json(),
            "local_Y": y.to_json(), 
            "predictions": predictions,
            "computation_phase": "local_0",
        }
    }

    log(output, args['state'])

    return output

def start(PARAM_DICT):
    PHASE_KEY = list(list_recursive(PARAM_DICT, "computation_phase"))

    if not PHASE_KEY:
        return local_0(PARAM_DICT)
    elif "remote_0" in PHASE_KEY:
        return local_0(PARAM_DICT)
    else:
        raise ValueError("Error occurred at Local")