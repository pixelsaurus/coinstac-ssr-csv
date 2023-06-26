#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
import pandas as pd
from sklearn import linear_model

from scripts.utils import list_recursive, log

def remote_0(args):   
    log(args['input'], args['state'])

    dfX = pd.DataFrame()
    dfY = pd.DataFrame()

    predictions = []

    for site, input in args['input'].items():
        siteDFX = pd.read_json(input['local_X'])
        siteDFY = pd.read_json(input['local_Y'])

        predictions = input['predictions']

        if dfX.empty:
            dfX = siteDFX
        else: 
            dfX = pd.concat([ dfX, siteDFX ])


        if dfY.empty:
            dfY = siteDFY
        else: 
            dfY = pd.concat([ dfY, siteDFY ])

    regr = linear_model.LinearRegression()

    regr.fit(dfX,dfY)

    predict = []

    for p in predictions:
        predict.append(str(regr.predict(p)))

    

    output = {
        'output': { 
            "predictions": str(predict),
            "computation_phase": "remote_0",
        },
       "success": True
    }

    log(output, args['state'])

    return output

def start(PARAM_DICT):
    PHASE_KEY = list(list_recursive(PARAM_DICT, "computation_phase"))

    if "local_0" in PHASE_KEY:
        return remote_0(PARAM_DICT)
    else:
        raise ValueError("Error occurred at Remote")