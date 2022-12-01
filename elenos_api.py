

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json, re, time, os, traceback, pickle, torch
from tqdm import tqdm
from collections import Counter
import torch.nn as nn
from flask import Flask
from flask import request
from flask import redirect, url_for, jsonify
from pprint import pprint
from openpyxl import load_workbook
import numpy as np
from transformers import AutoTokenizer, AutoModel

import logging
logging.basicConfig(filename='sdg_api.log', level=logging.INFO, format="%(asctime)s;%(levelname)s;%(message)s")

app = Flask(__name__)

bioclean_mod    = lambda t: re.sub(
    '[~`@#$-=<>/.,?;*!%^&_+():-\[\]{}]',
    '',
    t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').replace("-", ' ').strip()
)


@app.route("/get_one_pmid", methods=["POST", "GET"])
def get_one_pmid():
    pass

@app.route("/submit_question", methods=["POST", "GET"])
def submit_question():
    try:
        app.logger.debug("JSON received...")
        app.logger.debug(request.json)
        if request.json:
            mydata = request.json
            pprint(mydata)
            ###################################################################################################
            ret = {
                'request'               : mydata,
                'results'               : {}
            }
            ###################################################################################################
            return jsonify(ret)
        else:
            ret = {
                'success': 0,
                'message': 'request should be json formated'
            }
            app.logger.debug(ret)
            return jsonify(ret)
    except Exception as e:
        app.logger.debug(str(e))
        traceback.print_exc()
        ret = {'success': 0, 'message': str(e) + '\n' + traceback.format_exc()}
        app.logger.debug(ret)
        return jsonify(ret)

if __name__ == '__main__':
    app.run(
        host            = '0.0.0.0',
        port            = 29927,
        debug           = False,
        threaded        = True,
        # use_reloader    = False
    )


