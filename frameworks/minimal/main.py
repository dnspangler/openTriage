import json

import numpy as np

from flask import render_template

class Main:

    def __init__(self,
        code_dir,
        log,
        cache
        ):

        """
            Class should handle loading the model file upon initiation.
        """
        self.code_dir = code_dir
        self.log = log
        self.cache = cache

        self.model = self._load_model()
        
    def _load_model(self):
        """
            load_model should load and return the model
        """

        # store your model function (or object) in variable model

        self.log.info(f'Loading model which generates a random number between 0 and 1"')

        def model(input_data):
            # Return a random number between 0 and 1
            return float(np.random.uniform(0,1,1))
        
        # return model function or object.
        # make sure predict_function handles 'model' correctly.
        return model

    def input_function(self, request):
        """input_function is a required function to parse incoming data"""

        results = request.data

        return results

    def predict_function(self, input_data):
        """
            predict_function applies model to input_data.
        `input_data` should be expected to be the output of 
        interface.input_function
        """
        
        prediction = self.model(input_data)
        
        return prediction

    def output_function(self, prediction):
        """output_function prepares predictions to be sent back in response"""
        
        out_dict = {
            "case_identifier":{
                "score" : prediction}
            }

        return json.dumps(out_dict)