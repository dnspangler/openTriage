import json
import logging

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
            load_model should load and return the model. In this case, we define a simple logical statement to apply the NEWS 2 algorithm.
        """

        # store your model function (or object) in variable model
        self.log.info('Loading NEWS2 model')

        # Define NEWS scoring algorithm per
        # https://www.rcplondon.ac.uk/projects/outputs/national-early-warning-score-news-2

        #NOTE: This code should not be used in production. Needs more 
        # asserts and distribution checks. Will be implementing parsing 
        # for NEMSIS3 data and will build that out against that standard.

        def model(input_data):
            if input_data['respiration_rate'] <= 8:
                rr = 3
            elif input_data['respiration_rate'] <= 11:
                rr = 1
            elif input_data['respiration_rate'] <= 20:
                rr = 0
            elif input_data['respiration_rate'] <= 24:
                rr = 2
            else:
                rr = 3

            if input_data['oxygen_saturation'] <= 91:
                sat = 3
            elif input_data['oxygen_saturation'] <= 93:
                sat = 2
            elif input_data['oxygen_saturation'] <= 95:
                sat = 1
            else:
                sat = 0

            if input_data['supplemental_oxygen'] == "Oxygen":
                oxygen = 2
            elif input_data['supplemental_oxygen'] == "Air":
                oxygen = 0
            else:
                #self.log.warning("Invalid supplemental oxygen value given! Assuming patient is on room air.")
                oxygen = 0

            if input_data['systolic_blood_pressure'] <= 90:
                sbp = 3
            elif input_data['systolic_blood_pressure'] <= 100:
                sbp = 2
            elif input_data['systolic_blood_pressure'] <= 110:
                sbp = 1
            elif input_data['systolic_blood_pressure'] <= 219:
                sbp = 0
            else:
                sbp = 3

            if input_data['pulse'] <= 40:
                pulse = 3
            elif input_data['pulse'] <= 50:
                pulse = 1
            elif input_data['pulse'] <= 90:
                pulse = 0
            elif input_data['pulse'] <= 110:
                pulse = 1
            elif input_data['pulse'] <= 130:
                pulse = 2
            else:
                pulse = 3

            if input_data['consciousness'] == "Alert":
                avpu = 0
            elif input_data['consciousness'] in ["Verbal","Painful","Unconscious"]:
                avpu = 3
            else:
                #self.log.warning("Invalid Consciousness value given! Assuming patient is Alert.")
                avpu = 0

            if input_data['temperature'] <= 35.0:
                temp = 3
            elif input_data['temperature'] <= 36.0:
                temp = 1
            elif input_data['temperature'] <= 38.0:
                temp = 0
            elif input_data['temperature'] <= 39.0:
                temp = 1
            else:
                temp = 2
            
            news = rr + sat + oxygen + sbp + pulse + avpu + temp

            return news
        
        # return model function or object.
        # make sure predict_function handles 'model' correctly!
        return model

    def input_function(self, request_data):
        """input_function is a required function to parse incoming data.
        Do any pre-processing steps required to transform your 
        data into the format required by your model here.
        In this case, we don't need to apply any transformations, 
        and the model handles the raw dictionary directly. 
        
        If using ML models, you probably want to make the output 
        of this function a pandas dataframe."""

        results = {}

        for id, value in request_data.items():
            results[id] = value

        self.log.debug(results)
        return results

    def predict_function(self, input_data):
        """
            predict_function applies model to input_data.
        `input_data` should be expected to be the output of 
        interface.input_function
        """
        prediction = {}

        for id, value in input_data.items():
            prediction[id] = self.model(value)

        return prediction    

    def output_function(self, prediction):
        """output_function prepares predictions to be sent back in response"""

        out_dict = {}
        for id, value in prediction.items():
            out_dict[id] = {'score':value,
                            'link':f'/html/news_adhoc?id={id}'}

        return json.dumps(out_dict)

    def ui_function(self, id):
        """ui_function renders a web page to provide details about a prediction"""

        #load prediction data from cache (deserialize!)

        try:
            store = json.loads(self.cache.get(id))
        except:
            return "ID not found!"

        text = f'NEWS score is {store}'

        return render_template("testui_minimal.html", title = "Details", text = text)