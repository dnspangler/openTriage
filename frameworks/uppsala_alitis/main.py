import json
import os
import io
import base64
import pickle
import subprocess
import itertools
import xgboost
import time
import csv

import numpy as np
import pandas as pd
import statistics as stat

from flask import render_template
from io import StringIO
from datetime import datetime
from sklearn import metrics

from frameworks.uppsala_alitis.utils import (
    parse_json_data, 
    predict_instrument, 
    generate_trialid, 
    append_secret, 
    render_densplot, 
    fig_to_base64,
    generate_tokens, 
    trans_fun,
    bayes_opt_xgb,
    xgb_cv_fun,
    get_model_props,
    clean_data,
    generate_ui_data,
    generate_names
    )

class Main:

    def __init__(self,
        code_dir,
        log,
        cache,
        # Paths for all the things
        key_path ='models/mbs_key.csv',
        name_path ='models/pretty_names.json',
        model_path_dict = {
            'models':'models/models.p',
            'model_props':'models/model_props.json'
            },
        data_path_dict = {
            'train_labels':'data/train/labels.csv',
            'train_data':'data/train/data.csv',
            'train_text':'data/train/text.csv',
            'test_labels':'data/test/labels.csv',
            'test_data':'data/test/data.csv',
            'test_text':'data/test/text.csv'
            }, 
        clean_data_path_dict = {
            'clean_labels':'data/clean/labels.csv',
            'clean_data':'data/clean/data.csv',
            'clean_text':'data/clean/text.csv'
            },
        raw_data_path_dict = {
            'mbs_answers':'data/raw/mbs_answers.csv',
            'mbs_cases':'data/raw/mbs_cases.csv',
            'mbs_neg_groups':'data/raw/mbs_neg_groups.csv',
            'qliksense_export':'data/raw/qliksense_export.xlsx',
            'mbs_dict':'data/raw/mbs_dict.json'
            },
        # Hyperparameter tuning stuff
        params_ranges = {
            'subsample' : (0.5,1), 
            'colsample_bytree' : (0.5,1),
            'max_depth' : (4,12),
            'min_child_weight' : (0,8), 
            'learning_rate' : (0.01,0.2),
            'gamma' : (0.01,5)
            },
        opt_rounds=40,
        init_rounds=10,
        max_estimators=400,
        # Prediction instrument stuff
        out_weights = {
            'amb_intervention':1,
            'amb_prio':1,
            'hosp_critcare':1
        },
        instrument_trans = 'logit',
        filter_str = "_Bedmt_tillstnd_|_Nej",
        randomize = False,
        overwrite_models = False,
        overwrite_data = False,
        prod_ui_cols = ['value','mean_shap']
        ):

        """
            Class should handle loading the model file upon initiation.
        """
        
        self.code_dir = code_dir
        self.log = log
        self.cache = cache
        self.data_path_dict = data_path_dict
        self.params_ranges = params_ranges
        self.opt_rounds = opt_rounds
        self.init_rounds = init_rounds
        self.max_estimators = max_estimators
        self.out_weights = out_weights
        self.filter_str = filter_str
        self.instrument_trans = instrument_trans
        self.randomize = randomize
        self.prod_ui_cols = prod_ui_cols

        # Make and check full paths
        full_key_path = f'{self.code_dir}/{key_path}'

        full_name_path = f'{self.code_dir}/{name_path}'

        full_data_paths = {k:f'{self.code_dir}/{v}' for k,v in data_path_dict.items()}
        data_paths_exist = {k:os.path.exists(v) for k,v in full_data_paths.items()}

        full_model_paths = {k:f'{self.code_dir}/{v}' for k,v in model_path_dict.items()}
        model_paths_exist = {k:os.path.exists(v) for k,v in full_model_paths.items()}

        # Load key file needed for parsing API calls
        try:
            self.key = self._load_key(full_key_path)
        except NameError:
            self.log.exception("No key found!")
        
        # If a serialized model is available, load it

        if all(model_paths_exist.values()) and not overwrite_models:
            self.log.info("Models found! Loading...")
            self.model = self._load_model(full_model_paths)

        # Otherwise, try to load data
        else:
            if all(data_paths_exist.values()) and not overwrite_data:
                self.log.info("Data found! Loading...")
                self.data = self._load_data(full_data_paths)

            # If no clean data is available, try to parse clean data
            else:
                self.log.warning("Missing data file(s)! Parsing data...")
                data = clean_data(code_dir, raw_data_path_dict, clean_data_path_dict, full_name_path, self.key, self.filter_str, self.log)
                self._save_data(data, full_data_paths) # Write data to disk
                self.data = data
            
            # Once data has (hopefully) been loaded, train a model on it.
            try:
                self.log.info("No models found! Training...")
                model = self._train_model()
                self._save_model(model, full_model_paths) # Write model to disk
                self.model = model
                #TODO: Implement nicer reports on model performance to be provided upon training a new model
                #test_model(model, data)

            # If no data or models are available, crash the server.
            except NameError:
                self.log.exception("No data or models found!")
        
        # Load pretty names for display in UI
        if os.path.exists(full_name_path):
            self.log.info("Pretty names found! Loading...")
            with open(full_name_path, "r") as f:
                    names = json.load(f)
            self.model['names'] = names
        else:
            self.log.info("No pretty names found!")

    def input_function(self, request_data):
        """input_function is a required function to parse incoming data"""
        self.log.debug(request_data)
        # Loop through each item sent through the API
        results = {}
        for id, value in request_data.items():
            # Apply a parsing function (from functions.py) to each item
            results[id] = parse_json_data(value,self.model,log = self.log)

        return results

    def predict_function(self, input_data):
        """
        predict_function applies model to input_data.
        `input_data` should be expected to be the output of 
        interface.input_function
        """    
        
        # Loop through each item returned by the input function

        preds = {}

        for id,value in input_data.items():
            
            # Apply a prediction function to the parsed data
            prediction = predict_instrument(model = self.model,new_data = value, log = self.log)
            # self.logdebug(prediction)
            # Add the score to a dictionary to be returned for further processing
            preds[id] = prediction

        return preds

    def output_function(self, prediction):
        """output_function prepares predictions to be sent back in response"""

        # Generate a trial ID to keep track of which records were compared with eachother
        trialID = generate_trialid(prediction.keys())

        # Generate IDs for storage in db (with 32 byte secret to prevent guessing ids)
        store_ids = {key : append_secret(key) for key in prediction.keys()}
        
        for id,value in prediction.items():
            # Add info other included ids
            other_ids = list(store_ids.values())
            other_ids.remove(store_ids[id])
            value['other_ids'] = other_ids

            # Store the prediction
            self.cache.set(store_ids[id], json.dumps(value))
            # Set cache data to expire in one hour
            self.cache.expire(store_ids[id],60*60)
    

        # Get scores for each prediction
        scores = {key : value['score'] for key, value in prediction.items()}
        # Compare the scores of each call and rank them (Highest score = 1)
        ranks = {key : rank for rank, key in enumerate(sorted(scores, key=scores.get, reverse=True), 1)}
        # Loop through each score
        out_dict = {}

        for id,value in scores.items():
            # Add items which should be returned regardless of inclusion in control/intervention arm
            out_dict[id] = {'score':value,'trialID':trialID}

        # Apply randomization procedure (i.e., generate a 0/1 randomly with equal likelihoods) if desired
        if self.randomize:
            group = round(np.random.uniform()) # Randomize
        else:
            group = 1 # Don't Randomize

        if group == 0:
            # Add control arm output data to dict
            for id,value in prediction.items():
                out_dict[id]['text'] = '-'
                out_dict[id]['color'] = '#c0c0c0'
                out_dict[id]['link'] = f'/ui?id=control'
                out_dict[id]['group'] = 'control'

        else:
                
            for id,value in ranks.items():
                # Mark the highest ranked call with a red color (This is the one that should get an ambulance first!), and grey for all others
                if value == 1: 
                    col = '#ff0000'
                else:
                    col = '#c0c0c0'
                
                # Add output data to be displayed in intervention arm
                out_dict[id]['text'] = str(value)
                out_dict[id]['color'] = col
                out_dict[id]['link'] = f'/ui?id={store_ids[id]}'
                out_dict[id]['group'] = 'intervention'

        self.log.debug(out_dict)
        # Return dict as a json file
        return json.dumps(out_dict)

    def ui_function(self, id):
        
        #load prediction data from cache (deserialize!)
        if id == "control":
            return "Kontrollärende! Genomför prioritering enligt klinisk praxis."
        else:
            store = json.loads(self.cache.get(id))

        # Get other predictions in trial for comparison
        other_scores = []
        for oid in store['other_ids']:
            op = json.loads(self.cache.get(oid))
            other_scores.append(op['score'])

        ui_data = generate_ui_data(store,other_scores,self.prod_ui_cols,self.model,self.log)

        return render_template(
            "testui.html", 
            title = f"Övergripande risk: {np.round(store['score'],2)}", 
            fig_base64 = ui_data['fig_base64'], 
            components = ui_data['components'].render(), 
            feat_imp = ui_data['feat_imp_table'].render())

    def _load_data(self,data_path_dict):

        #TODO: Implement parsing text to BOW for terms and bigrams, and append to 
        # datasets. Simple, but works as well as anything else I've tried in our data!

        train_data = pd.read_csv(data_path_dict['train_data'],index_col='caseid')
        train_labels = pd.read_csv(data_path_dict['train_labels'],index_col='caseid')
        #train_text = pd.read_csv(data_path_dict['train_text'],index_col='caseid')

        test_data = pd.read_csv(data_path_dict['test_data'],index_col='caseid')
        test_labels = pd.read_csv(data_path_dict['test_labels'],index_col='caseid')
        #test_text = pd.read_csv(data_path_dict['test_text'],index_col='caseid')
        
        data = {
            'train':{
                'data':train_data,
                'labels':train_labels,
            },
            'test':{
                'data':test_data,
                'labels':test_labels
            }
        }

        return data
    
    def _save_data(self,data,data_paths):

        data['test']['data'].to_csv(data_paths['test_data'])
        data['test']['labels'].to_csv(data_paths['test_labels'])
        data['test']['text'].to_csv(data_paths['test_text'])

        data['train']['data'].to_csv(data_paths['train_data'])
        data['train']['labels'].to_csv(data_paths['train_labels'])
        data['train']['text'].to_csv(data_paths['train_text'])

    def _load_key(self, key_path):
        
        with open(key_path) as f:
            key = pd.read_csv(f, encoding='utf-8').fillna('')
        
        # Generate tokens corresponding to each possible question/answer combination
        # Note that in general, we try to conform with the variable prefixing 
        # scheme used in this article: https://doi.org/10.1371/journal.pone.0226518

        key['cat_token'], key['qa_token'], key['qa_name'] = generate_tokens(key_table = key)

        # We remove some unnecessary tokens (in our case, inactivated questions and 
        # "no" answers which are colinear with "yes" answers)
        key = key[~key['qa_token'].str.contains(self.filter_str)]

        return key

    def _load_model(self, model_paths):

        # Store model function (or object)
        self.log.info(f'Loading model at "{self.code_dir}"')
        
        with open(model_paths['model_props']) as f:
            model_props = json.load(f)

        with open(model_paths['models'], "rb") as f:
            models = pickle.load(f)

        mod_dict = {'models': models, 
                    'model_props': model_props,
                    'key':self.key}

        # Return model function or object.
        return mod_dict

        # Make sure predict_function handles 'model' correctly!
    
    def _save_model(self, model, model_paths):
        self.log.info(f'Saving model to "{self.code_dir}"')

        # Note that we save these seperately instead of just pickling 
        # everything to make the model properties human-readable 
        # for debugging purposes

        with open(model_paths['model_props'], "w") as f:
            json.dump(model['model_props'],f,indent=4)

        with open(model_paths['models'], "wb") as f:
            pickle.dump(model['models'],f)

    def _train_model(self):

        self.log.info("Training models...")

        date_min = min(self.data['train']['data']['disp_date'])
        date_max = max(self.data['train']['data']['disp_date'])
        span = date_max - date_min
        obs_weights = [(i - date_min)/span for i in self.data['train']['data']['disp_date']]

        # Tune Hyperparameters ----------------------------------------------------------------

        if not os.path.exists(f"{self.code_dir}/models/tune_logs.json"):
            self.log.info("No hyperparameter tuning logs found! Finding optimal hyperparameters... (this will take a while)")
            
            # 
            log_params = {}
            # For each label in the training dataset...
            for name, values in self.data['train']['labels'].iteritems(): 
                # Generate dmatrix for xgb model
                train_dmatrix = xgboost.DMatrix(self.data['train']['data'], label = values, weight=obs_weights)
                # Optomiz parameters and save to log
                log_params[name] = bayes_opt_xgb(
                    dmatrix=train_dmatrix, 
                    log = self.log, 
                    opt_fun = xgb_cv_fun, 
                    opt_rounds = self.opt_rounds, 
                    init_rounds = self.init_rounds,
                    params_ranges = self.params_ranges,
                    max_estimators = self.max_estimators
                    )
            # Save log to file
            json.dump(log_params, open(f"{self.code_dir}/models/tune_logs.json", "w" ), indent = 4)

        else:
            # If a file already exists, load it
            self.log.info("Hyperparameter tuning logs found, loading...")
            log_params = json.load(open(f"{self.code_dir}/models/tune_logs.json", "r"))

        # Train models -------------------------------------------------------------------------

        fits = {}
        for name, values in self.data['train']['labels'].iteritems(): 

            self.log.info(f"Training {name}")

            train_dmatrix = xgboost.DMatrix(self.data['train']['data'], label = values, weight=obs_weights)

            # Get best hyperparameters for label from log 
            log_best = log_params[name][max(log_params[name].keys())]
            best_params = log_best['params']

            fits[name] = xgboost.train(best_params,
                                train_dmatrix,
                                num_boost_round=log_best['fit_props']['n_estimators'])

            # Print some quick "sanity check" results
            test_dmatrix = xgboost.DMatrix(self.data['test']['data'], label = self.data['test']['labels'][name])
            self.log.info("Mean pred: "+
                str(np.mean(fits[name].predict(test_dmatrix))) +
                " Test AUC: " +
                str(metrics.roc_auc_score(self.data['test']['labels'][name],
                    fits[name].predict(test_dmatrix))))

        model_props = get_model_props(
            fits,
            data = self.data,
            out_weights = self.out_weights,
            instrument_trans = self.instrument_trans)

        model_dict = {
            'models':fits,
            'model_props':model_props
        }

        return model_dict