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
import xmltodict

import numpy as np
import pandas as pd
import statistics as stat

from flask import render_template
from io import StringIO
from datetime import datetime
from sklearn import metrics

from frameworks.uppsala_vitals.utils import (
    parse_json_data, 
    predict_instrument, 
    generate_trialid, 
    append_secret, 
    render_densplot,
    generate_tokens, 
    trans_fun,
    bayes_opt_xgb,
    xgb_cv_fun,
    get_model_props,
    clean_data,
    split_data,
    generate_ui_data,
    generate_names,
    parse_text_to_bow,
    get_stopword_set,
    nemsis3_to_vitals_dict
    )

class Main:

    def __init__(self,
        code_dir,
        log,
        cache,
        # Paths for all the things
        name_path ='models/pretty_names.json',
        stopword_path = 'models/stopwords.json',
        model_path_dict = {
            'models':'models/models.p',
            'model_props':'models/model_props.json'
            },
        data_path_dict = {
            'train_labels':'data/train/labels.csv',
            'train_data':'data/train/data.csv',
            'test_labels':'data/test/labels.csv',
            'test_data':'data/test/data.csv'
            }, 
        clean_data_path_dict = {
            'clean_labels':'data/clean/labels.csv',
            'clean_data':'data/clean/data.csv'
            },
        raw_data_path_dict = {
            'qliksense_export':'data/raw/qliksense_export.xlsx'
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
            'hosp_admit':1,
            'hosp_critcare':1,
            'hosp_2daymort':1
        },
        instrument_trans = 'logit',
        filter_str = '',
        randomize = False,
        # Data parsing stuff
        overwrite_models = False,
        overwrite_data = False,
        # Define inclusion criteria for qs data
        inclusion_criteria = ["valid_pin","valid_geo_dest","exists_amb"],
        # Define composite measures defined as any one of several variables:
        label_dict = {
            "amb_intervention" : ["amb_meds", "amb_cpr", "amb_o2", "amb_immob", "amb_crit", "amb_alert", "amb_ecg"],
            "amb_prio" : ["amb_prio"],
            "hosp_admit" : ["hosp_admit"],
            "hosp_critcare" : ["hosp_icu","hosp_30daymort"],
            "hosp_2daymort" : ["hosp_2daymort"]},
        # Define predictors to extract
        predictors = ['CreatedOn','disp_age','disp_gender','disp_cats', 'disp_prio', 'eval_breaths', 'eval_spo2','eval_sbp', 'eval_pulse', 'eval_temp','eval_avpu'],
        parse_text = None,
        max_ngram = 2, 
        text_prefix = 'text_', 
        min_terms = 500,
        # Test/train sample splitting
        test_cutoff_ymd = '20200319',
        test_sample = 0.3,
        test_criteria = [],
        # UI stuff
        prod_ui_cols = ['value','mean_shap']
        ):

        """
            Class should handle loading the model file upon initiation.
        """

        self.code_dir = code_dir
        self.log = log
        self.cache = cache
        self.params_ranges = params_ranges
        self.opt_rounds = opt_rounds
        self.init_rounds = init_rounds
        self.max_estimators = max_estimators
        self.out_weights = out_weights
        self.filter_str = filter_str
        self.instrument_trans = instrument_trans
        self.randomize = randomize
        self.parse_text = parse_text
        self.max_ngram = max_ngram
        self.text_prefix = text_prefix
        self.min_terms = min_terms
        self.test_cutoff_ymd = test_cutoff_ymd
        self.test_sample = test_sample
        self.test_criteria = test_criteria
        self.prod_ui_cols = prod_ui_cols

        # Make and check full paths
        full_name_path = f'{self.code_dir}/{name_path}'
        full_stopword_path = f'{self.code_dir}/{stopword_path}'

        full_data_paths = {k:f'{self.code_dir}/{v}' for k,v in data_path_dict.items()}
        data_paths_exist = {k:os.path.exists(v) for k,v in full_data_paths.items()}

        full_model_paths = {k:f'{self.code_dir}/{v}' for k,v in model_path_dict.items()}
        model_paths_exist = {k:os.path.exists(v) for k,v in full_model_paths.items()}
        
        # If a serialized model is available, load it

        if all(model_paths_exist.values()) and not overwrite_models:
            self.log.info("Models found! Loading...")
            self.model = self._load_model(full_model_paths)

        # Otherwise, try to load data
        else:
            if all(data_paths_exist.values()) and not overwrite_data:
                self.log.info("Data found! Loading...")
                self.data = self._load_data(full_data_paths,full_stopword_path)

            # If no clean data is available, try to parse clean data
            else:
                self.log.warning("Missing data file(s)! Cleaning data...")

                data_clean = clean_data(
                    code_dir=code_dir, 
                    raw_data_paths=raw_data_path_dict, 
                    clean_data_paths=clean_data_path_dict, 
                    full_name_path=full_name_path, 
                    overwrite_data=overwrite_data, 
                    inclusion_criteria=inclusion_criteria, 
                    label_dict=label_dict, 
                    predictors=predictors,
                    text_col=parse_text,
                    key_table=None, 
                    filter_str=self.filter_str, 
                    log=self.log
                    )

                data_split = split_data(
                    data = data_clean, 
                    test_cutoff_ymd = self.test_cutoff_ymd, 
                    test_sample = self.test_sample, 
                    test_criteria = self.test_criteria)

                self._save_data(data_split, full_data_paths) # Write data to disk
                # Load it from disk (to avoid making stupid mistakes)
                self.data = self._load_data(full_data_paths,full_stopword_path)
            
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
        
        # Maybe load pretty names for display in UI
        if os.path.exists(full_name_path):
            self.log.info("Pretty names found! Loading...")
            with open(full_name_path, "r") as f:
                    names = json.load(f)
            self.model['names'] = names
        else:
            self.log.info("No pretty names found!")
        
        # Maybe load stopwords for text parsing
        if os.path.exists(full_stopword_path):
            self.log.info("Stopwords found! Loading...")
            with open(full_stopword_path, "r") as f:
                    stopwords = json.load(f)
            self.stopword_set = set(stopwords)
        else:
            self.log.info("No stopwords found!")

    def input_function(self, request):
        """input_function is a required function to parse incoming data"""

        if request.content_type == 'application/json':
            request_data = request.data
        if request.content_type == 'application/xml':
            request_data = nemsis3_to_vitals_dict(request.data)

        self.log.debug(request_data)
        
        # Loop through each item sent through the API
        results = {}
        for id, value in request_data.items():
            # Apply a parsing function (from functions.py) to each item
            results[id] = parse_json_data(value,self.model,log = self.log)

            if self.parse_text is not None:
                
                text_df = pd.DataFrame({self.parse_text:value[self.parse_text]}, index=[0])
                term_list = results[id].loc[:,results[id].columns.str.startswith(self.text_prefix)]
                
                bow_df = parse_text_to_bow(
                    text_df, 
                    max_ngram = self.max_ngram, 
                    text_prefix = self.text_prefix, 
                    stopword_set = self.stopword_set, 
                    term_list = term_list,
                    log = self.log)

                results[id].update(bow_df)
                
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
            # self.log.debug(prediction)
            # Add the score to a dictionary to be returned for further processing
            preds[id] = prediction

        return preds

    def output_function(self, prediction):
        """output_function prepares predictions to be sent back in response"""

        # Generate IDs for storage in db (with 32 byte secret to prevent guessing ids)
        store_ids = {key : append_secret(key) for key in prediction.keys()}
        out_dict = {}
        for id,value in prediction.items():
            # Add info other included ids
            other_ids = list(store_ids.values())
            other_ids.remove(store_ids[id])
            value['other_ids'] = other_ids

            # Store the prediction
            # self.cache.set(store_ids[id], json.dumps(value))
            # Set cache data to expire in one hour
            # self.cache.expire(store_ids[id],60*60)

            #TODO: implement showing other scores on plot
            other_scores = []

            ui_data = generate_ui_data(
                value,
                other_scores, 
                self.prod_ui_cols,
                self.text_prefix,
                self.model,
                self.log
                )

            page = render_template(
                "testui.html", 
                title = f"Overall risk: {np.round(value['score'],2)}", 
                fig_base64 = ui_data['fig_base64'], 
                components = ui_data['components'].render(), 
                feat_imp = ui_data['feat_imp_table'].render())
            
            

            out_dict[id] = {
                'score':value['score'],
                'html':str(page)
                }

        #self.log.debug(out_dict)
        # Return dict as a json file
        return json.dumps(out_dict)

    def ui_function(self, id):
        
        pass

    def _load_data(self,data_path_dict,stopword_path):

        self.log.info("Loading data...")
        train_data = pd.read_csv(data_path_dict['train_data'],index_col='caseid')
        train_labels = pd.read_csv(data_path_dict['train_labels'],index_col='caseid')

        test_data = pd.read_csv(data_path_dict['test_data'],index_col='caseid')
        test_labels = pd.read_csv(data_path_dict['test_labels'],index_col='caseid')

        if self.parse_text:
            self.log.info("Parsing text data...")
            # Load text data
            train_text = pd.read_csv(data_path_dict['train_text'],index_col='caseid')
            test_text = pd.read_csv(data_path_dict['test_text'],index_col='caseid')

            # Get stop words - Note that we keep some clinically relevant negation terms here
            stopword_url = "https://gist.githubusercontent.com/hannseman/5608626/raw/7b11a75d393a68d0145bdcefb06fc06d2764daa8/swedish_stopwords.txt"
            neg_set = set(["inte","ej","icke","ingen","blir","bli","blev","blivit","mycket","nu","utan","var"])
            stopword_set = get_stopword_set(stopword_url,neg_set)

            # Save to disk for later use in API
            with open(stopword_path,'w', encoding='utf-8') as f:
                json.dump(list(stopword_set), f, ensure_ascii=False, indent=4)

            # Generate bag of words embedding for terms (yes yes, I know there are more sohpisticated ways of doing this)
            # Note that for test data, we use only the features included in the training set.
            train_bow = parse_text_to_bow(
                train_text,
                self.max_ngram, 
                self.text_prefix, 
                self.min_terms, 
                stopword_set)

            test_bow = parse_text_to_bow(
                test_text,
                self.max_ngram, 
                self.text_prefix, 
                self.min_terms, 
                stopword_set,
                term_list=list(train_bow.columns))

            assert len(train_bow.columns) == len(test_bow.columns)

            train_data = train_data.join(train_bow)
            test_data = test_data.join(test_bow)
        
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

        data['train']['data'].to_csv(data_paths['train_data'])
        data['train']['labels'].to_csv(data_paths['train_labels'])

    def _load_model(self, model_paths):

        # Store model function (or object)
        self.log.info(f'Loading model at "{self.code_dir}"')
        
        with open(model_paths['model_props']) as f:
            model_props = json.load(f)

        with open(model_paths['models'], "rb") as f:
            models = pickle.load(f)

        mod_dict = {'models': models, 
                    'model_props': model_props}

        # Return model function or object.
        return mod_dict

        # Make sure predict_function handles 'model' correctly!
    
    def _save_model(self, model, model_paths):
        self.log.info(f'Saving model to "{self.code_dir}"')

        # Note that we save these seperately instead of just pickling 
        # everything to make the model properties human-readable 
        # for debugging purposes

        with open(model_paths['model_props'], "w", encoding='utf-8') as f:
            json.dump(model['model_props'],f,ensure_ascii=False,indent=4)

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

            train_dmatrix = xgboost.DMatrix(
                self.data['train']['data'], 
                label = values, 
                weight=obs_weights, 
                feature_names=self.data['train']['data'].columns)

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