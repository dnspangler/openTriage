import pandas as pd
import numpy as np
import scipy
import requests
import matplotlib.pyplot as plt
import json
import io
import os
import base64
import xgboost
import secrets
import re
import datetime as dt
import time
import itertools
import collections
import random
import time
import calendar
import xmltodict

from pathlib import Path
from datetime import date, datetime
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction


# These functions are used by the main class to perform functions 
# particular to the use case of predicting outcomes based on EMD 
# center data in Uppsala. Much of this code is fairly ideosyncratic, 
# but may hopefully be of some use outside the context of our system. 

#NOTE: This code is in development and liable to change without warning.

# Interface functions ----------------------------------------------

def render_densplot(pred,dist,other_preds):
    """ Render a density plot of prediction distribution in population with vertical lines indicating predicted values """
    d = pd.Series(dist)
    fig, ax = plt.subplots(figsize=(6,3))
    d.plot.kde(bw_method=1)
    ax.axes.get_yaxis().set_visible(False)
    
    for i in other_preds:
        plt.axvline(i, color = 'grey', dashes = [10,10])

    plt.axvline(pred)

    img = io.BytesIO()
    fig.savefig(img, format='png',
                bbox_inches='tight')
    img.seek(0)

    plt.close(fig)

    return base64.b64encode(img.getvalue())

def generate_ui_data(store,other_scores,feat_imp_cols,text_prefix,model,log):

    """ Generate the graphs and tables used by the UI from cached prediction data """

    # Render figure
    fig = render_densplot(store['score'],model['model_props']['scores'],other_scores)
    fig_base64 = fig.decode('utf-8')

    # Components table
    components = pd.DataFrame(store['components']['orig'], index = ['Probability'])
    percentiles = [scipy.stats.percentileofscore(v,components[k].iloc[0]) for k,v in model['model_props']['sub_preds'].items()]

    components = components.transpose()
    components['Probability'] = np.round(components['Probability'],3)
    components['Percentile'] = percentiles

    if 'names' in model:
            components.index = pd.Series(components.index).replace(to_replace=model['names'])

    components = components.style.format({'Probability': '{:.1%}', 'Percentile': '{:.0f}'}).set_properties(**{'text-align': 'center'})
    
    # Set colors

    # Feature importance tables using shapley values
    feat_imp = pd.DataFrame(store['feats'])
    log.debug(feat_imp)
    shaps = feat_imp.loc[:,feat_imp.columns.str.startswith("shap_")]
    feat_imp['mean_shap'] = shaps.mean(axis=1)
    feat_imp['mean_abs_shap'] = shaps.mean(axis=1).abs()
    
    #feat_imp = feat_imp[feat_imp.index not in cat_preds | feat_imp['value'] > 0,:]

    feat_imp = feat_imp.sort_values(by = 'mean_abs_shap', ascending = False)

    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    
    if os.environ['DEV_MODE'] == True:
        feat_imp_table = feat_imp.style
    
    else:
        if 'names' in model:
            feat_imp.index = pd.Series(feat_imp.index).replace(to_replace=model['names'])

        feat_imp_table = feat_imp[feat_imp_cols]

        feat_imp_table = feat_imp_table.style.format({'value': "{:.0f}", 
                                                      'mean_shap': '{:.2f}'}).set_properties(**{'text-align': 'center'})
        feat_imp_table = feat_imp_table.bar(subset=['mean_shap'], align='mid', color=['#5fba7d','#d65f5f'])

        
        
    return {
        'fig_base64':fig_base64,
        'components':components,
        'feat_imp_table':feat_imp_table
    }

def append_secret(id,bts = 32):
    """ Generate a secret and append it to a value to make it difficult to guess """
    secret = secrets.token_urlsafe(bts)
    secret_id = str(id) + '_' + secret
    return secret_id

def generate_trialid(ids):
    """ Generate an identifier to keep track of which records were compared to each other """
    dt = datetime.now()
    dt = int(dt.strftime('%Y%m%d%H%M%S'))
    id_str = '_'.join(ids)
    out = f'{dt}_{id_str}_{secrets.token_urlsafe(2)}'
    return out

def sub_utf8_ascii(text_series, symbol = ''):
    """ Remove non-ascii characters and replace spaces with underscores """
    out_list = []
    space_re = re.compile(' ')
    alnum_re = re.compile('[^a-zA-Z0-9_]')
    for t in text_series:
        out = space_re.sub('_', t)
        out = alnum_re.sub(symbol, out)
        out_list.append(out)
    
    return pd.Series(out_list)

def string_to_dummies(df,string_col):
    dummies = df[string_col].str.get_dummies()
    dummies.columns = string_col + '_' + sub_utf8_ascii(dummies.columns)
    return(dummies)

def generate_cms(preds,labels,thresholds):
    """
    Generate confusion matrices for a list of threshold values and returns a dict
    of arrays, for a 2 by 2 matrix, this is in the format [[TN,FP],[FN,TP]]
    """
    from sklearn.metrics import confusion_matrix

    cm = {}
    for i in thresholds:
        pred_bin = [p>=i for p in preds]
        cm[i] = confusion_matrix(labels,pred_bin).tolist()

    return cm

def parse_json_data(inputData,model,cat_preds,log):
    """
    Takes dictionary of input data, request['data'] 
    Returns a dictionary of features
    """
    # Load features with all missing values (uses features from first model, assumes all are built on the same dataset)
    data = pd.DataFrame({k:np.nan for k in model['model_props']['feature_names']}, index=[0])
    input_df = pd.DataFrame(inputData, index=[0])
    log.debug(inputData)

    for k,v in inputData.items():
        if k not in ['disp_time','complaint_group','region']:
            data[k] = v


    # A bit of feature engineering
    # dates
    case_dt = pd.to_datetime(input_df['disp_time'],format='%Y-%m-%d %H:%M:%S')
    data = data.assign(
        # Number of days since jan 1 1970 (unix time)
        disp_date = (case_dt - pd.Timestamp("1970-01-01", tz='UTC')) // pd.Timedelta(days=1),
        disp_hour = case_dt.dt.hour.astype(int),
        disp_month = case_dt.dt.month.astype(int))

    for i in cat_preds:
        data.loc[:,data.columns.str.startswith(i)] = 0
        data.update(string_to_dummies(input_df,i))

    log.debug(data)
    return data

# Modelling functions --------------------------------------------------

def predict_instrument(model,new_data,cat_preds,log):

    """ Function to generate risk scores. The approach of 
    estimating several models and using a normalized average
    to represent overall patient risks is investigated in 
    this article: https://doi.org/10.1371/journal.pone.0226518 """

    # Set up data frame to store interemediate steps.
    p = pd.DataFrame(index = model['model_props']['names'], columns = ['orig','trans','scale'])

    #Make list of non-missing, documented data
    doc_feats = new_data.transpose()
    doc_feats.columns = ['value']
    doc_feats = doc_feats[doc_feats.value.notnull()]
    
    for i in cat_preds:
        doc_feats = doc_feats[(doc_feats.value > 0) | ~(doc_feats.index.str.startswith(i))]


    #log.debug(doc_feats)
    feat_imp = pd.DataFrame(model['model_props']['feat_props']['gain'], index=['gain']).transpose()

    # Loop over each model included in the risk score
    for k,v in model['models'].items():

        new_dmatrix = xgboost.DMatrix(new_data)
        # Estimate probability
        pred = float(v.predict(new_dmatrix))
        shap = v.predict(new_dmatrix, pred_contribs=True)
        shap = dict(zip(model['model_props']['feature_names'],shap[0][:-1])) # remove bias term
        shap = pd.DataFrame(shap,index=[f'shap_{k}']).transpose()
        feat_imp = feat_imp.join(shap)

        # Transform estimate as appropriate (logit for us by default)
        trans_pred = trans_fun(pred, model['model_props']['scale_params']['trans'])

        # Extract scaling paramters from model object
        scale = model['model_props']['scale_params']['pre_trans'][k]['scale']
        center = model['model_props']['scale_params']['pre_trans'][k]['center']

        # Save intermediate predcitions to df
        p['orig'][k] = pred
        p['trans'][k] = trans_pred

        # Scale and center the estimates based on the population mean and stdev (z-scoring)
        p['scale'][k] = (trans_pred - center) * scale
    
    
    doc_feats = doc_feats.join(feat_imp)

    # Generate final score (weighted average), save score and intermediate predictions as dict (dfs don't like being serialized)
    out = {"score": np.average(list(p['scale']),
           weights=list(model['model_props']['scale_params']['out_weights'].values())), #TODO: Don't assume these are in the right order.
           "components":p.to_dict(),
           "feats":doc_feats.to_dict()}

    return out

def trans_fun(p,trans):

    """ Helper function to apply transformations when combining model predictions""" 

    if trans == 'log':
        out = np.log(p)
    elif trans == 'logit':
        out = np.log(p/(1-p))

    return out

def xgb_cv_fun(
    dmatrix,
    max_estimators,
    subsample, 
    colsample_bytree,
    max_depth,
    min_child_weight,
    learning_rate,
    gamma
    ):
    """
    Function to be fed to bayesian optimization process
    """
    #Discretize a couple of hyperparameters (Uprincipled!)
    params = {
        'subsample' : subsample, 
        'colsample_bytree' : colsample_bytree,
        'max_depth' : int(max_depth),
        'min_child_weight' : min_child_weight, 
        'learning_rate' : learning_rate,
        'gamma' : gamma,
        'objective' : 'binary:logistic'
        }

    #Time process
    start = time.time()
    #Fit model, note using the non-sklearn API here
    fit = xgboost.cv(
        params,
        dmatrix,
        num_boost_round = max_estimators,
        stratified = True,
        nfold = 5,
        early_stopping_rounds = 20,
        metrics = 'auc'
        )

    stop = time.time()
    #Generate dict for return
    fit_props = {}
    fit_props['train_time']  = stop - start
    fit_props['train_score'] = fit['train-auc-mean'].iloc[-1]
    fit_props['val_score'] = fit['test-auc-mean'].iloc[-1]
    fit_props['n_estimators'] = len(fit.index)

    return params, fit_props

def bayes_opt_xgb(
    log,
    dmatrix,
    opt_fun,
    opt_rounds,
    init_rounds,
    params_ranges,
    max_estimators
    ):

    """ Function to perform bayesian optimization of model hyperparameters for xgboost models """

    # Instatiate BO object, utilty function, and log
    opt_xgb = BayesianOptimization(opt_fun, params_ranges)
    utility = UtilityFunction(kind="ei", kappa=2.5, xi=0.0)
    log_params = {}
    
    # Manually loop through opimization process (wanted better logs than the built-in funtion)
    for _ in range(opt_rounds):
        # Start selecting non-random points after 10 rounds
        if _ < init_rounds: 
            np.random.seed()
            next_point = {key: np.random.uniform(value[0],value[1]) for key, value in params_ranges.items()}
        else: 
            next_point = opt_xgb.suggest(utility)
        # Fit xgb model with selected hyperparams
        opt_params, fit_props = opt_fun(
            dmatrix = dmatrix,
            max_estimators = max_estimators,
            **next_point
            )
        
        target = fit_props['val_score']
        # Register results to BO object
        opt_xgb.register(params=next_point, target=target)
        # Print to keep user updated and save to log
        log.info(str(_) + str(fit_props) + str(opt_params))
        log_params.update({target:{'params':opt_params,'fit_props':fit_props}})
    
    return log_params

def get_model_props(
    fits,
    data,
    out_weights,
    instrument_trans,
    log,
    threshold_digits = 2,
    limit_gh = True,
    ):

    """ Function to generate a dict containing the properties of a given set of model fits """

    preds = {}
    trans_preds = {}
    scale_preds = {}
    scale = {}
    feat_gains = {}
    confusion_matrices = {}

    if len(data['test']['labels'].index) == 0:
        log.warn("Warning! No test data found! Generating model properties on training data, very likely to over-estimate model performance.")
        eval_data = data['train']
    else:
        eval_data = data['test']

    for name, values in eval_data['labels'].items(): 
        # For each label....
        test_dmatrix = xgboost.DMatrix(eval_data['data'], label = values)
        # Predict raw score
        preds[name] = [float(i) for i in list(fits[name].predict(test_dmatrix))]
        
        trans_preds[name] = [trans_fun(x, instrument_trans) for x in preds[name]]

        scale[name] = {"center": float(np.mean(trans_preds[name])),
                    "scale": float(np.std(trans_preds[name]))}

        scale_preds[name] = [(x-scale[name]['center'])/scale[name]['scale'] for x in trans_preds[name]]

        feat_gains[name] = fits[name].get_score(importance_type = 'gain')
    
    # Calculate overall scores in test data
    scores = []
    ld = [len(value) for key, value in scale_preds.items()]
    for i in range(0,max(ld)):
        p = []
        for key in scale_preds:
            p.append(scale_preds[key][i])
        scores.append(np.average(p, weights=list(out_weights.values())))

    # Generate feature importance and median value df
    feat_name_lists = [list(v.keys()) for v in feat_gains.values()]
    feat_gain_lists = [list(v.values()) for v in feat_gains.values()]

    gain_df = pd.DataFrame({"name" : list(itertools.chain.from_iterable(feat_name_lists)),
                            "gain" : list(itertools.chain.from_iterable(feat_gain_lists))})

    gainsum_df = gain_df.groupby(['name']).sum()

    median_values_df = pd.DataFrame({'median' : data['train']['data'].median(axis = 0, skipna = True)})
    feat_props = median_values_df.join(gainsum_df).dropna()

    threshold_value = 10**-threshold_digits
    # Generate confusion matrices for various score thresholds
    threshold_list = list(np.arange(
            min(scores),
            max(scores),
            threshold_value))
    
    threshold_list = [round(i,threshold_digits) for i in threshold_list]

    confusion_matrices = {}

    for name, values in eval_data['labels'].items():
        confusion_matrices[name] = generate_cms(scores,values,threshold_list)

    # Generate dictionary to be returned
    model_props = {
        'names': list(data['train']['labels'].columns),
        'feature_names': list(data['train']['data'].columns),
        'feat_props': feat_props.to_dict(),
        'scale_params': {
            'pre_trans':scale,
            'trans': instrument_trans,
            'out_weights': out_weights
        },
        'confusion_matrices': confusion_matrices,
        #Add some noise here... Just to be safe.
        'scores': list(np.add(scores,np.random.normal(0,0.0001,len(scores)))),
        'sub_preds': {k: list(np.add(v,np.random.normal(0,0.0001,len(v)))) for k, v in preds.items()}
            }
    
    # 
    if limit_gh and len(model_props['scores']) > 500000:
        # Github as a 100 mb limit on filesizes which this exceed when there are about a million observations...
        # Not necessary to have so much data for calculating percentiles and such, so subsample if necessary
        log.warn("Limiting number of observations in model_props to reduce file size")
        model_props['scores'] = random.sample(model_props['scores'], 500000)
        model_props['sub_preds'] = {k: random.sample(v, 500000) for k, v in model_props['sub_preds'].items()}


    return model_props

# Parsing functions ------------------------------------------------------

def separate_flat_data(full_df,label_dict,predictor_list, inclusion_list, test_list):

    """ generate composite labels and separate flat data """

    label_df = pd.DataFrame()

    for lab, comps in label_dict.items():
        label_df[lab] = full_df[comps].max(axis=1)

    data_df = full_df[predictor_list + inclusion_list + test_list]


    return label_df, data_df

def parse_export_data(code_dir, raw_data_paths, inclusion_criteria,test_criteria, label_dict, predictors, text_col, key_table, filter_str, log):

    """ Parse raw export data into a clean format for further processing """

    data_paths = {k:f'{code_dir}/{v}' for k,v in raw_data_paths.items()}

    full_df = pd.read_csv(data_paths['export'],index_col='id',na_values='NA')
    label_df, data_df = separate_flat_data(full_df,label_dict,predictors,inclusion_criteria,test_criteria)

    # A bit of feature engineering
    # dates
    case_dt = pd.to_datetime(data_df['disp_time'],format='ISO8601',utc = True)
    data_df = data_df.assign(
        # Number of days since jan 1 1970 (unix time)
        disp_date = (case_dt - pd.Timestamp("1970-01-01", tz='UTC')) // pd.Timedelta(days=1),
        disp_hour = case_dt.dt.hour.astype(int),
        disp_month = case_dt.dt.month.astype(int))
    data_df = data_df.drop('disp_time',axis=1)

    data_df = data_df.join(string_to_dummies(data_df,'complaint_group'))
    data_df = data_df.join(string_to_dummies(data_df,'region'))
    data_df = data_df.drop(['region','complaint_group'],axis = 1)

    return data_df, label_df

def clean_data(code_dir, raw_data_paths, clean_data_paths, full_name_path, overwrite_data, inclusion_criteria, test_criteria,label_dict, predictors, text_col, key_table, filter_str, log):
    
    """ Try to load clean data or parse raw export data if necessary, then generate final test/train data. """

    full_paths = {k:f'{code_dir}/{v}' for k,v in clean_data_paths.items()}
    paths_exist = {k:os.path.exists(v) for k,v in full_paths.items()}

    if all(paths_exist.values()) and not overwrite_data:
        log.info("Clean data found! Loading...")
        label_df = pd.read_csv(full_paths['clean_labels'], index_col='caseid')
        data_df = pd.read_csv(full_paths['clean_data'], index_col='caseid')

    else:
        log.warning("Parsing export data....")
        data_df, label_df = parse_export_data(
            code_dir, 
            raw_data_paths, 
            inclusion_criteria, 
            test_criteria,
            label_dict, 
            predictors,
            text_col,
            key_table, 
            filter_str, 
            log
            )

        label_df.to_csv(full_paths['clean_labels'])
        data_df.to_csv(full_paths['clean_data'])
    # Generate dict of pretty names of predictors and labels for display in ui
    # Load pretty names for display in UI
    if not os.path.exists(full_name_path):
        log.info("No pretty names found! Generating...")
        names = generate_names(full_name_path,data_df)
        with open(full_name_path, "w") as f:
                json.dump(names,f,indent=4,ensure_ascii=False)

    return {'data':data_df,
            'labels':label_df}

def split_data(data, test_cutoff_ymd, test_sample, test_criteria, inclusion_criteria):
    
    # Apply inclusion criteria before splitting

    for i in inclusion_criteria:
        inclobs = data['data'][i].eq(1)
        print("excluding",len(data['data'].index)-np.sum(inclobs),i)
        data['data'] = data['data'][inclobs]
        data['labels'] = data['labels'][inclobs]

    data['data'] = data['data'].drop(inclusion_criteria,axis=1)

    cutoff_date_epoch = calendar.timegm(time.strptime(test_cutoff_ymd, "%Y%m%d")) / 86400

    
    valid_ids = data['data'].index[data['data'].disp_date >= cutoff_date_epoch]
    print(f"{len(valid_ids)} observations after {test_cutoff_ymd}")

    for i in test_criteria:
        crit_incl = data['data'][i].eq(1)
        criteria_ids = data['data'].index[crit_incl]
        valid_ids = [j for j in valid_ids if j in criteria_ids]
        print(f"Keeping {len(valid_ids)} observations with {i} for testing")
    
    data['data'] = data['data'].drop(test_criteria,axis=1)

    test_ids = list(random.sample(list(valid_ids),int(len(valid_ids)*test_sample)))
    
    print(f"{len(test_ids)} test observations after random sampling")

    out_data = {
        'train':{
            'data':data['data'][~data['data'].index.isin(test_ids)],
            'labels':data['labels'][~data['labels'].index.isin(test_ids)]
        },
        'test':{
            'data':data['data'][data['data'].index.isin(test_ids)],
            'labels':data['labels'][data['labels'].index.isin(test_ids)]
        }
    }

    return out_data

def generate_names(code_dir, data):

    """ Generate a dict containing user-friendly variable names for display in the UI.
    This only generates a json file with the correct keys, needs to be filled in manually."""
    
    name_path = f'{code_dir}/models/pretty_names.json'

    if not os.path.exists(name_path):

        out_dict = dict(zip(data.columns,data.columns))

        return out_dict
    
    pass

def nemsis3_to_vitals_dict(nemsis_xml):

    n3_map = {
        'disp_gender':{
        '9906001':1, # Female
        '9906003':0 # Male
        },
        'disp_cats':{ # A first stab at this.. Non-matches will assume a missing category #TODO: Check how to handle multiple matches, our categories are more specific...
            '2301001':'disp_cats_Buk_flanksmrta|disp_cats_Diarr|disp_cats_Mag_tarmbldning',# Abdominal Pain/Problems
            '2301003':'disp_cats_Allergisk_reaktion',# Allergic Reaction/Stings
            '2301005':None,# Animal Bite
            '2301007':None,# Assault
            '2301009':None,# Automated Crash Notification
            '2301011':'disp_cats_Ryggsmrta',# Back Pain (Non-Traumatic)
            '2301013':'disp_cats_Andningsbesvr',# Breathing Problem
            '2301015':'disp_cats_Brnnskada',# Burns/Explosion
            '2301017':'disp_cats_CBRN|disp_cats_Kemisk_exponering',# Carbon Monoxide/Hazmat/Inhalation/CBRN
            '2301019':'disp_cats_Hjrtstopp',# Cardiac Arrest/Death
            '2301021':'disp_cats_Brstsmrta',# Chest Pain (Non-Traumatic)
            '2301023':'disp_cats_Luftvgsbesvr',# Choking #TODO: Technically... Got to check this one
            '2301025':'disp_cats_Kramper',# Convulsions/Seizure
            '2301027':'disp_cats_Blodsocker_hgt|disp_cats_Blodsocker_lgt',# Diabetic Problem 
            '2301029':'disp_cats_Elektrisk_skada',# Electrocution/Lightning
            '2301031':'disp_cats_gon',# Eye Problem/Injury
            '2301033':None,# Falls
            '2301035':'disp_cats_Brand|disp_cats_Rkexponering',# Fire
            '2301037':'disp_cats_Huvudvrk',# Headache
            '2301039':'disp_cats_Planerad',# Healthcare Professional/Admission
            '2301041':'disp_cats_ICD|disp_cats_Rytmrubbning|disp_cats_Pacemaker',# Heart Problems/AICD
            '2301043':'disp_cats_Hypertermi|disp_cats_Hypotermi|disp_cats_Kldskada|disp_cats_Vrmeslag',# Heat/Cold Exposure
            '2301045':'disp_cats_Srskada',# Hemorrhage/Laceration
            '2301047':None,# Industrial Accident/Inaccessible Incident/Other Entrapments (Non-Vehicle)
            '2301049':None,# Medical Alarm
            '2301051':'disp_cats_Annat',# No Other Appropriate Choice
            '2301053':'disp_cats_Intoxfrgiftning',# Overdose/Poisoning/Ingestion
            '2301055':None,# Pandemic/Epidemic/Outbreak
            '2301057':'disp_cats_Graviditet|disp_cats_Frlossning',# Pregnancy/Childbirth/Miscarriage
            '2301059':'disp_cats_Psykiska_besvr|disp_cats_Vldhotsuicidhot',# Psychiatric Problem/Abnormal Behavior/Suicide Attempt
            '2301061':'disp_cats_Allmn_vuxen|disp_cats_Allmn_ldring|disp_cats_Allmn_barn',# Sick Person
            '2301063':None,# Stab/Gunshot Wound/Penetrating Trauma
            '2301065':None,# Standby
            '2301067':'disp_cats_Stroke|disp_cats_Talpverkan|disp_cats_Sensoriskt_motoriskt_bortfall',# Stroke/CVA
            '2301069':'disp_cats_Trafikolycka|disp_cats_Sjolycka|disp_cats_Flygolycka',# Traffic/Transportation Incident
            '2301071':'disp_cats_Planerad',# Transfer/Interfacility/Palliative Care
            '2301073':'disp_cats_Trauma',# Traumatic Injury
            '2301075':None,# Well Person Check
            '2301077':'disp_cats_Snkt_vakenhet|disp_cats_Svimning|disp_cats_Yrsel',# Unconscious/Fainting/Near-Fainting
            '2301079':None,# Unknown Problem/Person Down
            '2301081':'disp_cats_Drunkningstillbud|disp_cats_Dykeriolycka',# Drowning/Diving/SCUBA Accident
            '2301083':'disp_cats_Planerad',# Airmedical Transport
            '2301085':'disp_cats_Snkt_vakenhet|disp_cats_Frvirring',# Altered Mental Status
            '2301087':None,# Intercept
            '2301089':'disp_cats_Illamende',# Nausea
            '2301091':'disp_cats_Krkning'# Vomiting
        },
        'disp_prio':{ # These are... Kinda right
            '2305001':1,#Critical
            '2305003':2,#Emergent
            '2305005':3,#Lower Acuity
            '2305007':4#Non-Acute [e.g., Scheduled Transfer or Standby]
        }
    }

    """
    Additional dispatch categories not matched to NEMSIS:
    "disp_cats_Arm_bensymtom_ej_trauma": "Arm/leg symptoms",
    "disp_cats_Blod_i_urin": "Blood in urine",
    "disp_cats_Blodig_upphostning": "Blood in sputum",
    "disp_cats_Feber": "Fever",
    "disp_cats_Frtskada": "Chemical burn",
    "": "Confusion",
    "disp_cats_Hallucination": "Hallucination",
    "disp_cats_Halsont": "Throat pain",
    "disp_cats_Infektion": "Infection",
    "disp_cats_Ns_svalgbldning": "Nosebleed",
    "disp_cats_Ormbett": "Snakebite",
    "disp_cats_Urinkateterstopp": "Urinary catheter blockage",
    "disp_cats_Urinstmma": "Dysuria",
    "disp_cats_Urogenitala_besvr": "Urogenital issue",
    "disp_cats_Vaginal_bldning": "Vaginal bleed",
    "": "Dizziness",
    """

    indata = xmltodict.parse(nemsis_xml)['EMSDataSet']['Header']['PatientCareReport']

    # Use pcr number for id if sent, otherwise generate a random string
    if indata['eRecord']['eRecord.01']:
        identifier = indata['eRecord']['eRecord.01']
    else:
        identifier = secrets.token_urlsafe(6)

    n3 = {
        'region':indata['eResponse']['eResponse.AgencyGroup'].get('eResponse.01'),
        'disp_created':indata['eTimes'].get('eTimes.02'), 
        'disp_age':indata['ePatient']['ePatient.AgeGroup'].get('ePatient.15'), 
        'disp_gender':indata['ePatient'].get('ePatient.13'),
        'disp_cats':indata['eDispatch'].get('eDispatch.01'),
        'disp_prio':indata['eDispatch'].get('eDispatch.05'),
        # Get first set of vitals
        'eval_breaths': indata['eVitals']['eVitals.VitalGroup'][0].get('eVitals.14'),
        'eval_spo2': indata['eVitals']['eVitals.VitalGroup'][0].get('eVitals.12'),
        'eval_sbp': indata['eVitals']['eVitals.VitalGroup'][0]['eVitals.BloodPressureGroup'].get('eVitals.06'),
        'eval_pulse':indata['eVitals']['eVitals.VitalGroup'][0]['eVitals.HeartRateGroup'].get('eVitals.10'),
        'eval_avpu':indata['eVitals']['eVitals.VitalGroup'][0].get('eVitals.26'),
        'eval_temp':indata['eVitals']['eVitals.VitalGroup'][0]['eVitals.TemperatureGroup'].get('eVitals.24') # Note that NEMSIS uses celcius for body temperature like other civilized people!
    }

    # Apply transformations
    # Reformat dispatch date
    n3['disp_created'] = n3['disp_created'].split('+')[0].replace('T', ' ')
    # Set age to 0 if age documented as other than years (though perhaps some crews document 1-2 year olds in terms of months? Should revisit this)
    if indata['ePatient']['ePatient.AgeGroup']['ePatient.16'] in ['2516001','2516003','2516005','2516007']:
        indata['ePatient']['ePatient.AgeGroup']['ePatient.15'] = 0
    n3['disp_gender'] =  n3_map['disp_gender'].get(n3['disp_gender'])
    n3['disp_cats'] =  n3_map['disp_cats'].get(n3['disp_cats'])
    n3['disp_prio'] =  n3_map['disp_prio'].get(n3['disp_prio'])

    outdict = {}
    outdict[identifier] = n3
    return outdict
# Text parsng -------------------------------------------------------

def ngrams(words, n):
    # Nice solution https://stackoverflow.com/questions/17531684/n-grams-in-python-four-five-six-grams
    d = collections.deque(maxlen=n)
    d.extend(words[:n])
    words = words[n:]
    out_list = []
    for window, word in zip(itertools.cycle((d,)), words):
        out_list.append('_'.join(window))
        d.append(word)
    
    return out_list

def process_text_token(text,max_ngram,text_prefix,stopword_set = None):
    out_tokens = []
    for i in text:
        t = ' '.join([word.lower() for word in re.findall(r"[\w]+", str(i))]).split()
        t = [x for x in t if not x.isdigit()]
        if stopword_set:
            t = [x for x in t if x not in stopword_set]

        allgrams = []
        for i in range(max_ngram):
            allgrams = allgrams + ngrams(t,i+1)

        if text_prefix:
            allgrams = [text_prefix + i for i in allgrams]

        out_tokens.append(allgrams)
    return out_tokens

def get_count_dict(in_list):
    out_dict = {}
    for i in in_list:
        if i in out_dict.keys():
            out_dict[i] += 1
        else:
            out_dict[i] = 1
    return out_dict

def get_stopword_set(stopword_url,neg_set):
    
    response = requests.get(stopword_url)
    stopword_list = response.text.splitlines()
    stopword_set = set([x for x in stopword_list if x not in neg_set])

    return stopword_set 

def parse_text_to_bow(text_df, max_ngram, text_prefix = None, min_terms = None, stopword_set = None, term_list=None,log = None):

    text_df['tokens'] = process_text_token(
        text_df['FreeText'],
        max_ngram, 
        text_prefix,
        stopword_set
        )
    
    if term_list is not None:
        terms = term_list
    else:
        corpus_list = list(itertools.chain.from_iterable(text_df['tokens']))
        corpus_dict = get_count_dict(corpus_list)
        if min_terms:
            corpus_dict = {k:v for k,v in corpus_dict.items() if v > min_terms}
        terms = corpus_dict.keys()

    full_term_dict = {}
    for i in text_df.index:
        term_list = text_df['tokens'][i]
        term_dict = dict(zip(terms,[0]*len(terms)))
        term_dict.update(get_count_dict(term_list))
        term_dict = {k:v for k,v in term_dict.items() if k in terms}
        full_term_dict[i] = term_dict

    bow_df = pd.DataFrame(full_term_dict).fillna(0).transpose()

    return bow_df