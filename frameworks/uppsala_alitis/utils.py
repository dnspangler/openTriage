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

    plt.axvline(pred, color = 'red')

    return fig

def fig_to_base64(fig):
    """ Convert matplotlib figure to a data string """
    img = io.BytesIO()
    fig.savefig(img, format='png',
                bbox_inches='tight')
    img.seek(0)

    return base64.b64encode(img.getvalue())

def generate_ui_data(store,other_scores,feat_imp_cols,text_prefix,model,log):

    """ Generate the graphs and tables used by the UI from cached prediction data """

    score = store['score']

    # For the randomized trial, we need to blind dispatchers to the order of the calls not selected for first dispatch in case these recur in further comparisons
    # This code ensures that the plot is identical regardless of which specific prediction is examined, and the score with the highest risk is always highlighted

    other_scores = [score] + other_scores
    score = max(other_scores)

    # Render figure
    fig = render_densplot(score,list(model['model_props']['scores'].values()),other_scores)
    fig_encode = fig_to_base64(fig)
    fig_base64 = fig_encode.decode('utf-8')

    # Components table
    components = pd.DataFrame(store['components']['orig'], index = ['pct_risk'])
    percentiles = [scipy.stats.percentileofscore(v,components[k].iloc[0]) for k,v in model['model_props']['sub_preds'].items()]
    components = components.transpose()
    components['percentile'] = percentiles

    if 'names' in model:
            components.index = pd.Series(components.index).replace(to_replace=model['names'])
    
    components.columns = ['Abs. Risk',"Percentil"]

    components = components.style.format({'Abs. Risk': "{:.0%}", 'Percentil': '{:.0f}'}).set_properties(**{'text-align': 'center'})
    
    # Set colors

    # Feature importance tables using shapley values
    feat_imp = pd.DataFrame(store['feats'])
    #log.debug(feat_imp)
    shaps = feat_imp.loc[:,feat_imp.columns.str.startswith("shap_")]
    feat_imp['mean_shap'] = shaps.mean(axis=1)
    feat_imp['mean_abs_shap'] = shaps.mean(axis=1).abs()
    feat_imp = feat_imp.sort_values(by = 'mean_abs_shap', ascending = False)

    #TODO: Get pretty names if available

    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    
    if os.environ['DEV_MODE'] == True:
        feat_imp_table = feat_imp.style
    
    else:
        if 'names' in model:
            feat_imp = feat_imp[feat_imp.index.isin(model['names']) | feat_imp.index.str.startswith(text_prefix)]
            feat_imp.index = pd.Series(feat_imp.index).replace(to_replace=model['names'])

        #Show all positive answers and negtive answers with high shap values
        feat_imp_table = feat_imp[(feat_imp.mean_abs_shap > 0.02) | (feat_imp.value != 0)][feat_imp_cols]

        feat_imp_table = feat_imp_table.style.format({'value': "{:.0f}", 'mean_shap': '{:.2f}'}).set_properties(**{'text-align': 'center'})
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

def generate_tokens(key_table):

    """ Generate ascii tokens and clean names in key table for each possible question/answer combo"""

    # Generate tokens corresponding to each possible question/answer combination

    # Note that in general, for model features we conform to the prefixing 
    # scheme used in this article: https://doi.org/10.1371/journal.pone.0226518

    cat_token = sub_utf8_ascii('disp_cat_' + key_table['category_name'])

    qa_token = sub_utf8_ascii('disp_q_' + key_table['question_name'] + '_' + key_table['answer_name'])

    qa_name = key_table['questiongroup_name'] + ': ' + key_table['question_name'] + ' - ' + key_table['answer_name']

    # Remove positive answers from pretty names (these are implied)
    pos_re = re.compile(' - Ja')
    remove_pos = lambda x: pos_re.sub('', x)
    qa_name = list(map(remove_pos, qa_name))

    return cat_token, qa_token, qa_name

def parse_json_data(inputData,model,log):
    """
    Takes dictionary of input data, request['data'] 
    Returns a dictionary of features
    """
    # Load features with all missing values (uses features from first model, assumes all are built on the same dataset)

    feats = model['model_props']['feature_names']
    log.debug(len(feats))
    data = pd.DataFrame({k:np.nan for k in feats}, index=[0])


    #log.debug(data)
    #parse unnested values
    data['disp_age'] = inputData['Age']
    data['Priority'] = inputData['Priority']
    data['RecomendedPriority'] = inputData['RecomendedPriority']
    if inputData['Gender'] == 'Male':
        data['disp_gender'] = 0
    elif inputData['Gender'] == 'Female':
        data['disp_gender'] = 1
    #data['disp_postupdate'] = 1
    data['disp_date'] = (date.today() - date(1970, 1, 1)).days
    data['disp_hour'] = datetime.now().hour
    data['disp_month'] = date.today().month
    data['disp_lat'] = inputData['Coord_n']
    data['disp_lon'] = inputData['Coord_e']

    # Process main answers
    ma_dict, unparsed_ma = parse_json_mainanswers(inputData['MainAnswers'],model,log)
    # Process mbs categories
    qa_dict, unparsed_cat = parse_json_categories(inputData['Categories'],model,log)
    #Combine dicts
    qa_dict.update(ma_dict)
    # Transpose the dict to a single-row df and reset the index values
    qa_wide = pd.DataFrame(qa_dict, index=[0])
    # update the full feature set
    data.update(qa_wide)

    #log.debug(qa_wide.transpose())

    return data

def parse_json_mainanswers(mainanswers,model,log):

    """ parses main answers in alitis data file """

    key_table = model['key']
    out_dict = {}
    unparsed_list = []

    for ma in mainanswers:
        questionID = ma['QuestionID'].upper()
        answerID = ma['AnswerID'].upper()

        # Add questions to intermediate table with 0 as documented value and append
        mq_list = key_table[(key_table['QuestionID'] == questionID)].qa_token.drop_duplicates()
        if len(mq_list) > 0:
            mq_dict = dict(zip(mq_list,[0] * len(mq_list)))
            out_dict.update(mq_dict)

        # Add answers to intermediate table with 1 as documented value and append
        ma_list = key_table[(key_table['QuestionID'] == questionID) & 
                            (key_table['AnswerID'] == answerID)].qa_token
        if len(ma_list) > 0:
            ma_dict = dict(zip(ma_list,[1] * len(ma_list)))
            out_dict.update(ma_dict)
        else:
            unparsed = {
                'QuestionID':questionID,
                'AnswerID':answerID
            }
            log.warn("Uparsed Main answer!")
            log.warn(unparsed)
            unparsed_list.append(unparsed)
    
    out_dict.update({'disp_cat_Main':len(mainanswers)})
    
    return out_dict, unparsed_list

def parse_json_categories(categories,model,log,neg_name = "Bedömt utan fynd"):
    
    """ Function to parse question/answer data sent by the Alitis dispatch system into a dict. """

    # Get possible key combinations
    key_table = model['key']

    # Set ids corresponding to negative question group answers
    missreason_ids = key_table[key_table['answer_name'] == neg_name].AnswerID.tolist()
    missreason_ids = [x.lower() for x in missreason_ids]

    #log.debug(missreason_ids)

    unparsed_list = []

    # Generate dict of categories to be updated with quesion/answer combos, and number of answers for each category
    cat_list = list(key_table.cat_token.drop_duplicates())
    out_dict = dict(zip(cat_list,[0] * (len(cat_list))))

    # Generate dict of missing reasons to be updated
    miss_null = key_table[key_table['CategoryID'] == 'Missreason'].qa_token.unique()
    miss_dict = dict(zip(miss_null,[0] * (len(miss_null)+1)))

    out_dict.update(miss_dict)

    # Loop over each catagory selected by the dispatcher
    for category in categories:

        categoryID = category['CategoryID'].upper()
        
        # Track number of questions answered for use as feature
        nqs = 0

        # Loop over each documented reason for missing answers
        for missingReason in category['MissingReasons']:

            questionGroupID = missingReason['QuestionGroupID'].upper()
            missingReasonID = missingReason['MissingReasonID'].upper()
            
            miss_list = key_table[(key_table['AnswerID'] == missingReasonID) & 
                            (key_table['QuestionGroupID'] == questionGroupID)].qa_token.tolist()

            if len(miss_list) == 1:
                out_dict[miss_list[0]] = 1
            else:
                unparsed = {
                    'CategoryID':categoryID,
                    'QuestionGroupID':questionGroupID,
                    'MissingReasonID':missingReasonID
                }
                log.warn("Uparsed missing reason!")
                log.warn(unparsed)
                unparsed_list.append(unparsed)
            
            # Here we handle adding implied negative answers. 
            # Implied negative answers are drawn from two sources: The 
            # documentation of having reviewed all questions in a group and 
            # identifying no positive answers, or documenting an answer to 
            # a question, in which case we assume that all non-documented 
            # answers are negative.

            # if question group marked as investigated with only negative answers...
            if any(missingReason['MissingReasonID'] == id for id in missreason_ids): 

                # Identify all matching tokens in category/question group
                neg_list = key_table[(key_table['CategoryID'] == categoryID) & 
                                       (key_table['QuestionGroupID'] == questionGroupID)].qa_token
                #log.debug("neg group:")
                #log.debug(neg_list)
                if len(neg_list) > 0:
                    # Add all qas in group to intermidiate table with 0 as the documented value
                    neg_dict = dict(zip(neg_list,[0] * (len(neg_list))))
                    # We don't want to overwrite postitive answers, so remove already documented tokens from this list
                    neg_dict = {key: neg_dict[key] for key in neg_dict.keys() if key not in out_dict.keys()}
                    # Add all missing answers to main table
                    out_dict.update(neg_dict)
                else:
                    unparsed = {
                        'CategoryID':categoryID,
                        'QuestionGroupID':questionGroupID,
                        'MissingReasonID':missingReasonID
                    }
                    log.warn("Uparsed negative reason!")
                    log.warn(unparsed)
                    unparsed_list.append(unparsed)

        # Loop over each question
        for question in category['Questions']:
            questionID = question['QuestionID'].upper()

            # Add questions to intermediate table with 0 as documented value and append as above
            q_list = key_table[(key_table['CategoryID'] == categoryID) & 
                                (key_table['QuestionID'] == questionID)].qa_token
            if len(q_list) > 0:
                q_dict = dict(zip(q_list,[0] * (len(q_list))))
                q_dict = {key: q_dict[key] for key in q_dict.keys() if key not in out_dict.keys()}
                out_dict.update(q_dict)
                #log.debug("neg question:")
                #log.debug(out_dict)
            else:
                unparsed = {
                    'CategoryID':categoryID,
                    'QuestionID':questionID
                }
                log.warn("Uparsed question!")
                log.warn(unparsed)
                unparsed_list.append(unparsed)

            # Loop over each answer within the question (Some questions can have multiple positive answers)
            for answer in question['Answers']:
                answerID = answer['AnswerID'].upper()
                multipleChoiceID = answer['MultipleChoiceID'].upper()
                # Append answered questions to main table (will have values of >0)
                qa_df = key_table[
                    (key_table['CategoryID'] == categoryID) & 
                    (key_table['QuestionID'] == questionID) & 
                    (key_table['AnswerID'] == answerID) & 
                    (key_table['MultipleChoiceID'] == multipleChoiceID)
                    ]
                #log.debug("pos answer:")
                #log.debug(qa_list)
                if len(qa_df.index) > 0:
                    qa_dict = dict(zip(qa_df.qa_token,qa_df.value))
                    out_dict.update(qa_dict)
                    # Increment number of answered questions
                    nqs += 1
                else:
                    unparsed = {
                        'CategoryID':categoryID,
                        'QuestionID':questionID,
                        'AnswerID':answerID,
                        'MultipleChoiceID':multipleChoiceID
                    }
                    log.warn("Uparsed answer!")
                    log.warn(unparsed)
                    unparsed_list.append(unparsed)

        # Append the number of questions answered for each category
        cat = key_table[key_table['CategoryID'] == categoryID].cat_token.drop_duplicates().str.cat()
        out_dict.update({cat : nqs})

    #log.debug(out_dict)
    return out_dict, unparsed_list

# Modelling functions --------------------------------------------------

def predict_instrument(model,new_data,log):

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
    #log.debug(doc_feats)
    feat_imp = pd.DataFrame(model['model_props']['feat_props']['gain'], index=['gain']).transpose()

    # Loop over each model included in the risk score
    for k,v in model['models'].items():
        
        new_dmatrix = xgboost.DMatrix(new_data)
        # Estimate probability
        pred = float(v.predict(new_dmatrix))

        # Get SHAP values
        shap = v.predict(new_dmatrix, pred_contribs=True)
        shap = dict(zip(model['model_props']['feature_names'],shap[0][:-1])) # remove bias term
        shap = pd.DataFrame(shap,index=[f'shap_{k}']).transpose()
        feat_imp = feat_imp.join(shap)

        # Transform estimate as appropriate (logit for us by default)
        if model['model_props']['scale_params']['trans']:
            trans_pred = trans_fun(pred, model['model_props']['scale_params']['trans'])
        else:
            trans_pred = pred


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
    objective,
    metric,
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
        'objective' : objective
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
        metrics = metric
        )

    stop = time.time()
    #Generate dict for return
    fit_props = {}
    fit_props['train_time']  = stop - start
    fit_props['train_score'] = fit[f'train-{metric}-mean'].iloc[-1]
    fit_props['val_score'] = fit[f'test-{metric}-mean'].iloc[-1]
    fit_props['n_estimators'] = len(fit.index)

    return params, fit_props

def bayes_opt_xgb(
    log,
    dmatrix,
    opt_fun,
    opt_rounds,
    init_rounds,
    params_ranges,
    max_estimators,
    objective,
    metric
    ):

    """ Function to perform bayesian optimization of model hyperparameters for xgboost models """

    # Instatiate BO object, utilty function, and log
    opt_xgb = BayesianOptimization(opt_fun, params_ranges)
    utility = UtilityFunction(kind="ei", kappa=2.5, xi=0.0)
    log_params = {}
    
    # Manually loop through optimization process (wanted better logs than the built-in function)
    for _ in range(opt_rounds):
        # Start selecting non-random points after n rounds
        if _ < init_rounds: 
            np.random.seed()
            next_point = {key: np.random.uniform(value[0],value[1]) for key, value in params_ranges.items()}
        else: 
            next_point = opt_xgb.suggest(utility)
        # Fit xgb model with selected hyperparams
        opt_params, fit_props = opt_fun(
            dmatrix = dmatrix,
            max_estimators = max_estimators,
            objective = objective,
            metric = metric,
            **next_point
            )
        
        target = fit_props['val_score']
        # Register results to BO object
        opt_xgb.register(params=next_point, target=target)
        # Print to keep user updated and save to log
        log.info(str(_) + str(fit_props) + str(opt_params))
        log_params.update({target:{'params':opt_params,'fit_props':fit_props}})
    
    return log_params

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

def get_model_props(
    fits,
    data,
    out_weights,
    instrument_scale,
    instrument_trans,
    log,
    threshold_digits = 2
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
        
        if instrument_trans:
            trans_preds[name] = [trans_fun(x, instrument_trans) for x in preds[name]]
        else:
            trans_preds[name] = preds[name]


        if instrument_scale:
            scale[name] = {"center": float(np.mean(trans_preds[name])),
                           "scale": float(np.std(trans_preds[name]))}
        else:
            scale[name] = {"center": 0,
                           "scale": 1}

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
        if values.nunique() == 2:
            confusion_matrices[name] = generate_cms(scores,values,threshold_list)
        else:
            confusion_matrices[name] = 'More than 2 values'

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
        'scores': dict(zip(eval_data['labels'].index, scores)),
        'sub_preds': {k: v for k, v in preds.items()}
            }

    return model_props

# Parsing functions ------------------------------------------------------

def recur_update(d, u):
    """ clever function for doing recursive dictionary updating from https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recur_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def join_flat_data(qs_df,alit_df):
    
    """ Function to join outcome data from qliksense with predictor data from Alitis db """

    # Check for duplicate IDs
    assert len(qs_df.index) == qs_df.index.nunique(), 'Duplicate IDs in Qliksense data'
    assert len(alit_df.index) == alit_df.index.nunique(), 'Duplicate IDs in Alitis data'

    full_df = qs_df.join(alit_df,how='inner')
    print("alitis:",len(alit_df.index),"qliksense:",len(qs_df.index),"final:",len(full_df.index))
    return full_df

def load_qliksense_data(qs_excel_path,incl_vars):

    """ load and filter data exported from Qliksense """
    qs_df = pd.read_excel(qs_excel_path,index_col='caseid',na_values='-')

    for i in incl_vars:
        inclobs = qs_df[i].eq(1)
        print("excluding",len(qs_df.index)-np.sum(inclobs),i)
        qs_df = qs_df[inclobs]

    return qs_df

def load_alitis_cases_data(alitis_cases_path):

    """ Load and filter flat data exported from Alitis db """
    
    alit_df = pd.read_csv(alitis_cases_path,index_col='caseid')
    
    # Exclude calls with no MBS priority (non medical)
    hasrecprio = alit_df.RecomendedPriority.notnull()
    print("excluding",len(alit_df.index)-np.sum(hasrecprio), "with no recprio")
    alit_df = alit_df[hasrecprio]

    # Exclude calls where the MBS wasn't used
    hasanswers = alit_df.HasAnswers.eq(1)
    print("excluding",len(alit_df.index)-np.sum(hasanswers), "with no answers")
    alit_df = alit_df[hasanswers]

    return alit_df

def separate_flat_data(full_df,label_dict,predictor_list,test_list,inclusion_list,text_col = "FreeText"):

    """ generate composite labels and separate flat data """

    label_df = pd.DataFrame()

    for lab, comps in label_dict.items():
        label_df[lab] = full_df[comps].max(axis=1)

    data_df = full_df[predictor_list + test_list + inclusion_list]

    text_df = pd.DataFrame(full_df[text_col]) # Make sure this doesn't get saved as a series

    return label_df, data_df, text_df

def parse_flat_data(
    qs_path,
    alitis_cases_path
    ):
    """ Load flat data from Qliksense and Alitis """
    if qs_path.endswith('.csv'):
        qs_df = pd.read_csv(qs_path,index_col='caseid',na_values='NA',encoding = "utf-8")
    elif qs_path.endswith('.xlsx'):
        qs_df = pd.read_excel(qs_path,index_col='caseid',na_values='-')
    alit_df = load_alitis_cases_data(alitis_cases_path)

    return join_flat_data(qs_df,alit_df)

def load_alitis_mbs_data(incl_caseids, alitis_answers_path, alitis_neg_groups_path):

    """ Load question/answer data and filter to include only relevant calls """
    
    answers = pd.read_csv(alitis_answers_path,index_col='caseid').fillna('')
    neg_groups = pd.read_csv(alitis_neg_groups_path,index_col='caseid').fillna('')

    # Filter only included calls
    answers = answers[answers.index.isin(incl_caseids)]
    neg_groups = neg_groups[neg_groups.index.isin(incl_caseids)]

    return answers, neg_groups

def get_categories(df,key_table):

    """ Get the number of answers provided for each category and return them as a dict """

    out_dict = {}
    unparsed_dict = {}

    cats = df.groupby(['caseid','CategoryID']).size().reset_index()
    catnames = key_table[['CategoryID','cat_token']].reset_index().drop_duplicates()
    group_cats = cats.merge(catnames,on=['CategoryID'],how='left').groupby('caseid')
    for k,v in group_cats:
        if len(v.index) > 0:
            cat_dict = dict(zip(catnames['cat_token'],[0] * (len(catnames.index)+1)))
            cat_dict.update(dict(zip(v['cat_token'],v[0])))
            if k in out_dict:
                out_dict[k].update(cat_dict)
            else:
                out_dict[k] = cat_dict
        else:
            unparsed_dict[k] = {
                'CategoryID':v['CategoryID']
            }

    return out_dict, unparsed_dict

def get_neg_groups(df,key_table,neg_name = "Bedömt utan fynd"):

    """ Get implied negative answers based on documentation in question groups """

    out_dict = {}
    unparsed_dict = {}
    # Define null missing reasons
    miss_null = key_table[key_table['CategoryID'] == 'Missreason'].qa_token.unique()
    miss_null_dict = dict(zip(miss_null,[0] * (len(miss_null)+1)))

    for k,v in df.iterrows():
        #Add missing reason to dict
        miss_dict = {}
        miss_val = key_table[(key_table['AnswerID'] == v['MissingReasonID']) & 
                            (key_table['QuestionGroupID'] == v['QuestionGroupID'])].qa_token.tolist()
        
        if len(miss_val) == 1:
            miss_dict[miss_val[0]] = 1
        else:
            print(k,v)
            unparsed_dict[k] = {
                'CategoryID':v['CategoryID'],
                'MissingReasonID':v['MissingReasonID'],
                'QuestionGroupID':v['QuestionGroupID']
            }

        # If reason implies negative answers, add them to dict
        if key_table[(key_table['AnswerID'] == v['MissingReasonID'])].answer_name.eq(neg_name).all():
            neg_list = key_table[(key_table['CategoryID'] == v['CategoryID']) & 
                            (key_table['QuestionGroupID'] == v['QuestionGroupID'])].qa_token
            if len(neg_list) > 0:
                neg_dict = dict(zip(neg_list,[0] * (len(neg_list)+1)))
                miss_dict.update(neg_dict)
        else:
            unparsed_dict[k] = {
                'CategoryID':v['CategoryID'],
                'MissingReasonID':v['MissingReasonID'],
                'QuestionGroupID':v['QuestionGroupID']
            }
        
        if k in out_dict:
            out_dict[k].update(miss_dict)
        else:
            out_dict[k] = miss_null_dict.copy()
            out_dict[k].update(miss_dict)

    return out_dict, unparsed_dict

def get_questions(df,key_table,filter_str):
    
    """ Get implied negative answers based on positive answers to multiple-choice question """

    out_dict = {}
    unparsed_dict = {}
    for k,v in df.iterrows():
        q_series = key_table[(key_table['CategoryID'] == v['CategoryID']) & 
                        (key_table['QuestionID'] == v['QuestionID'])].qa_token
        q_series = q_series[~q_series.str.contains(filter_str)]
        if len(q_series) > 0:
            q_dict = dict(zip(q_series,[0] * (len(q_series)+1)))
            if k in out_dict:
                out_dict[k].update(q_dict)
            else:
                out_dict[k] = q_dict
        else:
            unparsed_dict[k] = {
                'CategryID':v['CategoryID'],
                'QuestionID':v['QuestionID']
            }
    
    return out_dict, unparsed_dict

def get_answers(df,key_table,filter_str):

    """ Get positive answers """

    out_dict = {}
    unparsed_dict = {}
    for k,v in df.iterrows():
        qa_df = key_table[(key_table['CategoryID'] == v['CategoryID']) & 
                    (key_table['QuestionID'] == v['QuestionID']) & 
                    (key_table['AnswerID'] == v['AnswerID']) & 
                    ((key_table['MultipleChoiceID'] == v['MultipleChoiceID']))]
        qa_df = qa_df[~qa_df.qa_token.str.contains(filter_str)]
        if len(qa_df) == 1:
            qa_dict = dict(zip(qa_df.qa_token,qa_df.value))
            if k in out_dict:
                out_dict[k].update(qa_dict)
            else:
                out_dict[k] = qa_dict
            
        else:
            unparsed_dict[k] = {
                'CategoryID':v['CategoryID'],
                'QuestionID':v['QuestionID'],
                'AnswerID':v['AnswerID'],
                'MultipleChoiceID':v['MultipleChoiceID'],
                'Text':v['Text']
            }
    
    return out_dict, unparsed_dict

def generate_mbs_dicts(answers,neg_groups,key_table,filter_str):
    """Extracts categories, decision support system answers, 
    and implied negative answers to questions, and combines 
    them into a dictionary"""
    unparsed_dict = {}

    #TODO: Optimize this, either by multithreading or using vectorized functions
    print("parsing categories")

    # Get 
    category_dict, unparsed = get_categories(answers,key_table)
    unparsed_dict = {'categories': unparsed}
    out_dict = category_dict
    
    print("parsing neg groups")

    neg_group_dict, unparsed = get_neg_groups(neg_groups,key_table)
    unparsed_dict = {'neg_groups': unparsed}
    out_dict = recur_update(out_dict, neg_group_dict)

    print("parsing questions")

    neg_q_dict, unparsed = get_questions(answers,key_table,filter_str)
    unparsed_dict = {'neg_questions': unparsed}
    out_dict = recur_update(out_dict, neg_q_dict)

    print("parsing answers")
        
    answer_dict, unparsed = get_answers(answers,key_table,filter_str)
    unparsed_dict = {'answers': unparsed}
    out_dict = recur_update(out_dict, answer_dict)

    return out_dict, unparsed_dict

def na_before_first(series):
    """ Function to replace all values prior to a positive value with NAs """
    ind = series.index.get_loc((series > 0).idxmax())

    if ind > 0:
        series.at[0:ind-1] = np.NaN
    return series

def parse_export_data(code_dir, raw_data_paths, test_criteria, inclusion_criteria, label_dict, predictors, key_table, filter_str, log):

    """ Parse raw export data into a clean format for further processing """

    data_paths = {k:f'{code_dir}/{v}' for k,v in raw_data_paths.items()}

    full_df = parse_flat_data(
        f"{data_paths['outcome_data']}",
        f"{data_paths['mbs_cases']}")
        
    label_df, data_df, text_df = separate_flat_data(full_df,label_dict,predictors,test_criteria,inclusion_criteria)

    # Check for intermediate dictionary of mbs tokens
    unparsed_dict = {}
    try:
        with open(data_paths['mbs_dict']) as f:
            mbs_dict = json.load(f)
    except:
        mbs_dict = {}

    # Only new, unparsed ids

    mbs_ids = np.setdiff1d(list(data_df.index),list(mbs_dict.keys()))
    log.info(f'Parsing {len(mbs_ids)} new mbs entries...')

    if len(mbs_ids) >0:
        answers, neg_groups = load_alitis_mbs_data(
            mbs_ids,
            data_paths['mbs_answers'],
            data_paths['mbs_neg_groups']
            )

        # inefficient code here, takes an hour or two to parse ~100k records)
        new_mbs_dict, unparsed_dict = generate_mbs_dicts(answers,neg_groups,key_table,filter_str)
        mbs_dict.update(new_mbs_dict)

    with open(data_paths['mbs_dict'], 'w') as f:
        json.dump(mbs_dict, f, indent=4)
        
    with open(f'{code_dir}/data/clean/unparsed_dict.json', 'w') as f:
        json.dump(unparsed_dict, f, indent=4)

    mbs_df = pd.DataFrame(mbs_dict).transpose()

    # To avoid considering newly added MBS questions as negative in historical data, set answers prior to first documented positive answers to NA
    # Might want to do this during MBS answer parsing, but the ways I can think of all involve major refactoring or disgusting function side effects.

    qcols = [col for col in mbs_df if col.startswith('disp_q') and not col.endswith('_Ja')]
    mbs_df[qcols] = mbs_df[qcols].apply(na_before_first)  

    log.info(f'Joining {len(mbs_df.index)} mbs and {len(data_df.index)} flat data entries...')
    data_df = data_df.join(mbs_df,how='inner') 

    # A bit of feature engineering
    case_dt = pd.to_datetime(data_df['CreatedOn'],format='ISO8601')
    data_df = data_df.assign(
        # Number of days since jan 1 1970 (unix time)
        disp_date = (case_dt.dt.date - dt.date(1970, 1, 1)).astype('timedelta64[ns]').dt.days,
        disp_hour = case_dt.dt.hour.astype(int),
        disp_month = case_dt.dt.month.astype(int),
        IsValid = [1 if x and not np.isnan(x) else 0 for x in data_df.IsValid])
    data_df = data_df.drop('CreatedOn',axis=1)
    data_df.index.rename('caseid',inplace=True)

    return data_df, label_df, text_df

def clean_data(code_dir, raw_data_paths, clean_data_paths, full_name_path, overwrite_data, update_data, test_criteria, inclusion_criteria, label_dict, predictors, key_table, filter_str, log):
    
    """ Try to load clean data or parse raw export data if necessary. """

    full_paths = {k:f'{code_dir}/{v}' for k,v in clean_data_paths.items()}
    paths_exist = {k:os.path.exists(v) for k,v in full_paths.items()}

    if all(paths_exist.values()) and not overwrite_data:
        log.info("Clean data found! Loading...")
        data_df = pd.read_csv(full_paths['clean_data'], index_col='caseid')
        label_df = pd.read_csv(full_paths['clean_labels'], index_col='caseid')
        text_df = pd.read_csv(full_paths['clean_text'], index_col='caseid')

    if overwrite_data or update_data or not all(paths_exist.values()):
        log.warning("Parsing export data....")
        new_data_df, new_label_df, new_text_df = parse_export_data(
            code_dir, 
            raw_data_paths, 
            test_criteria,
            inclusion_criteria, 
            label_dict, 
            predictors,
            key_table, 
            filter_str, 
            log
            )
        
        if update_data:
            log.warning("Updating data....")
            data_df = new_data_df.combine_first(data_df)
            label_df = new_label_df.combine_first(label_df)
            text_df = new_text_df.combine_first(text_df)
        
        if overwrite_data or not all(paths_exist.values()):
            log.warning("Overwriting data....")
            data_df = new_data_df
            label_df = new_label_df
            text_df = new_text_df

        # Save new data to disk
        data_df.to_csv(full_paths['clean_data'])
        label_df.to_csv(full_paths['clean_labels'])
        text_df.to_csv(full_paths['clean_text'])

    # Generate dict of pretty names of predictors and labels for display in ui
    # Load pretty names for display in UI
    if not os.path.exists(full_name_path):
        log.info("No pretty names found! Generating...")
        names = generate_names(full_name_path,data_df,key_table)
        with open(full_name_path, "w") as f:
                json.dump(names,f,indent=4,ensure_ascii=False)

    return {'data':data_df,
            'labels':label_df,
            'text':text_df}

def split_data(data, train_start_ymd, test_cutoff_ymd, test_end_ymd, test_sample, test_criteria, inclusion_criteria, criteria_weight, date_weight):
    
    # Apply inclusion criteria

    for i in inclusion_criteria:
        inclobs = data['data'][i].eq(1)
        print("excluding",len(data['data'].index)-np.sum(inclobs),i)
        data['data'] = data['data'][inclobs]
        data['labels'] = data['labels'][inclobs]
        data['text'] = data['text'][inclobs]
    
    # Drop variables used to define inclusion criteria
    data['data'] = data['data'].drop(inclusion_criteria,axis=1)

    # Generate training weights

    if criteria_weight:
        assert criteria_weight >= 0 and criteria_weight <= 1,"Criteria weight should be between 0 and 1"
        print("Gererating criteria weights")
        criteria_weights = data['data'][test_criteria].mean(axis=1)
        criteria_weights = criteria_weights.apply(lambda x: (x * (1 - criteria_weight)) + criteria_weight) # weight the range based on the supplied value (date_weight:1)
    else:
        print("Criteria weights not used")
        criteria_weights = pd.Series([1] * len(data['data'].index), index = data['data'].index)

    if date_weight:
        assert date_weight >= 0 and date_weight <= 1,"Date weight should be between 0 and 1"
        print("Gererating date weights")
        date_min = min(data['data']['disp_date'])
        date_max = max(data['data']['disp_date'])
        span = date_max - date_min
        date_weights = data['data']['disp_date'].apply(lambda x:(x - date_min)/span) # transform values to 0-1 range
        date_weights = date_weights.apply(lambda x:(x * (1 - date_weight)) + date_weight)
    else:
        print("Date weights not used")
        criteria_weights = pd.Series([1] * len(data['data'].index), index = data['data'].index)

    train_weights = pd.Series(list(map(lambda x,y: x * y,criteria_weights,date_weights)), index = criteria_weights.index)

    assert min(train_weights) >= criteria_weight * date_weight,"Training weights < lowest possible value"   
    assert max(train_weights) <= 1,"Training weights > 1"
    
    # Generate test/training sets
    
    # Convert to unix epoch (start 19700101)
    start_date_epoch = calendar.timegm(time.strptime(train_start_ymd, "%Y%m%d")) / 86400
    cutoff_date_epoch = calendar.timegm(time.strptime(test_cutoff_ymd, "%Y%m%d")) / 86400
    end_date_epoch = calendar.timegm(time.strptime(test_end_ymd, "%Y%m%d")) / 86400

    # Get indexes of test/train observations
    train_ids = list(data['data'].index[(data['data'].disp_date >= start_date_epoch) & (data['data'].disp_date < cutoff_date_epoch)])
    print(f"{len(train_ids)} observations between {train_start_ymd} and {test_cutoff_ymd}")

    valid_test_ids = data['data'].index[(data['data'].disp_date >= cutoff_date_epoch) & (data['data'].disp_date < end_date_epoch)]
    print(f"{len(valid_test_ids)} observations between {test_cutoff_ymd} and {test_end_ymd}")

    test_ids = list(random.sample(list(valid_test_ids),int(len(valid_test_ids)*test_sample)))
    post_cutoff_train_ids = list([j for j in valid_test_ids if j not in test_ids])
    print(f"Post cutoff observations split {len(test_ids)} test / {len(post_cutoff_train_ids)} train")

    for i in test_criteria:
        crit_incl = data['data'][i].eq(1)
        criteria_ids = data['data'].index[crit_incl]
        test_ids = [j for j in test_ids if j in criteria_ids]
        print(f"{len(test_ids)} observations with {i}")

    train_ids.extend(post_cutoff_train_ids)

    print(f"{len(train_ids)} observations included in training set")
    print(f"{len(test_ids)} observations included in test set")

    # drop variables used to define test criteria
    data['data'] = data['data'].drop(test_criteria,axis=1)

    # Return a dict containing test and training data
    out_data = {
        'train':{
            'data':data['data'][data['data'].index.isin(train_ids)],
            'labels':data['labels'][data['labels'].index.isin(train_ids)],
            'text':data['text'][data['text'].index.isin(train_ids)],
            'weights':train_weights[train_weights.index.isin(train_ids)]
        },
        'test':{
            'data':data['data'][data['data'].index.isin(test_ids)],
            'labels':data['labels'][data['labels'].index.isin(test_ids)],
            'text':data['text'][data['text'].index.isin(test_ids)]
        }
    }

    return out_data

def generate_names(code_dir, data, key_table):

    """ Generate a dict containing user-friendly variable names for display in the UI. 
    Note that any features not in the key_table will have a null value and can be added 
    manually to the JSON file generated based on the dict later."""
    
    name_path = f'{code_dir}/models/pretty_names.json'

    if not os.path.exists(name_path):

        out_dict = dict(zip(data.columns,data.columns))

        qa_dict = dict(zip(key_table['qa_token'],key_table['qa_name']))
        out_dict.update(qa_dict)

        cat_dict = dict(zip(key_table['cat_token'].drop_duplicates(),key_table['category_name'].drop_duplicates()))
        out_dict.update(cat_dict)


        return out_dict

# Text parsng -------------------------------------------------------

# From: https://stackoverflow.com/questions/17531684/n-grams-in-python-four-five-six-grams
def get_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])

def get_m_2_ngrams(input_list, min, max):
    for s in itertools.chain(*[get_ngrams(input_list, k) for k in range(min, max+1)]):
        yield '_'.join(s)

def process_text_token(text,max_ngram,text_prefix,stopword_set = None):

    t = ' '.join([word.lower() for word in re.findall(r"[\w]+", str(text))]).split()
    t = [x for x in t if not x.isdigit()]

    if stopword_set:
        t = [x for x in t if x not in stopword_set]

    allgrams = list(get_m_2_ngrams(t,1,max_ngram))

    if text_prefix:
        allgrams = [text_prefix + i for i in allgrams]

    return allgrams

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

    tokens = {}
    for i, r in text_df.iterrows():
        tokens[i] = process_text_token(
            r['FreeText'],
            max_ngram, 
            text_prefix,
            stopword_set
            )

    #log.debug(tokens)

    if term_list is not None:
        terms = term_list
    else:
        corpus_list = []
        for v in tokens.values():
            corpus_list.extend(v)
        corpus_dict = get_count_dict(corpus_list)
        if min_terms:
            corpus_dict = {k:v for k,v in corpus_dict.items() if v > min_terms}
        terms = corpus_dict.keys()

    full_term_dict = {}
    for k,v in tokens.items():
        term_dict = dict(zip(terms,[0]*len(terms)))
        term_dict.update(get_count_dict(v))
        term_dict = {k:v for k,v in term_dict.items() if k in terms}
        full_term_dict[k] = term_dict

    bow_df = pd.DataFrame(full_term_dict).fillna(0).transpose()
    
    return bow_df

def generate_old_test_feats(df,old_names,log):

    names_add = np.setdiff1d(old_names,list(df.columns))
    names_remove = np.setdiff1d(list(df.columns),old_names)
    names_final = [x for x in old_names if x not in names_remove]

    log.warning("Features added/removed:")
    log.warning(names_add)
    log.warning(names_remove)

    df_out = df.drop(names_remove,axis=1)
    df_out = df_out.reindex(columns=names_final, fill_value=0)

    return df_out
