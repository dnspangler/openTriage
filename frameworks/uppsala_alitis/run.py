
import logging

from frameworks.uppsala_alitis.main import Main

#TODO: Write shell script to automate model updating

if __name__ == '__main__':

    code_dir = "frameworks/uppsala_alitis"

    # Instantiate logger
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)
    
    # Instantiate the class to train and serialize models.
    Main(
        code_dir = code_dir,
        log = logger, 
        cache = None,
        init_rounds = 5, #NOTE: Low for testing, use maybe 10/40/400
        opt_rounds = 20,
        max_estimators = 400,
        update_models = True,
        overwrite_data = True,
        #parse_text = False,
        train_start_ymd = '20160101',
        test_cutoff_ymd = '20200901',
        test_end_ymd = '20201201',
        #label_dict = {
        #    "hosp_critcare" : ["hosp_admit","hosp_30daymort"]},
        #out_weights = {
        #    'hosp_critcare':1
        #},
        test_sample=1,
        test_criteria_weight = False,
        )