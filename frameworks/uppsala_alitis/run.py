
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
        init_rounds = 10,
        opt_rounds = 40,
        max_estimators = 400,
        update_models = True,
        overwrite_data = False,
        train_start_ymd = '20160101',
        test_cutoff_ymd = '20230601',
        test_end_ymd = '20240101',
        test_sample = 0.5,
        criteria_weight = 0.1,
        date_weight = 0.5
        )