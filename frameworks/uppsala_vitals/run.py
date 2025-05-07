
import logging

from frameworks.uppsala_vitals.main import Main

#TODO: Write shell script to automate model updating

if __name__ == '__main__':

    code_dir = "frameworks/uppsala_vitals"

    # Instantiate logger
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    
    # Instantiate the class to train and serialize models.
    Main(
        code_dir = code_dir,
        log = logger, 
        cache = None,
        init_rounds = 10, #NOTE: Low for testing, use maybe 10/40/400
        opt_rounds = 40, 
        max_estimators = 400,
        overwrite_models = True,
        overwrite_data = True,
        test_cutoff_ymd='20230601',
        test_sample=0.5
        )