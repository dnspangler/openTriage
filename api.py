
import sys
import os
import logging

from gevent.pywsgi import WSGIServer
from importlib import import_module
from pathlib import Path
from flask import request, Response
from flask.logging import default_handler
from flask_api import FlaskAPI, status, exceptions
from redis import Redis
from importlib import import_module

#TODO: Define types for all the things

if __name__ == "__main__":
    # Define link to redis db
    r = Redis(host='redis_db', port=6379, db=0)

    # Instantiate logger
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.addHandler(default_handler)

    # Parse environment variables
    if os.environ['FRAMEWORK'] == '':
        logger.warning("No framework defined, using default framework")
        os.environ['FRAMEWORK'] = os.environ['DEFAULT_FRAMEWORK']

    if os.environ['DEV_MODE'] == '':
        logger.warning("No mode defined, defaulting to development")
        os.environ['DEV_MODE'] = os.environ['DEFAULT_DEV_MODE']
    
    fw_names = os.environ['FRAMEWORK'].split(' ')

    #Set path for framework-specific templates
    template_path = os.path.abspath(f'./lib/templates')

    # Instantiate API
    app = FlaskAPI(__name__,template_folder=template_path)
                            
    # Set up endpoints -------------------------------------------------

    @app.route("/predict/<fw>/", methods=['POST'])
    def predict(fw):
        """
        Return a prediction for provided data.
        """
        #TODO: Will want to implement proper exception handling at some point
        # in the future for the sake of maintainability, but this does the job 
        # of preventing information leakage via the API for now.. Settings?

        logger.info('Start Processing')
        logger.debug(request.headers)
        logger.debug(request.data)
        if request.data:
            try:
                # Pass request payload to input function
                input_data = Main[fw].input_function(request)

            except Exception as e:
                logger.exception("Error upon parsing input data:")
                return f"Server error upon predicting model: {e}", status.HTTP_500_INTERNAL_SERVER_ERROR
            
        else:
            logger.info('data missing: 400 BAD REQUEST')
            return "No data found", status.HTTP_400_BAD_REQUEST

        logger.info('Start Prediction')
        try:
            # Pass result of input function to predict function
            prediction = Main[fw].predict_function(input_data)

        except Exception as e:
            logger.exception("Error upon predicting model:")
            return f"Server error upon predicting model: {e}", status.HTTP_500_INTERNAL_SERVER_ERROR
        
        logging.info('Start Output') 
        try:
            # Pass result of predict function to output function
            output_data = Main[fw].output_function(prediction)

        except Exception as e:
            logger.exception("Error upon generating output data:")
            return f"Server error upon generating output data: {e}", status.HTTP_500_INTERNAL_SERVER_ERROR
        
        # return output_data, status.HTTP_201_CREATED
        resp = Response(output_data, status=status.HTTP_201_CREATED, mimetype='application/json')

        # Add ids to rsponse header if they were sent
        if 'ids' in request.headers:
            resp.headers['ids'] = request.headers['ids']

        return resp

    # Add ui endpoint if ui_function is defined by framework
    @app.route("/html/<fw>", methods=['GET'])
    def html_gen(fw):

        """
        Return a page with details regarding a prediction.
        """
        if "ui_function" in dir(Main[fw]):
            # Get requested store_id from argument
            id = request.args.get('id')

            try:
                # pass requested id to ui function and return generated web page
                ui = Main[fw].ui_function(id)
                return ui

            except Exception as e:
                logger.exception("Error upon generating html:")
                return f"Server error upon generating html: {e}", status.HTTP_500_INTERNAL_SERVER_ERROR
        else:
            return f"No html generation implemented for {fw}", status.HTTP_501_NOT_IMPLEMENTED
    
    @app.route("/reinit/<fw>", methods=['GET'])
    def reinit(fw):

        """
        Reinitialize the framework class.
        """
        global Main

        if request.args.get('secret'):
            arg_secret = request.args.get('secret') 
        else:
            arg_secret = ''

        # Note we're using a key here (should match the SECRET value in the .env file) to prevent abuse of this endpoint.
        if os.environ['SECRET'] == arg_secret:
            # Get requested store_id from argument
            try:

                logger.info(f"Reinitializing framework {fw}...")

                Main[i] = import_module(f"frameworks.{fw}.main").Main(
            code_dir = f'frameworks/{fw}',
            log = logger,
            cache = r
            )
                return "Done!"

            except Exception as e:
                logger.exception("Error upon initializing Main class:")
                return f"Server error upon initializing Main class: {e}", status.HTTP_500_INTERNAL_SERVER_ERROR
        else:
            return f"Incorrect secret provided!", status.HTTP_401_UNAUTHORIZED
            
    # Load required classes from framework modules

    Main = {}

    for i in fw_names:

        logger.info(f"Initializing framework {i}...")

        Main[i] = import_module(f"frameworks.{i}.main").Main(
            code_dir = f'frameworks/{i}',
            log = logger,
            cache = r
            )

    # Run Server ----------------------------------------------------------
    port = 5000
    host = '0.0.0.0'

    if os.environ['DEV_MODE'] == "True":
        logger.setLevel(logging.DEBUG)
        logger.warning(f'Starting app in development mode')
        
        app.run(host=host, 
                debug=True, 
                port=port
                #ssl_context='adhoc'
                )
    else:
        logger.info('Starting app in production mode')
        try:
            ssl_cert_path = os.path.abspath(os.environ['SSL_CERT_PATH'])
            ssl_key_path = os.path.abspath(os.environ['SSL_KEY_PATH'])
        except Exception as e:
            logger.exception("type error: " + str(e))
            logger.exception("\n\tFor production use:\n\t\tPlease make sure environment variables 'SSL_CERT_PATH' and 'SSL_KEY_PATH' are set up correctly")

        http_server = WSGIServer(
            (host, port), 
            app, 
            log=logger
            #keyfile=ssl_key_path,
            #certfile=ssl_cert_path
            )
        http_server.serve_forever()
            
