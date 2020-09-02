# openTriage

openTriage is a platform for deploying machine learning models for use in Clinical Decision Support Systems (CDSS). This software is developed and maintained by the [Uppsala Center for Prehospital Research](http://ucpr.se/projects/emdai/) at Uppsala University, and was developed as part of a project funded by the [Swedish Agency for Innovation](https://www.vinnova.se/en/p/emdai-a-machine-learning-based-decision-support-tool-for-emergency-medical-dispatch/). The purpose of this software is to implement the methods described in the research paper [*A validation of machine learning-based risk scores in the prehospital setting*](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0226518) in an API for use in an upcoming randomized clinical trial in Uppsala, Sweden. The source code for this research may be found [here](https://github.com/dnspangler/openTriage_validation), and a demonstration app based on public-release versions of the models may be found [here](https://ucpr.shinyapps.io/openTriage_demo/). The software is designed to be:

* Free
* Open-source
* Implementable on local IT architecture
* Compliant with Region Uppsala/GDPR information security requirements
* Suitable for extension to other applications and contexts

This software is provided to the public with the intent of providing transparency in the methods we employ to process patient data, and to encourage the further use of open-source software in the healthcare sector. The primary intended users of this software are researchers and public sector IT departments seeking to serve relatively light-weight, open machine learning models to proprietary user-facing medical record systems. Technically, openTriage implements a REST API using Python in a docker container, functionality to serve interactive user interfaces using the Shiny R package, and an nginx reverse proxy to handle encryption and tie it all together. The software is provided under the terms of the [GPLv3 license](https://www.gnu.org/licenses/gpl-3.0.en.html). The functions used by the API are organized into frameworks which contain context-specific functionality for parsing data and performing model training, inference, and testing. Currently openTriage includes the following frameworks:

* [**uppsala_vitals**](frameworks/uppsala_alitis): A simplified version of the ambulance data-data based models described in the article. Provides risk estimates for a number of outcome measures based on patient demographics, vital signs, dispatch categories, and time. Includes trained models and sample data. A GUI using R and shiny for interactively setting model parameters via the API is also included. A demonstration of this framework is available at [opentriage.net](https://opentriage.net/ui/vitals)

* [**uppsala_alitis**](frameworks/uppsala_alitis): An implementation of the methods for predicting outomes based on dispatch data described in our research. Based on data from-, and designed to interface with the [Alitis](https://www.alecom.se/) Emergency Medical Dispatching system. 

* [**news_adhoc**](frameworks/news_adhoc): An implementation of the [National Emergency Warning System](https://www.rcplondon.ac.uk/projects/outputs/national-early-warning-score-news-2) algorithm to parse data in an ad-hoc JSON format. (Planning to implement parsing of [NEMSIS3](https://nemsis.org/)-formatted data)

* [**minimal**](frameworks/minimal): A set of minimal functions for testing and development.

This readme will describe the process of getting the API up and running with the NEWS algorithm (no data or models required!), provide an overview of the steps needed to set up the system for use in other contexts, and briefly describe how the functions which the API will expect to find in a framework are implemented and used here in uppsala.

## Getting started

openTriage is designed to be implementable on a wide range of systems, and has been tested on linux (Ubuntu and RHEL) and Windows 10-based systems. openTriage is designed to be implemented as a docker-based service, and requires that you have some tools installed on your system:

```
python 3
git
docker
Docker-compose
```
For the purpose of this guide, we'll assume you're using an Ubuntu 18 linux distribution. Python 3 is installed by default in Ubuntu. These guides explain how to install [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git), [docker](https://docs.docker.com/install/#server), and [docker compose](https://docs.docker.com/compose/install/) on your machine.

Upon installing the above tools, clone this repository and navigate into the main openTriage directory using the following commands: 

```
git clone https://github.com/dnspangler/openTriage
cd openTriage
```

You now need to build and run the docker container. For a simple start in development mode using the NEWS framework run the following (the -d argument detaches the container so that you can continue using the terminal):

```
docker-compose up -d
```

Congratulations! You are now serving a REST API on port 443 which implements the NEWS2 triage algorithm. To make sure the API is working, you can send some [test data](frameworks/news_adhoc/data/api/test.json) using this command:

```
python tests\api_test.py
```

This command should print the score for each record, print a link to the ui (consider this the "hello world" of UIs), and notify you that an unverified HTTPS request is being made (Use an actual SSL certificate in production!). To actually do something useful however, you will need to understand a bit more about what's actually going on under the hood:

## API

The primary functionality of openTriage is accessed via the `/predict/<fw>/` endpoint, with the `fw` value corresponding to the framework name you wish to use. openTriage will process data posted to this endpoint, and return a payload containing risk prediction information to be used by the system making the POST request. At a minimum, This returned data should contain the *score* of the data posted. Optionally, the returned data may contain other information, including a *link* to a URL which will return an HTML page intended to provide some additional details about the risk prediction (accessed via GET request to the `/html/<fw>/` endpoint). A `/reinit/<fw>/` endpoint is also provided which will reinitialize the framework upon receiving a GET request (used for loading updated models without server downtime).

The system is designed such that the specifics of how data are to be handled are defined by a *framework*. The API expects each framework to contain a `main` python module with a `Main` class containing a number of required methods. These are:

* **\_\_init__**: Is executed upon initialization of the class (either upon starting the server or by a get request to the /reinit endpoint), and should load a model into memory.
* **input_function**: Transforms the data provided via the API into a 'clean' format expected by the model
* **predict_function**: Applies the loaded model to the clean data, and returns a prediction
* **output_function**: Applies any necessary transformations to the prediction before returning it via the API. If implementing a UI, the function should cache the prediction.

Additional optional functions are:

* **ui_function**: Retrieves the cached prediction, and serves a web page containing details regarding the model prediction (eg., feature importance, percentiles, etc.)

Schematically, the flow is thus:

![Flowchart of API functionality](/lib/assets/flow.svg)

## Deployment

Two modes are defined when deploying a server: Development mode will provide more logs and ui output, as well as using an ad-hoc ssl certificate. Production mode will limit logs to so as to not save individual level data for GDPR compliance, require that actual SSL certificates are provided, and provide a 'cleaner' ui for the uppsala_alitis package.

Ensure that a set of SSL keys are available in the ssl directory (talk to your IT department about how to obtain an SSL certificate for your server if you are unsure about this). By default, gunicorn will expect a private key (named `key.pem`) and certificate (named `certificate.pem`) in the lib/ssl directory. If no certificate is available, you can generate a self-signed certificate for use in trusted environments using OpenSSL (in linux/macos). Update ssl/openssl.cnf using a text editor to reflect your information, and run the following:

```
openssl req -newkey rsa:2048 -nodes -keyout ./lib/ssl/key.pem -x509 -days 365 -out ./lib/ssl/certificate.pem -config ./lib/ssl/openssl.cnf
```

In the root openTriage directory there is a file called `RENAME.env` - upon renaming the file to only `.env`, any variables defined here will be loaded into the docker container as environment variables. You can use this file to define the framework and mode (FRAMEWORK=*name of framework directory*, DEV_MODE= *True* or *False*), and provide a secret for use in the API (currently used to prevent abuse of the /reinit/ endpoint by requiring a matching `secret` argument). Note that multiple frameworks can be served by separating the framework names with spaces. The renamed file will be ignored by git. Once you have edited the .env file as appropriate run `docker-compose up` to start the server.

## Testing

As noted above, tests may be run using the api_test module:

```
python tests\api_test.py
```

This script sends a test file (default: test.json) via POST request to the /predict endpoint at a given address (default 127.0.0.1 aka localhost) assuming a given framework (default: NEWS). These defaults can be overridden using the command line arguments --file, --addr, and --fw respectively. The script will attempt to parse each returned item to extract the 'score' value, as well as send a get request expecting a web page to the endpoint passed as 'link' in the returned json file. This test reflects the set of API calls expected from the front-end system in a typical intervention using these tools.

## Development

To extend this openTriage to a new setting, we suggest copying and renaming one of the included frameworks and modifying the code found within it to your particular context. It is strongly recommended to use a virtual environment (e.g., [anaconda](https://www.anaconda.com/) as employed in the docker implementation, or the native python [venv](https://docs.python.org/3/tutorial/venv.html) module) if you aim to develop and execute code outside of the docker container. Note that if using anaconda, you can use the conda_env.yml file in the repo to ensure that your environment has the packages you'll need installed. You can use the following command to install all packages used by openTriage in a conda environment using the following command:

```
conda env update -f lib\conda_env.yml
```

We hope that if you use this software to implement triage models at your organization that you will be willing to donate your code back to the repository through pull requests, though of course this is not obligatory! By default, git is set to ignore any files added to the `data` and `models` directories in all frameworks to avoid accidental disclosures of patient data and IP. If you're interested in extending this platform for use at your organization, shoot me an email at douglas.spangler@akademiska.se and I'll do what I can to help you out! Please note that the frameworks included in the repo are in active development, are liable to change in functionality, and should not be depended upon by your own code. Unfortunately, we are currently unable to include files relating to the structure of the CDSS used in the Alitis dispatching system in this repository, as the copyright is jointly owned by the Sörmland, Västmanland, and Uppsala dispatch centers. If you're interested in in using the CDSS for some purpose, get in touch and I'll point you in the right direction. 

Below we briefly review how we have implemented the functions:

### \_\_init__ (loading/training the model)

In the uppsala_alitis framework, upon initialization the class will attempt to load a stored model into memory. If no model files are present, it will look for training and testing datasets to use (`data/train/data.csv` and `data/train/labels.csv` respectively), and then train models. If no testing/training data is available, it will attempt to clean data exported from our databases (stored in `data/raw`) before proceeding. Models are trained for each provided label (using bayesian optimization to identify optimal hyperparameters), and stored as serialized objects (using pickle) along with some ancillary human-readable JSON data in the `models` folder. It will also print some basic performance metrics based on testing datasets and labels (`data/test/data.csv` and `data/test/labels.csv`). Note that a script to initialize the main class is provided as run.py, and new models can thus be trained outside of a docker container by running `python -m frameworks.uppsala_alitis.run`.

In the case of news_adhoc, a simple function which can be applied directly to data to return a NEWS2 score is loaded. 

### input_function

Parses the data provided in the form of a JSON payload sent via a POST request into the expected format (note that the JSON file is automatically transformed into a python dictionary).

For the uppsala_alitis framework, this is done using a function stored in the utils.py module which is applied to each item provided in the JSON payload, each of which is assumed to represent a distinct patient record. 

For news_adhoc, no formatting is currently necessary.

### predict_function

Applies the model defined by load_model to the output of input_function, and returns a risk score. 

For uppsala_alitis, this involves predicting and averaging likelihoods of multiple binary outcomes included as labels as described in our paper.

### output_function

Transforms the output of predict_function into a JSON file which is returned as the response to the POST request. 

In the case of uppsala_alitis, in addition to a raw score, we calculate the rank of the score among all records provided in the POST request, apply a control/intervention arm randomization (if the variable randomize = True), define a color associated with the prediction, and return these in a format defined by an internal specification. The output function also stores modelling data temporarily (expires after 1 hour to minimize data exposure) to a Redis database for later retrieval by the ui function.

### ui_function

Retrieves risk prediction data from the cache and renders HTML to display based on a template stored in the /templates directory. In the uppsala_alitis implementation, it renders a density plot using matplotlib, marks out the risk associated with the current record and any other records in the trial, and stores the plot image as a base64 value in the HTML itself. It also generates risk scores and percentile ranks for each of the included component outcome measures, as well as a table reporting [SHAP values](https://christophm.github.io/interpretable-ml-book/shap.html) for each documented feature. Currently the UI looks like this (Still needs a bit of prettying up, and sorry for the Swedish):

![UI screen dump](/lib/assets/ui_img.png)

For news_adhoc a simple "hello world" UI is implemented. Note that this function can be completely removed if not required for the use case, as in the "minimal" framework.

## Future development

There are plenty of features not essential to our use case that we haven't had time to implement yet! Regarding the API itself, some of these include:

* Serving predictions for multiple frameworks in same container
* Implementing strict typing, more unit tests, and continuous integration functionality

With regards to the uppsala_alitis framework specifically (above and beyond the myriad TODO comments in the code):

* Implement generation of synthetic datasets to enable data sharing
* Make UI prettier
* Implement more sophisticated NLP methods

Shoot me an email (douglas.spangler@akademiska.se) if you're interested in taking a crack at any of these or have other ideas for improving these tools! If you use this work in research, please cite our related paper:

Spangler D, Hermansson T, Smekal D, Blomberg H (2019)  
A validation of machine learning-based risk scores in the prehospital setting.  
PLoS ONE 14(12): e0226518. https://doi.org/10.1371/journal.pone.0226518  

Thank you!
