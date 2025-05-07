# uppsala_vitals

This framework implements models based on Emergency Medical Dispatch system and vital sign using the methodology described in [this research paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0226518). This is the modelling framework is being investigated for use for prehospital triage in Sweden.

The framework implements the [xgboost](https://xgboost.readthedocs.io/en/latest/) algorithm on dispatch system data and vital signs. The data used has been selected based on the predictive values of the predictors, and their ability to be collected in a variety of settings. The intent is to reflect a set of paramters that could be collected for instance by a care provider on the scene of an incident to gauge the need for an ambulance response. Note that the Dispatch categories and priorities may vary from cotext to context.

The framework is designed to enable a relatively easy model updating process by running `python -m frameworks.amb_refer.run`. This instantiates the Main class with parameters set to clean data found in `data/raw`, generate appropriate training and test datasets, and train new models using the updated data. This naturally won't work unless you have data in exactly the correct format. Some sample data is provided in `data/clean` to demonstrate the format the model will expect to recieve data in. Sample data in the format expected by API is provided in `data/api`. Note that this framework directly returns an html sting in the payload rather than using th ui endpoint.

If adapting these methods in your own research, please cite our paper:

Spangler D, Hermansson T, Smekal D, Blomberg H (2019)  
A validation of machine learning-based risk scores in the prehospital setting.  
PLoS ONE 14(12): e0226518. https://doi.org/10.1371/journal.pone.0226518  

Thank you!