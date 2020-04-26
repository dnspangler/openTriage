# uppsala_vitals

This framework implements models based on Emergency Medical Dispatch system data from the [Alitis](https://www.alecom.se/tjanster/alitis-command-control.html) platform using the methodology described in [this research paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0226518). This is the modelling framework which we use at our dispatch center here in Uppsala.

The framework implements the [xgboost](https://xgboost.readthedocs.io/en/latest/) algorithm on data sent from the dispatch system. This includes structured data including patient demographics and findings documented in the clinical decision support system, and free-text notes entered by nurses. The framework is set up to be used in the context of a prospective randomized trial and will, by default in production mode, only display risk prediction information to the user 50% of the time.

The framework is designed to enable a relatively easy model updating process by running `python -m frameworks.uppsala_vitals.run`. This instantiates the Main class with parameters set to clean data found in `data/raw`, generate appropriate training and test datasets, and train new models using the updated data. This naturally won't work unless you have data in exactly the correct format (Working on generating some synthetic data to demonstrate the code with!).

If adapting these methods in your own research, please cite our paper:

Spangler D, Hermansson T, Smekal D, Blomberg H (2019)  
A validation of machine learning-based risk scores in the prehospital setting.  
PLoS ONE 14(12): e0226518. https://doi.org/10.1371/journal.pone.0226518  

Thank you!