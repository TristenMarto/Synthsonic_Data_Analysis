README


This repo contains an adaptation of the [Synthsonic package](https://github.com/mbaak/synthsonic), implemented using the smote-variants package. The performance of synthsonic is tested against other oversampling techniques in imbalanced learning problems.



Datasets: Contains (real) datasets

Notebooks: Contains notebooks used for testing, evaluation and for creating reproducable results

Scripts
 - sv_synthsonic.py: Implementation of synsthsonic in Smote-variants
 - handle_data.py: functions related to loading and analysing data
 - evaluate_oversamplers_cross_val.py: script to run k-fold cross validation for a selection of datasets and oversamplers
 - handle_results: small class with functions to help in analysis of results
 - run_test.py: example script of running a test
