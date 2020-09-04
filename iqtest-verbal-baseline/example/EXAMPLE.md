# Description

This is a simple example baseline model for the IQtest, which is able to do analogy and classification questions in the Verbal category.

# How to use

## run_script.py

The run_script.py provides client_mode, server_mode, pack_model for the model
To customize the location of the extracted dataset, please specify  `DATA_ROOT`.

## entry_model.py

The model entry file that 

* defines get_eval_cls

>    ```python
>    def get_eval_cls(category: str) -> object:
>        pass
>    ```

* or defines get_model_object

>    ```python
>    def get_model_object(category: str) -> object:
>       pass
>    ```
>
>    get_model_object would override get_eval_cls
>    return a model class to each of the three categories, return `None` would ignore the corresponding competition

* provide global_pre_run (optional)

>    ```python
>    def global_pre_run():
>       pass
>    ```
>
>    Used for global setup, such as setup environment, load  train data so etc.

Model class will also need to provide

>    ```python
>    def solve(self, question):
>       return [question['id'], [1]]
>    ```
>
>    the variable 'question' will be a question from the question list that needs to be answerd by the model object, answers should be returned along with its question id

# Data

## group_data and train_data

please refer to README.md under source directory root



## eval_data

The eval_data is the set of stardardized verbal questions we picked out from the dataset to work on.

 Feel free to run on it yourself, remember adjust the `DATA_ROOT` in run_script.py and adjust the "test_suites" in the config file.

