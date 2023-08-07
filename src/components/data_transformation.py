import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from utils.utils import save_object
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

from exception import CustomException
from logger import logging
import os

from utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")


class GenderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.gender_dict = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.gender_dict is None:
            self.gender_dict = {'male', 'female'}  

        def Gen(x):
            if x in self.gender_dict:
                return str(x)
            else:
                return 'other'
        
        X['New Gender'] = X["Gender "].apply(Gen)

        
        logging.info("Columns after1:")
        logging.info(X.columns)
        
        return X
    
class Function(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):        

        
        logging.info("Columns after1:")
        logging.info(X.columns)
        
        return X
class HiringSource(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):        
        
        logging.info("Columns after1:")
        logging.info(X.columns)
        
        return X
    
class tengrp(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):        
        
        logging.info("Columns after1:")
        logging.info(X.columns)
        
        return X
    

class PromotedTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def Promoted(x):
            if x == 'Promoted':
                return int(1)
            else:
                return int(0)

        X['New Promotion'] = X["Promoted/Non Promoted"].apply(Promoted)
        logging.info("Columns after2:")
        logging.info(X.columns)
        return X

class Location(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.location_dict_new = {
            'Chennai':       7,
            'Noida':         6,
            'Bangalore':     5,
            'Hyderabad':     4,
            'Pune':          3,
            'Madurai':       2,
            'Lucknow':       1,
            'other place':   0,
        }
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def location(x):                     
            if str(x) in self.location_dict_new.keys():
                return self.location_dict_new[str(x)]
            else:
                return self.location_dict_new['other place']
        
        X['New Location'] = X["Location"].apply(location)
        logging.info("Columns after3:")
        logging.info(X.columns)
        return X  
    
class Marraige(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.Marital_dict = None
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.Marital_dict = X["Marital Status"].value_counts().to_dict()
        def Mar(x):
            if str(x) in self.Marital_dict.keys() and self.Marital_dict[str(x)] > 100:
                return str(x)
            else:
                return 'other status'

        X['New Marital'] = X["Marital Status"].apply(Mar)

        logging.info("Columns after4:")
        logging.info(X.columns) 
        return X 
    
class Emp_group(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.Emp_dict_new = {
            'B1': 4,
            'B2': 3,
            'B3': 2,
            'D2': 1,
            'other group': 0,
        }

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        def emp(x):
            if str(x) in self.Emp_dict_new:
                return str(x)
            else:
                return 'other group'
            
        # Print unique values before transformation
        logging.info("Unique values in Emp. Group column before transformation:")
        logging.info(X["Emp. Group"].unique())
        
        X['New EMP'] = X["Emp. Group"].apply(emp)
        
        # Print unique values after transformation
        logging.info("Unique values in New EMP column after transformation:")
        logging.info(X["New EMP"].unique())
        return X
   
class Job_Role(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def Job(x):
            if x == 'Yes':
                return int(1)
            else:
                return int(0)

        X['New Job Role Match'] = X["Job Role Match"].apply(Job)
        logging.info("Columns after6:")
        logging.info(X.columns)     
        return X

class Gender(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.gender_dict = None
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.gender_dict = X["Gender "].value_counts().to_dict()
        def Gen(x):
            if str(x) in self.gender_dict.keys() and self.gender_dict[str(x)] > 100:
                return str(x)
            else:
                return 'other status'
        
        X['New Gender'] = X["Gender "].apply(Gen)
        logging.info("Columns after7:")
        logging.info(X.columns)         
        return X 

class Droping(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if self.columns_to_drop:
            X = X.drop(self.columns_to_drop, axis=1)
            logging.info("Columns after dropping:")
            logging.info(X.columns)
            
        return X
columns_to_drop_names = ["table id", "name", "Marital Status", "Promoted/Non Promoted", 
                         "Function", "Job Role Match", "Location", "Emp. Group",
                         "Hiring Source", "Gender ", "Tenure", "Tenure Grp.", "phone number",]
drop_columns_transformer = Droping(columns_to_drop=columns_to_drop_names)


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            categorical_features = ["Gender ", "Function", "Hiring Source", "Marital Status", "Emp. Group", "Tenure Grp."]

            preprocessor = make_column_transformer(
                (GenderTransformer(), ["Gender "]),
                (Function(), ["Function"]),
                (PromotedTransformer(), ["Promoted/Non Promoted"]),
                (HiringSource(), ["Hiring Source"]),
                (Location(), ["Location"]),
                (Marraige(), ["Marital Status"]),
                (Emp_group(), ["Emp. Group"]),
                (Job_Role(), ["Job Role Match"]),
                (tengrp(), ["Tenure Grp."]),
                (OneHotEncoder(), categorical_features),  # Include 'Tenure Grp.' here
                (drop_columns_transformer, columns_to_drop_names),
            )

            data_pipeline = Pipeline([
                ('preprocessor', preprocessor),
            ])


            return data_pipeline
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="Stay/Left"

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)