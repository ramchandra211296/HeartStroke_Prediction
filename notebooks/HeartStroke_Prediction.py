import pandas as pd
import numpy as np
import shap
import socket
from category_encoders.one_hot import OneHotEncoder
from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import DataframeInput
from shapash.explainer.smart_explainer import SmartExplainer
## Since we are using Scikit-learn model, we use SklearnModelArtifact
from bentoml.frameworks.sklearn import SklearnModelArtifact
from category_encoders.one_hot import OneHotEncoder
from sklearn.model_selection import train_test_split
from category_encoders import (BackwardDifferenceEncoder,BinaryEncoder,HashingEncoder,HelmertEncoder,
                               OneHotEncoder
                               ,OrdinalEncoder,SumEncoder,PolynomialEncoder)
from sklearn.preprocessing import StandardScaler


from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

from custom_services import MyModelArtifact
# --------- 12.0
import pandas as pd
import numpy as np
import shap
from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import DataframeInput
from shapash.explainer.smart_explainer import SmartExplainer
from bentoml.frameworks.sklearn import SklearnModelArtifact
from sklearn.preprocessing import StandardScaler

@env(infer_pip_packages=True)
@artifacts([SklearnModelArtifact('shap_loaded_model'), SklearnModelArtifact('mlflow_loaded_model')])
class HeartStrokePredictionService_v(BentoService):
    """
    Model Purpose:
    The goal of this algorithm is to forecast a heart attack using a patient's personal and medical data. 
    The target column is discrete in nature, making it a classification problem.
    
     --- Input Data Format
    
         gender                         : object   --- values --- Male , Female, Other

         age                            : float64  --- values --- 0.08 to 82.0

         hypertension                   : int64    --- values --- 0,1

         heart_disease                  : int64    --- values --- 0,1

         ever_married                   : object   --- values --- Yes, No

         work_type                      : object   --- values --- Private, Self-employed, Govt_job, children, Never_worked
 
         Residence_type                 : object   --- values --- Urban, Rural

         avg_glucose_level              : float64  --- values --- 55.12 to 271.74

         bmi                            : float64  --- values --- 10.3 to 97.6
        
         smoking_status                 : object   --- values --- formerly smoked, never smoked, smokes,Unknown
    
    Output Data Format:
    The output will be a JSON array containing the predicted stroke status for each input record.
    
   --- Example Input:
    [{"gender": "Male", 
    "age": 32, 
    "hypertension": 0, 
    "heart_disease": 0, 
    "ever_married": "Yes",
    "work_type": "Private", 
    "Residence_type": "Urban", 
    "avg_glucose_level": 92.50, 
    "bmi": 21.56,
    "smoking_status": 
    "never smoked"}]
    
    
    
   --- Example Output:
          stroke : Stroke (or) No Stroke
       
   """
    Columns=["gender", "age","hypertension","heart_disease",
             "ever_married","work_type", 
             "Residence_type", "avg_glucose_level", 
             "bmi", "smoking_status"]
    # Note :
    # 1. The input provided in the Request body has to be in double quotes instead of single quotes.
    # 2. The mlflow by default runs on localhost:5000 and bentoml also does the same. 
    # So, need to set either one on a different port. Otherwise there wil
    

    @api(input=DataframeInput(orient='records'), batch=True)
    def predict(self, df: pd.DataFrame):
        """
        Model Purpose:
        The goal of this algorithm is to forecast a heart attack using a patient's personal and medical data. 
        The target column is discrete in nature, making it a classification problem.
    
    
         --- Example Input Sample:
        [{"gender": "Male", 
        "age": 32, 
        "hypertension": 0, 
        "heart_disease": 0, 
        "ever_married": "Yes",
        "work_type": "Private", 
        "Residence_type": "Urban", 
        "avg_glucose_level": 92.50, 
        "bmi": 21.56,
        "smoking_status": 
        "never smoked"}]
        
        
        --- Output sample 
                [Stroke or No Stroke ] 
    
    """
        
        out = self.artifacts.mlflow_loaded_model.predict(df)
        return np.where(out == 1, 'Stroke', 'No Stroke')
    
    @api(input=DataframeInput(orient='records'), batch=True)
    def get_probabilities(self, df: pd.DataFrame):
        
        """
        This is Heart Stroke Prediction Service.
        
        This is the endpoint to get probabilities for with respect to target variable.
        
        Sample Input: [{"gender": "Male","age": 81,"hypertension": 1,"heart_disease": 1,"ever_married": "Yes",
         "work_type": "Private","Residence_type": "Urban","avg_glucose_level": 92.50,"bmi": 21.56,
           "smoking_status": "smokes"}]
        Sample output: [{"No Stroke": 0.9948551654815674,"Stroke": 0.005144850816577673 }]
        
        """
        out = pd.DataFrame(self.artifacts.mlflow_loaded_model.predict_proba(df),
                           columns=['No Stroke', 'Stroke'])
        return out.to_dict(orient='records')

    
    
    
    @api(input=DataframeInput(orient='records'), batch=True)
    def explain(self, df: pd.DataFrame, port=1005, bentoml_url="http://localhost:5050"):
        """
        This is Heart Stroke Prediction Service.

        Given a minimum of two observations, this API endpoint will interpret the model and subsequently return a dashboard URL.

        --- Input sample ---
        [{"gender": "Male", "age": 38, "hypertension": 0, "heart_disease": 0, "ever_married": "Yes", "work_type": "Private", "Residence_type": "Urban", "avg_glucose_level": 87.35, "bmi": 23.1, "smoking_status": "never smoked"},
                {"gender": "Female", "age": 41, "hypertension": 0, "heart_disease": 1, "ever_married": "Yes", "work_type": "Govt_job", "Residence_type": "Rural", "avg_glucose_level": 65.25, "bmi": 26.7, "smoking_status": "formerly smoked"},
                {"gender": "Male", "age": 45, "hypertension": 1, "heart_disease": 0, "ever_married": "Yes", "work_type": "Private", "Residence_type": "Urban", "avg_glucose_level": 129.54, "bmi": 31.8, "smoking_status": "smokes"},
                {"gender": "Male", "age": 35, "hypertension": 0, "heart_disease": 0, "ever_married": "Yes", "work_type": "Self-employed", "Residence_type": "Rural", "avg_glucose_level": 94.76, "bmi": 27.2, "smoking_status": "never smoked"},
                {"gender": "Female", "age": 28, "hypertension": 0, "heart_disease": 0, "ever_married": "No", "work_type": "Private", "Residence_type": "Urban", "avg_glucose_level": 71.49, "bmi": 21.7, "smoking_status": "never smoked"},
                {"gender": "Male", "age": 55, "hypertension": 0, "heart_disease": 0, "ever_married": "Yes", "work_type": "Private", "Residence_type": "Urban", "avg_glucose_level": 87.12, "bmi": 29.4, "smoking_status": "never smoked"},
                {"gender": "Female", "age": 32, "hypertension": 0, "heart_disease": 0, "ever_married": "Yes", "work_type": "Self-employed", "Residence_type": "Rural", "avg_glucose_level": 78.65, "bmi": 24.6, "smoking_status": "formerly smoked"},
                {"gender": "Male", "age": 49, "hypertension": 1, "heart_disease": 0, "ever_married": "Yes", "work_type": "Govt_job", "Residence_type": "Rural", "avg_glucose_level": 105.8, "bmi": 26.5, "smoking_status": "smokes"},
                {"gender": "Male", "age": 60, "hypertension": 1, "heart_disease": 1, "ever_married": "Yes", "work_type": "Private", "Residence_type": "Rural", "avg_glucose_level": 171.23, "bmi": 32.7, "smoking_status": "formerly smoked"},
                {"gender": "Female", "age": 42, "hypertension": 0, "heart_disease": 0, "ever_married": "Yes", "work_type": "Private", "Residence_type": "Urban", "avg_glucose_level": 94.34, "bmi": 28.3, "smoking_status": "never smoked"},
                {"gender": "Male", "age": 29, "hypertension": 0, "heart_disease": 0, "ever_married": "No", "work_type": "Private", "Residence_type": "Urban", "avg_glucose_level": 72.89, "bmi": 23.8, "smoking_status": "never smoked"},
                {"gender": "Female", "age": 33, "hypertension": 0, "heart_disease": 0, "ever_married": "Yes", "work_type": "Private", "Residence_type": "Urban", "avg_glucose_level": 91.43, "bmi": 20.4, "smoking_status": "never smoked"},
                {"gender": "Male", "age": 46, "hypertension": 0, "heart_disease": 0, "ever_married": "Yes", "work_type": "Private", "Residence_type": "Urban", "avg_glucose_level": 106.54, "bmi": 27.8, "smoking_status": "formerly smoked"},
                {"gender": "Male", "age": 48, "hypertension": 1, "heart_disease": 0, "ever_married": "Yes", "work_type": "Private", "Residence_type": "Rural", "avg_glucose_level": 167.67, "bmi": 29.5, "smoking_status": "smokes"},
                {"gender": "Female", "age": 50, "hypertension": 0, "heart_disease": 0, "ever_married": "Yes", "work_type": "Self-employed", "Residence_type": "Urban", "avg_glucose_level": 92.01, "bmi": 25.3, "smoking_status": "never smoked"},
                {"gender": "Female", "age": 26, "hypertension": 0, "heart_disease": 0, "ever_married": "No", "work_type": "Private", "Residence_type": "Rural", "avg_glucose_level": 79.17, "bmi": 22.0, "smoking_status": "never smoked"},
                {"gender": "Male", "age": 54, "hypertension": 1, "heart_disease": 0, "ever_married": "Yes", "work_type": "Govt_job", "Residence_type": "Rural", "avg_glucose_level": 132.45, "bmi": 28.7, "smoking_status": "smokes"},
                {"gender": "Male", "age": 39, "hypertension": 0, "heart_disease": 0, "ever_married": "Yes", "work_type": "Private", "Residence_type": "Urban", "avg_glucose_level": 75.8, "bmi": 22.9, "smoking_status": "never smoked"},
                {"gender": "Male", "age": 57, "hypertension": 1, "heart_disease": 0, "ever_married": "Yes", "work_type": "Self-employed", "Residence_type": "Rural", "avg_glucose_level": 168.37, "bmi": 31.1, "smoking_status": "formerly smoked"},
                {"gender": "Female", "age": 30, "hypertension": 0, "heart_disease": 0, "ever_married": "Yes", "work_type": "Private", "Residence_type": "Rural", "avg_glucose_level": 95.64, "bmi": 26.4, "smoking_status": "never smoked"},
                {"gender": "Female","age": 81,"hypertension": 1,"heart_disease": 1,"ever_married": "Yes","work_type": "Private","Residence_type": "Rural","avg_glucose_level": 189.68,"bmi": 56.98,"smoking_status": "never smoked"}
                          ]
          
        

        --- Output sample ---
        [
            "http://localhost:1005"
        ]
        """

        COLUMNS = ['age','hypertension','heart_disease','avg_glucose_level','bmi',
                   'gender_Male', 'gender_Female', 'gender_Other','ever_married_Yes', 
                   'ever_married_No','work_type_Private','work_type_Self-employed', 
                   'work_type_Govt_job', 'work_type_children','work_type_Never_worked',
                   'Residence_type_Urban','Residence_type_Rural','smoking_status_formerly smoked',
                   'smoking_status_never smoked', 'smoking_status_smokes','smoking_status_Unknown']

        print("Dataframe:")
        print(df)
        print(df.dtypes)
        df_processed = self.artifacts.mlflow_loaded_model.named_steps['pre_processing'].transform(df)

        data_asframe = pd.DataFrame(df_processed, columns=COLUMNS)

        # Print the processed DataFrame
        print("Processed DataFrame:")
        print(df_processed)
        print(data_asframe)
        print("Available keys:", self.artifacts.mlflow_loaded_model.named_steps.keys())

        explainer = shap.KernelExplainer(self.artifacts.mlflow_loaded_model.named_steps['clf_xgbc'].predict, data_asframe)
        ypred = pd.DataFrame(self.artifacts.mlflow_loaded_model.predict(df), columns=['pred'], index=df.index)
        print("Getting ready explainer...")
        shap_contrib = explainer.shap_values(data_asframe)
        print(self.artifacts.shap_loaded_model)

        # Note: The SmartExplainer accepts a pipeline model as input. But, there were errors saying
        # few Column Transformers were not included in the Shapash module. Hence, a simple model
        # without a pipeline was created and provided as input to it.

        xE = SmartExplainer(model=self.artifacts.shap_loaded_model)
        xE.compile(contributions=shap_contrib,
                   x=data_asframe,
                   y_pred=ypred,
                   )
        app = xE.run_app(title_story='HeartStrokePrediction', port=port)
        url = 'http://' + str(bentoml_url.replace('http://', '').replace('https://', '').split(':')[0]) + ':' + str(port)
        return [url]
