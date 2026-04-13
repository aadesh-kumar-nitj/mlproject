import sys
import pandas as pd
import os
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # This identifies the folder where THIS file (predict_pipeline.py) sits
            # C:\mlproject\src\pipelines\
            current_path = os.path.dirname(os.path.abspath(__file__))

            # We go up two levels to reach C:\mlproject\ and then into artifacts
            model_path = os.path.join(current_path, '..', '..', 'artifacts', 'model.pkl')
            preprocessor_path = os.path.join(current_path, '..', '..', 'artifacts', 'preprocessor.pkl')

            print("Loading objects...")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("Finished Loading")

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)



class CustomData:
    def __init__(  self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethnicity], # Changed from race_ethnicity
                "parental level of education": [self.parental_level_of_education], # Changed from parental_level_of_education
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course], # Changed from test_preparation_course
                "reading score": [self.reading_score], # Changed from reading_score
                "writing score": [self.writing_score], # Changed from writing_score
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)