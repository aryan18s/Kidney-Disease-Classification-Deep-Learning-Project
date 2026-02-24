import os
import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
import dagshub 

from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml, create_directories, save_json

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    
    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    
    def log_into_mlflow(self):
        # --- Terminal Authentication (REQUIRED FOR MAIN.PY) ---
        os.environ["MLFLOW_TRACKING_URI"] = self.config.mlflow_uri
        os.environ["MLFLOW_TRACKING_USERNAME"] = "aryan18s"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "2073199cd64e8aab8d337be5dcdf023d084babd6" #token

        dagshub.init(
            repo_owner='aryan18s', 
            repo_name='Kidney-Disease-Classification-Deep-Learning-Project', 
            mlflow=True
        )
        
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_registry_uri(self.config.mlflow_uri)
        
        # --- Clear memory and force an experiment section ---
        if mlflow.active_run():
            mlflow.end_run()
            
        mlflow.set_experiment("Kidney_Project_Runs") 
        # ----------------------------------------------------
        
        with mlflow.start_run():
            # Convert all parameters to strings so lists don't crash MLflow
            clean_params = {k: str(v) for k, v in self.config.all_params.items()}
            mlflow.log_params(clean_params)
            
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            
            # --- THE VERSIONING FIX ---
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            
            if tracking_url_type_store != "file":
                # Giving it a name here triggers the automatic V1, V2, V3 tracking!
                mlflow.keras.log_model(self.model, "model", registered_model_name="KidneyDiseaseModel")
            else:
                mlflow.keras.log_model(self.model, "model")
            # --------------------------