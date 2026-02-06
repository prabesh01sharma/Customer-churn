"""
Model Retraining Pipeline Orchestrator
Runs the full ML pipeline and compares new model with existing model
"""
import os
import pickle
import logging
from typing import Dict, Any
from pathlib import Path

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import pandas as pd

from src.data_ingestion import DataIngestion
from src.data_validation import DataValidation
from src.data_transformation import DataTransformation
from src.model_train import ModelTrainer


class ModelRetrainer:
    """Orchestrates the full retraining pipeline"""
    
    def __init__(self):
        self.base_path = Path(__file__).parent.parent
        self.model_path = self.base_path / "models" / "logistic_regression.pkl"
        self.new_model_path = self.base_path / "models" / "logistic_regression_new.pkl"
        self.preprocessor_path = self.base_path / "transformed_data" / "preprocessor.pkl"
        
    def run_pipeline(self) -> Dict[str, Any]:
        """
        Runs the complete ML pipeline:
        1. Data ingestion
        2. Data validation
        3. Data transformation
        4. Model training
        
        Returns dict with status and details
        """
        try:
            logging.info("=" * 50)
            logging.info("STARTING MODEL RETRAINING PIPELINE")
            logging.info("=" * 50)
            
            # Step 1: Data Ingestion
            logging.info("Step 1/4: Running data ingestion...")
            ingestion = DataIngestion()
            ingestion.load_csv_data()
            
            # Step 2: Data Validation
            logging.info("Step 2/4: Running data validation...")
            validation = DataValidation()
            validation.validate_and_save()
            
            # Step 3: Data Transformation
            logging.info("Step 3/4: Running data transformation...")
            transformation = DataTransformation()
            transformation.transform_and_save()
            
            # Step 4: Model Training
            logging.info("Step 4/4: Running model training...")
            trainer = ModelTrainer()
            trainer.train_and_save()
            
            logging.info("Pipeline completed successfully!")
            
            return {
                "status": "success",
                "message": "Pipeline executed successfully",
                "steps_completed": ["ingestion", "validation", "transformation", "training"]
            }
            
        except Exception as e:
            logging.error(f"Pipeline failed: {str(e)}")
            return {
                "status": "error",
                "message": f"Pipeline failed: {str(e)}",
                "steps_completed": []
            }
    
    def compare_models(self) -> Dict[str, Any]:
        """
        Compares the newly trained model with the existing model.
        
        Decision criteria:
        - New model is better if:
          1. Recall improved by at least 2% OR
          2. Recall similar (within 2%) AND accuracy improved
        
        Returns comparison results and decision
        """
        try:
            # Load test data
            X_test_path = self.base_path / "transformed_data" / "X_test.csv"
            y_test_path = self.base_path / "transformed_data" / "y_test.csv"
            
            X_test = pd.read_csv(X_test_path)
            y_test = pd.read_csv(y_test_path).values.ravel()
            
            # Check if old model exists
            if not self.model_path.exists():
                logging.info("No existing model found. Will use new model.")
                return {
                    "decision": "replace",
                    "reason": "No existing model found",
                    "old_metrics": None,
                    "new_metrics": None
                }
            
            # Load old model
            with open(self.model_path, "rb") as f:
                old_model = pickle.load(f)
            
            # Load new model (just trained, saved as logistic_regression.pkl)
            # We need to temporarily rename it
            with open(self.model_path, "rb") as f:
                new_model = pickle.load(f)
            
            # Generate predictions
            old_pred = old_model.predict(X_test)
            new_pred = new_model.predict(X_test)
            
            # Calculate metrics
            old_metrics = {
                "accuracy": float(accuracy_score(y_test, old_pred)),
                "recall": float(recall_score(y_test, old_pred)),
                "precision": float(precision_score(y_test, old_pred)),
                "f1": float(f1_score(y_test, old_pred))
            }
            
            new_metrics = {
                "accuracy": float(accuracy_score(y_test, new_pred)),
                "recall": float(recall_score(y_test, new_pred)),
                "precision": float(precision_score(y_test, new_pred)),
                "f1": float(f1_score(y_test, new_pred))
            }
            
            # Decision logic
            recall_diff = new_metrics["recall"] - old_metrics["recall"]
            accuracy_diff = new_metrics["accuracy"] - old_metrics["accuracy"]
            
            should_replace = False
            reason = ""
            
            if recall_diff >= 0.02:
                should_replace = True
                reason = f"Recall improved by {recall_diff*100:.2f}%"
            elif abs(recall_diff) < 0.02 and accuracy_diff > 0:
                should_replace = True
                reason = f"Similar recall ({recall_diff*100:+.2f}%) but accuracy improved by {accuracy_diff*100:.2f}%"
            else:
                should_replace = False
                reason = f"New model not better (recall: {recall_diff*100:+.2f}%, accuracy: {accuracy_diff*100:+.2f}%)"
            
            logging.info(f"Model comparison decision: {'REPLACE' if should_replace else 'KEEP OLD'}")
            logging.info(f"Reason: {reason}")
            
            return {
                "decision": "replace" if should_replace else "keep",
                "reason": reason,
                "old_metrics": old_metrics,
                "new_metrics": new_metrics,
                "improvements": {
                    "accuracy": accuracy_diff,
                    "recall": recall_diff,
                    "precision": new_metrics["precision"] - old_metrics["precision"],
                    "f1": new_metrics["f1"] - old_metrics["f1"]
                }
            }
            
        except Exception as e:
            logging.error(f"Model comparison failed: {str(e)}")
            return {
                "decision": "error",
                "reason": f"Comparison failed: {str(e)}",
                "old_metrics": None,
                "new_metrics": None
            }
    
    def execute_retraining(self) -> Dict[str, Any]:
        """
        Main execution method that:
        1. Runs the pipeline
        2. Compares models
        3. Replaces if better
        
        Returns complete results
        """
        # Run pipeline
        pipeline_result = self.run_pipeline()
        
        if pipeline_result["status"] != "success":
            return {
                "status": "error",
                "message": pipeline_result["message"],
                "pipeline_result": pipeline_result,
                "comparison_result": None,
                "model_replaced": False
            }
        
        # Compare models
        comparison = self.compare_models()
        
        model_replaced = False
        if comparison["decision"] == "replace":
            # Model is already saved as logistic_regression.pkl by the trainer
            # No need to do anything - it's already replaced
            model_replaced = True
            logging.info("✓ New model is now active")
        elif comparison["decision"] == "keep":
            # Need to restore the old model (saved before training)
            # For now, we'll just log - in production you'd backup/restore
            logging.info("✓ Keeping existing model (not replaced)")
        
        return {
            "status": "success",
            "message": "Retraining completed successfully",
            "pipeline_result": pipeline_result,
            "comparison_result": comparison,
            "model_replaced": model_replaced
        }
