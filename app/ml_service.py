"""
ML Service module for T-Mobile Installation Cost Prediction API
Handles model loading and prediction logic.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime
import traceback
from fastapi import HTTPException
from enum import Enum

from .models import PredictionRequest, PredictionResponse, PredictionBreakdown, ConfidenceLevel
from .config import settings

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG,  
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

class MLService:
    """Handles ML model loading and prediction operations."""
    
    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {}
        self.is_initialized = False
        self.model_dir = settings.MODEL_DIR


    @staticmethod
    def convert_json_to_dataframe(data_json):
        equipment_types = [
            "Antenna", "Coax", "Splitter", "Jumper", "Connector",
            "MPROU", "Fiber", "iBiU_POI", "oDU_oEU"
        ]

        name_map = {
            "Antenna": "Antenna",
            "Coax Cable": "Coax",
            "Splitter": "Splitter",
            "Jumper": "Jumper",
            "Connector": "Connector",
            "MPROU": "MPROU",
            "Fiber": "Fiber",
            "iBiU POI": "iBiU_POI",
            "oDU_oEU": "oDU_oEU"
        }

        classifications = ["Active", "Passive", "Unclassified"]

        row = {
            "Name": data_json.get("name", ""),
            "Coverage Area (sqft)": data_json.get("installation", {}).get("totalCoverageArea", 0),
            "Floors": data_json.get("installation", {}).get("numberOfFloors", 0),
            "Buildings": data_json.get("installation", {}).get("numberOfBuildings", 0),
            "Solution": data_json.get("installation", {}).get("solutionType", "").lower(),
            "Venue_type": data_json.get("installation", {}).get("venueType", "").lower(),
        }

        for eq in equipment_types:
            for suffix in ["Total", "Active", "Passive", "Unclassified", "Shared", "Dedicated"]:
                row[f"{eq}_{suffix}"] = 0

        for eq in data_json.get("equipment", []):
            eq_type_raw = eq.get("type", "")
            eq_type = name_map.get(eq_type_raw, None)
            if eq_type is None:
                continue

            qty = eq.get("qty", 0)
            shared = eq.get("shared", False)
            classification = eq.get("classification", "Unclassified")
            if isinstance(classification, Enum):
                classification = classification.value.capitalize()
            if classification not in classifications:
                classification = "Unclassified"

            row[f"{eq_type}_Total"] += qty
            row[f"{eq_type}_{classification}"] += qty

            if shared:
                row[f"{eq_type}_Shared"] += qty
            else:
                row[f"{eq_type}_Dedicated"] += qty

        df = pd.DataFrame([row])
        return df

    async def initialize(self):
        try:
            logger.info("Initializing ML models...")
            
            # Load default models
            await self._load_model_set("default", "model_rf.pkl", "model_xgb.pkl", "model_features.pkl")
            
            # Load DAS models
            await self._load_model_set("das", "das_model_rf.pkl", "das_model_xgb.pkl", "das_model_features.pkl")
            
            # Load granular models dynamically
            await self.load_granular_models(folder_name="granular_model")  # dynamic folder
            
            self.is_initialized = True
            logger.info("ML models initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ML models: {e}")
            raise
        
    


    async def load_granular_models(self, folder_name="granular_model"):
        """
        Load all granular cost models (shared/dedicated/Miscellaneous)
        from a separate folder relative to the current working directory.
        """
        # Get current working directory
        base_dir = os.getcwd()
        granular_model_dir = os.path.join(base_dir, folder_name)
        granular_model_dir = os.path.normpath(granular_model_dir)
        print("Contents of base_dir:", os.listdir(os.getcwd()))
        print(f"Granular model directory: {granular_model_dir}")

        # Check if the folder exists
        if not os.path.exists(granular_model_dir):
            logger.error(f"Granular models folder does not exist: {granular_model_dir}")
            return

        # Mapping of model names to filenames
        granular_models = {
            'shared_active_cost': 'model_shared_active_cost.pkl',
            'shared_passive_cost': 'model_shared_passive_cost.pkl',
            'dedicated_active_cost': 'model_dedicated_active_cost.pkl',
            'dedicated_passive_cost': 'model_dedicated_passive_cost.pkl',
            'Miscellaneous_cost': 'model_Miscellaneous_cost.pkl'
        }

        # Load each model
        for name, file in granular_models.items():
            try:
                path = os.path.join(granular_model_dir, file)
                path = os.path.normpath(path)
                
                if not os.path.exists(path):
                    logger.error(f"Model file does not exist: {path}")
                    continue

                model = joblib.load(path)
                self.models[name] = model
                logger.info(f"Loaded granular cost model '{name}' from '{path}'")
            except Exception as e:
                logger.error(f"Failed to load granular model '{name}': {e}")
                raise




    async def _load_model_set(self, model_type: str, rf_file: str, xgb_file: str, features_file: str):
        """Load a set of models (RF, XGBoost, and features)."""
        try:
            rf_path = os.path.join(self.model_dir, rf_file)
            xgb_path = os.path.join(self.model_dir, xgb_file)
            features_path = os.path.join(self.model_dir, features_file)
            
            # Load models using joblib (same as main Streamlit app)
            rf_model = joblib.load(rf_path)
            xgb_model = joblib.load(xgb_path)
            model_features = joblib.load(features_path)
            
            self.models[model_type] = {
                'rf_model': rf_model,
                'xgb_model': xgb_model,
                'features': model_features
            }
            
            logger.info(f"Loaded {model_type} models successfully")
            
        except Exception as e:
            logger.error(f"Failed to load {model_type} models: {e}")
            raise
    
    def is_ready(self) -> bool:
        """Check if ML service is ready to make predictions."""
        return self.is_initialized and len(self.models) > 0 #updated from here
    
 






# Inside your MLService class:
    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        body = request.dict()
        logger.info(f"Incoming JSON: {request.dict()}")
        try:
            if not self.is_ready():
                raise RuntimeError("ML service is not ready")

            # Extract features for full (non-shared) prediction
            ml_features = self._extract_ml_features(request)
            solution_type = request.installation.solutionType.value.lower()
            model_key = "das" if solution_type == "das" else "default"

            if model_key not in self.models:
                raise ValueError(f"No models available for solution type: {solution_type}")

            models = self.models[model_key]

            # Build input for standard prediction
            input_df = self._prepare_input_dataframe(ml_features, request)
            input_encoded = pd.get_dummies(input_df, columns=['Solution', 'type_of_venue'])
            input_aligned = input_encoded.reindex(columns=models['features'], fill_value=0)

            rf_pred = models['rf_model'].predict(input_aligned)[0]
            xgb_pred = models['xgb_model'].predict(input_aligned)[0]

            if solution_type == "das":
                final_prediction = rf_pred * 0.8 + xgb_pred * 0.2
            else:
                final_prediction = (rf_pred + xgb_pred) / 2

            confidence = self._calculate_confidence(rf_pred, xgb_pred)
            # ---------- Granular Category Predictions ----------
            def _predict_for_equipment(equipment_list):
                if not equipment_list:
                    return 0.0
                subset_request = PredictionRequest(
                    name=request.name,
                    origin=request.origin,
                    equipment=equipment_list,
                    installation=request.installation
                )
                features = self._extract_ml_features(subset_request)
                input_df = self._prepare_input_dataframe(features, subset_request)
                input_encoded = pd.get_dummies(input_df, columns=['Solution', 'type_of_venue'])
                input_aligned = input_encoded.reindex(columns=models['features'], fill_value=0)

                rf = models['rf_model'].predict(input_aligned)[0]
                xgb = models['xgb_model'].predict(input_aligned)[0]
                if solution_type == "das":
                    return rf * 0.8 + xgb * 0.2
                else:
                    return (rf + xgb) / 2
                
            unclassified_equipment = [item for item in request.equipment if  getattr(item, 'classification', '') == 'Unclassified']
            if not unclassified_equipment:
                Miscellaneous_cost = 0
            else:
                Miscellaneous_cost = 0.05*_predict_for_equipment(unclassified_equipment)
            final_prediction1 = final_prediction-Miscellaneous_cost

            df_total = self.convert_json_to_dataframe(request.dict())

            df_total_encoded = pd.get_dummies(df_total, columns=['Solution', 'Venue_type'])
  

            request_dict = request.dict()

            t = request_dict  # JSON payload as dict
            equipment = t.get("equipment", [])

            # Filter equipment based on shared / classification
            shared_active_equipment = [item for item in equipment if item.get('shared', False) and item.get('classification') == 'Active']
            shared_passive_equipment = [item for item in equipment if item.get('shared', False) and item.get('classification') == 'Passive']
            dedicated_active_equipment = [item for item in equipment if not item.get('shared', False) and item.get('classification') == 'Active']
            dedicated_passive_equipment = [item for item in equipment if not item.get('shared', False) and item.get('classification') == 'Passive']
            miscellaneous_equipment = [item for item in equipment if item.get('classification') == 'Unclassified']

            # Prepare predictions dict
            predictions = {}

            # shared_active_cost
            if shared_active_equipment:
                df_sa = self.convert_json_to_dataframe(t)
                df_sa = df_sa.reindex(columns=self.models['shared_active_cost'].feature_names_in_, fill_value=0)
                df_sa = df_sa.apply(pd.to_numeric, errors='coerce').fillna(0)
                predictions['shared_active_cost'] = float(self.models['shared_active_cost'].predict(df_sa)[0])
            else:
                predictions['shared_active_cost'] = 0.0

            # shared_passive_cost
            if shared_passive_equipment:
                df_sp = self.convert_json_to_dataframe(t)
                df_sp = df_sp.reindex(columns=self.models['shared_passive_cost'].feature_names_in_, fill_value=0)
                df_sp = df_sp.apply(pd.to_numeric, errors='coerce').fillna(0)
                predictions['shared_passive_cost'] = float(self.models['shared_passive_cost'].predict(df_sp)[0])
            else:
                predictions['shared_passive_cost'] = 0.0

            # dedicated_active_cost
            if dedicated_active_equipment:
                df_da = self.convert_json_to_dataframe(t)
                df_da = df_da.reindex(columns=self.models['dedicated_active_cost'].feature_names_in_, fill_value=0)
                df_da = df_da.apply(pd.to_numeric, errors='coerce').fillna(0)
                predictions['dedicated_active_cost'] = float(self.models['dedicated_active_cost'].predict(df_da)[0])
            else:
                predictions['dedicated_active_cost'] = 0.0

            # dedicated_passive_cost
            if dedicated_passive_equipment:
                df_dp = self.convert_json_to_dataframe(t)
                df_dp = df_dp.reindex(columns=self.models['dedicated_passive_cost'].feature_names_in_, fill_value=0)
                df_dp = df_dp.apply(pd.to_numeric, errors='coerce').fillna(0)
                predictions['dedicated_passive_cost'] = float(self.models['dedicated_passive_cost'].predict(df_dp)[0])
            else:
                predictions['dedicated_passive_cost'] = 0.0

            # Miscellaneous_cost
            if miscellaneous_equipment:
                df_misc = self.convert_json_to_dataframe(t)
                df_misc = df_misc.reindex(columns=self.models['Miscellaneous_cost'].feature_names_in_, fill_value=0)
                df_misc = df_misc.apply(pd.to_numeric, errors='coerce').fillna(0)
                predictions['Miscellaneous_cost'] = float(self.models['Miscellaneous_cost'].predict(df_misc)[0])
            else:
                predictions['Miscellaneous_cost'] = 0.0

            shared_active_cost = predictions.get('shared_active_cost', 0.0)
            shared_passive_cost = predictions.get('shared_passive_cost', 0.0)
            dedicated_active_cost = predictions.get('dedicated_active_cost', 0.0)
            dedicated_passive_cost = predictions.get('dedicated_passive_cost', 0.0)
            Miscellaneous_cost = predictions.get('Miscellaneous_cost', 0.0)

            costs_dict = {
    'shared_active_cost': shared_active_cost,
    'shared_passive_cost': shared_passive_cost,
    'dedicated_active_cost': dedicated_active_cost,
    'dedicated_passive_cost': dedicated_passive_cost,
    'Miscellaneous_cost': Miscellaneous_cost
}

            # Identify non-zero costs
            non_zero_keys = [k for k, v in costs_dict.items() if v > 0]

            if len(non_zero_keys) == 1:
                # Only one cost has value, assign final_prediction to it
                costs_dict[non_zero_keys[0]] = final_prediction

            # Update variables
            shared_active_cost = costs_dict['shared_active_cost']
            shared_passive_cost = costs_dict['shared_passive_cost']
            dedicated_active_cost = costs_dict['dedicated_active_cost']
            dedicated_passive_cost = costs_dict['dedicated_passive_cost']
            Miscellaneous_cost = costs_dict['Miscellaneous_cost']

            costs_dict = {
    'shared_active_cost': predictions.get('shared_active_cost', 0.0),
    'shared_passive_cost': predictions.get('shared_passive_cost', 0.0),
    'dedicated_active_cost': predictions.get('dedicated_active_cost', 0.0),
    'dedicated_passive_cost': predictions.get('dedicated_passive_cost', 0.0),
    'Miscellaneous_cost': predictions.get('Miscellaneous_cost', 0.0)
}

            # --- Adjust costs to match final_prediction ---
            non_zero_costs = {k: v for k, v in costs_dict.items() if v > 0}
            num_non_zero = len(non_zero_costs)

            if num_non_zero == 0:
                # All zero: assign prediction to first cost
                first_key = next(iter(costs_dict))
                costs_dict[first_key] = final_prediction
            elif num_non_zero == 1:
                # Only one non-zero: assign prediction to it
                only_key = next(iter(non_zero_costs))
                costs_dict[only_key] = final_prediction
            else:
                # Multiple non-zero: scale proportionally
                current_sum = sum(non_zero_costs.values())
                scale_factor = final_prediction / current_sum
                for k in non_zero_costs:
                    costs_dict[k] = non_zero_costs[k] * scale_factor

            # Update variables for response
            shared_active_cost = costs_dict['shared_active_cost']
            shared_passive_cost = costs_dict['shared_passive_cost']
            dedicated_active_cost = costs_dict['dedicated_active_cost']
            dedicated_passive_cost = costs_dict['dedicated_passive_cost']
            Miscellaneous_cost = costs_dict['Miscellaneous_cost']

            # ---------- Response ----------
            response = PredictionResponse(
                success=True,
                prediction=round(final_prediction, 2),
                confidence=confidence,
                breakdown=PredictionBreakdown(
                    randomForest=round(rf_pred, 2),
                    xgboost=round(xgb_pred, 2)
                ),

                shared_active_cost=round(shared_active_cost, 2),
                shared_passive_cost=round(shared_passive_cost, 2),
                dedicated_active_cost=round(dedicated_active_cost, 2),
                dedicated_passive_cost=round(dedicated_passive_cost, 2),
                Miscellaneous_cost = round(Miscellaneous_cost,2),
                timestamp=datetime.utcnow()
            )  # updated0817

            return response

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def _extract_ml_features(self, request: PredictionRequest) -> Dict[str, float]:
        """Extract ML features from equipment list (similar to Streamlit app logic)."""
        features = {
            'Antenna Counts': 0,
            'Coax Length': 0,
            'Splitter Counts': 0,
            'Jumper Counts': 0,
            'Connector Counts': 0,
            "MPROU's": 0,
            'Fiber Length': 0,
            'iBiU / POI Count': 0,
            'oDU / oEU Count': 0
        }
        
        for equipment in request.equipment:
            equipment_type = equipment.type.lower()
            description = equipment.description.lower()
            qty = equipment.qty
            
            # Map equipment to features based on type and description
            if 'antenna' in equipment_type or 'antenna' in description:
                features['Antenna Counts'] += qty
            elif 'coax' in equipment_type or 'coax' in description or 'coaxial' in description:
                features['Coax Length'] += qty
            elif 'splitter' in equipment_type or 'splitter' in description:
                features['Splitter Counts'] += qty
            elif 'jumper' in equipment_type or 'jumper' in description:
                features['Jumper Counts'] += qty
            elif 'connector' in equipment_type or 'connector' in description:
                features['Connector Counts'] += qty
            elif 'mprou' in description or 'mpr' in description:
                features["MPROU's"] += qty
            elif 'fiber' in equipment_type or 'fiber' in description or 'fibre' in description:
                features['Fiber Length'] += qty
            elif 'ibiu' in description or 'poi' in description:
                features['iBiU / POI Count'] += qty
            elif 'odu' in description or 'oeu' in description:
                features['oDU / oEU Count'] += qty
        
        return features
    
    def _prepare_input_dataframe(self, ml_features: Dict[str, float], request: PredictionRequest) -> pd.DataFrame:
        """Prepare input dataframe for model prediction."""
        # Create dataframe with ML features
        input_data = ml_features.copy()
        
        # Add installation parameters
        input_data.update({
            'Coverage Area (sqft)': request.installation.totalCoverageArea,
            'Floor Counts': request.installation.numberOfFloors,
            'Building Counts': request.installation.numberOfBuildings,
            'Average_Sqft': request.installation.totalCoverageArea,
            'Solution': request.installation.solutionType.value,
            'type_of_venue': request.installation.venueType.value
        })
        
        # Create dataframe
        df = pd.DataFrame([input_data])
        
        # Ensure all required columns are present
        required_cols = [
            'Coverage Area (sqft)', 'Floor Counts', 'Building Counts',
            'Average_Sqft', 'Solution', 'Antenna Counts', 'Coax Length',
            'Splitter Counts', 'Jumper Counts', 'Connector Counts', "MPROU's",
            'Fiber Length', 'iBiU / POI Count', 'oDU / oEU Count', 'type_of_venue'
        ]
        
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0
        
        return df[required_cols]
    
    def _calculate_confidence(self, rf_prediction: float, xgb_prediction: float) -> ConfidenceLevel:
        """
        Calculate confidence level based on agreement between models.
        
        Args:
            rf_prediction: Random Forest prediction
            xgb_prediction: XGBoost prediction
            
        Returns:
            ConfidenceLevel: Confidence level enum
        """
        # Calculate percentage difference between predictions
        avg_prediction = (rf_prediction + xgb_prediction) / 2
        if avg_prediction == 0:
            return ConfidenceLevel.LOW
        
        percentage_diff = abs(rf_prediction - xgb_prediction) / avg_prediction * 100
        
        if percentage_diff <= 10:
            return ConfidenceLevel.HIGH
        elif percentage_diff <= 25:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
