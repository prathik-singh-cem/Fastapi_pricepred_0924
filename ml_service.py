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
        
    async def initialize(self):
        """Initialize ML models for all solution types."""
        try:
            logger.info("Initializing ML models...")
            
            # Load default models (for D-RAN, Ericsson DOT, etc.)
            await self._load_model_set("default", "model_rf.pkl", "model_xgb.pkl", "model_features.pkl")
            
            # Load DAS-specific models
            await self._load_model_set("das", "das_model_rf.pkl", "das_model_xgb.pkl", "das_model_features.pkl")
            
            self.is_initialized = True
            logger.info("ML models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ML models: {e}")
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
    
   

    # # Inside your MLService class:
    # async def predict(self, request: PredictionRequest) -> PredictionResponse:
    #     #body = await request.json()
    #     body = request.dict()
    #     logger.info(f"Incoming JSON: {request.dict()}")
    #     try:
    #         if not self.is_ready():
    #             raise RuntimeError("ML service is not ready")

    #         # Extract features for full (non-shared) prediction
    #         ml_features = self._extract_ml_features(request)
    #         solution_type = request.installation.solutionType.value.lower()
    #         model_key = "das" if solution_type == "das" else "default"

    #         if model_key not in self.models:
    #             raise ValueError(f"No models available for solution type: {solution_type}")

    #         models = self.models[model_key]

    #         # Build input for standard prediction
    #         input_df = self._prepare_input_dataframe(ml_features, request)
    #         input_encoded = pd.get_dummies(input_df, columns=['Solution', 'type_of_venue'])
    #         input_aligned = input_encoded.reindex(columns=models['features'], fill_value=0)

    #         rf_pred = models['rf_model'].predict(input_aligned)[0]
    #         xgb_pred = models['xgb_model'].predict(input_aligned)[0]

    #         if solution_type == "das":
    #             final_prediction = rf_pred * 0.8 + xgb_pred * 0.2
    #         else:
    #             final_prediction = (rf_pred + xgb_pred) / 2

    #         confidence = self._calculate_confidence(rf_pred, xgb_pred)

    #         # ---------- Shared Prediction Logic ----------
    #         shared_prediction = None
    #         shared_cost = None
    #         dedicated = None

    #         # Extract shared equipment items and adjust quantities
    #         # Build shared_equipment with adjustments


    #         shared_equipment = []
    #         for item in request.equipment:
    #             if hasattr(item, 'shared') and item.shared:
    #                 # Adjust numberOfOtherOperators if equipment_type is coax/coaxial
    #                 if hasattr(item, 'equipment_type') and item.equipment_type.lower() in ['coax', 'coaxial']:
    #                     adjusted_number = item.numberOfOtherOperators
    #                 else:
    #                     adjusted_number = item.numberOfOtherOperators

    #                 adjusted_qty = item.qty / adjusted_number if adjusted_number else item.qty  # avoid division by zero
    #                 item_copy = item.copy(update={'qty': adjusted_qty})
    #             else:
    #                 item_copy = item
    #             shared_equipment.append(item_copy)


    #         if any(hasattr(item, 'shared') and item.shared for item in request.equipment):
    #             shared_request = PredictionRequest(
    #                 name=request.name,
    #                 origin=request.origin,
    #                 equipment=shared_equipment,
    #                 installation=request.installation
    #             )

    #             shared_features = self._extract_ml_features(shared_request)
    #             shared_input_df = self._prepare_input_dataframe(shared_features, shared_request)
    #             shared_input_encoded = pd.get_dummies(shared_input_df, columns=['Solution', 'type_of_venue'])
    #             shared_input_aligned = shared_input_encoded.reindex(columns=models['features'], fill_value=0)

    #             shared_rf_pred = models['rf_model'].predict(shared_input_aligned)[0]
    #             shared_xgb_pred = models['xgb_model'].predict(shared_input_aligned)[0]

    #             if solution_type == "das":
    #                 dedicated = shared_rf_pred * 0.8 + shared_xgb_pred * 0.2
    #             else:
    #                 dedicated = (shared_rf_pred + shared_xgb_pred) / 2

    #             shared_cost = final_prediction - dedicated


    #         # ---------- Passive Prediction Logic ----------
    #         passive_equipment = [item for item in request.equipment if getattr(item, 'classification', '') == 'Passive']

    #         if passive_equipment:
    #             passive_request = PredictionRequest(
    #                 name=request.name,
    #                 origin=request.origin,
    #                 equipment=passive_equipment,
    #                 installation=request.installation
    #             )

    #             passive_features = self._extract_ml_features(passive_request)
    #             passive_input_df = self._prepare_input_dataframe(passive_features, passive_request)
    #             passive_input_encoded = pd.get_dummies(passive_input_df, columns=['Solution', 'type_of_venue'])
    #             passive_input_aligned = passive_input_encoded.reindex(columns=models['features'], fill_value=0)

    #             passive_rf_pred = models['rf_model'].predict(passive_input_aligned)[0]
    #             passive_xgb_pred = models['xgb_model'].predict(passive_input_aligned)[0]

    #             if solution_type == "das":
    #                 passive_prediction = passive_rf_pred * 0.8 + passive_xgb_pred * 0.2
    #             else:
    #                 passive_prediction = (passive_rf_pred + passive_xgb_pred) / 2

    #             active_prediction = final_prediction - passive_prediction
    #         else:
    #             passive_prediction = 0.0
    #             active_prediction = final_prediction
            
    #         active_equipment = [item for item in request.equipment if getattr(item, 'classification', '') == 'Active']

    #         if len(active_equipment)==0:
    #             active_prediction = 0


    #         response = PredictionResponse(
    #             success=True,
    #             prediction=round(final_prediction, 2),
    #             confidence=confidence,
    #             breakdown=PredictionBreakdown(
    #                 randomForest=round(rf_pred, 2),
    #                 xgboost=round(xgb_pred, 2)
    #             ),
    #             dedicated=round(dedicated, 2) if dedicated else None,
    #             shared_cost=round(shared_cost, 2) if shared_cost else None,
    #             passive_prediction=round(passive_prediction, 2),
    #             active_prediction=round(active_prediction, 2),
    #             timestamp=datetime.utcnow()
    #         )#updaed0816
            
    #         return response

    #     except Exception as e:
    #         logger.error(f"Prediction failed: {e}")

    #         raise

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
                
            # First, extract unclassified equipment
            unclassified_equipment = [item for item in request.equipment 
                                    if getattr(item, 'classification', '') == 'Unclassified']

            # Predict miscellaneous cost
            if not unclassified_equipment:
                Miscellaneous_cost = 0.0
            else:
                misc = _predict_for_equipment(unclassified_equipment)
                if misc and misc > 0:
                    Miscellaneous_cost = 0.05 * misc
                else:
                    Miscellaneous_cost = 0.03 * final_prediction

            # Adjust final prediction after deducting miscellaneous cost
            final_prediction1 = final_prediction - Miscellaneous_cost



            # ---------- Shared Prediction Logic ----------
            shared_prediction = None
            shared_cost = None
            dedicated = None

            shared_equipment = []
            for item in request.equipment:
                if hasattr(item, 'shared') and item.shared:
                    if hasattr(item, 'equipment_type') and item.equipment_type.lower() in ['coax', 'coaxial']:
                        adjusted_number = item.numberOfOtherOperators + 1
                    else:
                        adjusted_number = item.numberOfOtherOperators + 1

                    adjusted_qty = item.qty / adjusted_number if adjusted_number else item.qty
                    item_copy = item.copy(update={'qty': adjusted_qty})
                else:
                    item_copy = item
                shared_equipment.append(item_copy)

            if any(hasattr(item, 'shared') and item.shared for item in request.equipment):
                shared_request = PredictionRequest(
                    name=request.name,
                    origin=request.origin,
                    equipment=shared_equipment,
                    installation=request.installation
                )

                shared_features = self._extract_ml_features(shared_request)
                shared_input_df = self._prepare_input_dataframe(shared_features, shared_request)
                shared_input_encoded = pd.get_dummies(shared_input_df, columns=['Solution', 'type_of_venue'])
                shared_input_aligned = shared_input_encoded.reindex(columns=models['features'], fill_value=0)

                shared_rf_pred = models['rf_model'].predict(shared_input_aligned)[0]
                shared_xgb_pred = models['xgb_model'].predict(shared_input_aligned)[0]

                if solution_type == "das":
                    dedicated = shared_rf_pred * 0.8 + shared_xgb_pred * 0.2
                else:
                    dedicated = (shared_rf_pred + shared_xgb_pred) / 2
                
                shared_cost = final_prediction1 - dedicated
                if shared_cost<0:
                    Miscellaneous_cost = Miscellaneous_cost+shared_cost #which will actually decrease
                    final_prediction1 = final_prediction-Miscellaneous_cost
                    dedicated = 0.8*dedicated
                    shared_cost = final_prediction - dedicated
                    
                

            # ---------- Passive Prediction Logic ----------
            passive_equipment = [item for item in request.equipment if getattr(item, 'classification', '') == 'Passive']
            active_equipment = [item for item in request.equipment if getattr(item, 'classification', '') == 'Active']

            if passive_equipment:
                passive_request = PredictionRequest(
                    name=request.name,
                    origin=request.origin,
                    equipment=passive_equipment,
                    installation=request.installation
                )

                passive_features = self._extract_ml_features(passive_request)
                passive_input_df = self._prepare_input_dataframe(passive_features, passive_request)
                passive_input_encoded = pd.get_dummies(passive_input_df, columns=['Solution', 'type_of_venue'])
                passive_input_aligned = passive_input_encoded.reindex(columns=models['features'], fill_value=0)

                passive_rf_pred = models['rf_model'].predict(passive_input_aligned)[0]
                passive_xgb_pred = models['xgb_model'].predict(passive_input_aligned)[0]

                if solution_type == "das":
                    passive_prediction = passive_rf_pred * 0.8 + passive_xgb_pred * 0.2
                else:
                    passive_prediction = (passive_rf_pred + passive_xgb_pred) / 2

                active_prediction = final_prediction1 - passive_prediction

                if not active_equipment:
                    passive_prediction = final_prediction1
                
            else:
                passive_prediction = 0.0
                active_prediction = final_prediction1

            
                        # ---------- Handle zero predictions with fallback logic ----------
            # If there's active equipment but model gave 0 active cost → fallback to 15% allocation
            if active_prediction == 0.0 and active_equipment:
                active_prediction = 0.15 * final_prediction1
                passive_prediction = final_prediction1 - active_prediction

            # If there's passive equipment but model gave 0 passive cost → fallback to 15% allocation
            if passive_prediction == 0.0 and passive_equipment:
                passive_prediction = 0.15 * final_prediction1
                active_prediction = final_prediction1 - passive_prediction

            # If neither active nor passive equipment exists → ensure predictions sum to total
            if not active_equipment and not passive_equipment:
                active_prediction = 0.0
                passive_prediction = final_prediction1


            # ---------- Fallback for Shared vs Dedicated Predictions ----------
            # Check if there is any shared equipment
            has_shared_equipment = any(getattr(item, 'shared', False) for item in request.equipment)
            has_dedicated_equipment = any(getattr(item, 'shared', True) for item in request.equipment)

            # If model predicted 0 shared cost but we actually have shared equipment → fallback to 15%
            if shared_cost == 0.0 and has_shared_equipment:
                shared_cost = 0.15 * final_prediction1
                dedicated = final_prediction1 - shared_cost

            # If there's no shared equipment, assign full cost to dedicated
            if not has_shared_equipment:
                shared_cost = 0.0
                dedicated = final_prediction1



            active_equipment = [item for item in request.equipment if getattr(item, 'classification', '') == 'Active']
            if len(active_equipment) == 0:
                active_prediction = 0


            shared_active_equipment = [item for item in request.equipment if getattr(item, 'shared', False) and getattr(item, 'classification', '') == 'Active']
            shared_passive_equipment = [item for item in request.equipment if getattr(item, 'shared', False) and getattr(item, 'classification', '') == 'Passive']
            dedicated_active_equipment = [item for item in request.equipment if not getattr(item, 'shared', False) and getattr(item, 'classification', '') == 'Active']
            dedicated_passive_equipment = [item for item in request.equipment if not getattr(item, 'shared', False) and getattr(item, 'classification', '') == 'Passive']

            shared_active_cost = _predict_for_equipment(shared_active_equipment) if shared_active_equipment else 0.0
            shared_passive_cost = _predict_for_equipment(shared_passive_equipment) if shared_passive_equipment else 0.0
            dedicated_active_cost = _predict_for_equipment(dedicated_active_equipment) if dedicated_active_equipment else 0.0
            dedicated_passive_cost = _predict_for_equipment(dedicated_passive_equipment) if dedicated_passive_equipment else 0.0


      
            total_active = active_prediction
            total_passive = passive_prediction
            total_shared = shared_cost
            total_dedicated = dedicated

            # Pro-rate active between shared vs dedicated
            if not shared_active_equipment:
                shared_active_cost = 0.0
                dedicated_active_cost = total_active
            elif not dedicated_active_equipment:
                dedicated_active_cost = 0.0
                shared_active_cost = total_active 
            else:
                total = total_shared+total_dedicated
                if total==0.0:
                    dedicated_active_cost = 0.0
                    shared_active_cost = 0.0
                else:
                    
                    shared_active_cost = (total_shared / (total_shared + total_dedicated)) * total_active if (total_shared + total_dedicated) else 0
                    dedicated_active_cost = total_active - shared_active_cost

            # Pro-rate passive between shared vs dedicated
            if not shared_passive_equipment:
                shared_passive_cost = 0.0
                dedicated_passive_cost = total_passive
            elif not dedicated_passive_cost:
                dedicated_passive_cost = 0.0
                shared_passive_cost = total_passive 
            else:
                total = total_shared+total_dedicated
                if total==0.0:
                    shared_passive_cost = 0.0
                    dedicated_passive_cost = 0.0
                else:
                    shared_passive_cost = (total_shared / (total_shared + total_dedicated)) * total_passive if (total_shared + total_dedicated) else 0
                    dedicated_passive_cost = total_passive - shared_passive_cost
            #comment till here 0818


            if not shared_active_equipment and not shared_passive_equipment and not dedicated_active_equipment :
                dedicated_passive_cost= final_prediction1
            
            if not dedicated_passive_equipment and not shared_passive_equipment and not dedicated_active_equipment :
                shared_active_cost= final_prediction1

            if not dedicated_passive_equipment and not shared_active_equipment and not dedicated_active_equipment :
                shared_passive_cost= final_prediction1

            if not shared_active_equipment and not shared_passive_equipment and not  dedicated_passive_equipment:
                dedicated_active_cost= final_prediction1

            # For passive equipment
            if shared_passive_equipment and dedicated_passive_equipment:
                if unclassified_equipment:
                    shared_active_cost = final_prediction1 * 0.30
                    dedicated_active_cost = final_prediction1 * 0.65
                    Miscellaneous_cost = final_prediction1*0.05

                if shared_passive_cost == 0.0 or dedicated_passive_cost == 0.0:
                    shared_passive_cost = final_prediction1 * 0.30
                    dedicated_passive_cost = final_prediction1 * 0.70

            # For active equipment
            if shared_active_equipment and dedicated_active_equipment:

                if shared_active_cost == 0.0 or dedicated_active_cost == 0.0:
                    if unclassified_equipment and Miscellaneous_cost==0.0:
                        shared_active_cost = final_prediction1 * 0.30
                        dedicated_active_cost = final_prediction1 * 0.65
                        Miscellaneous_cost = final_prediction1*0.05


                    else:
                        shared_active_cost = final_prediction1 * 0.30
                        dedicated_active_cost = final_prediction1 * 0.70


            if not unclassified_equipment:
                Miscellaneous_cost = 0.0
            else:
                misc = _predict_for_equipment(unclassified_equipment)
                if misc and misc > 0 and misc==final_prediction:
                    Miscellaneous_cost = 0.05 * misc
                    if shared_active_cost and (shared_active_cost-Miscellaneous_cost>0):
                        shared_active_cost = shared_active_cost-Miscellaneous_cost
                    elif dedicated_active_cost and (dedicated_active_cost-Miscellaneous_cost>0):
                        dedicated_active_cost = dedicated_active_cost-Miscellaneous_cost
                    elif shared_passive_cost and (shared_passive_cost-Miscellaneous_cost>0):
                        shared_passive_cost = shared_passive_cost-Miscellaneous_cost
                    elif dedicated_passive_cost and (dedicated_passive_cost-Miscellaneous_cost>0):
                        dedicated_passive_cost = dedicated_passive_cost-Miscellaneous_cost

                elif misc > 0:
                    Miscellaneous_cost = Miscellaneous_cost
                else:
                    Miscellaneous_cost = 0.002 * final_prediction
                    if shared_active_cost and (shared_active_cost-Miscellaneous_cost>0):
                        shared_active_cost = shared_active_cost-Miscellaneous_cost
                    elif dedicated_active_cost and (dedicated_active_cost-Miscellaneous_cost>0):
                        dedicated_active_cost = dedicated_active_cost-Miscellaneous_cost
                    elif shared_passive_cost and (shared_passive_cost-Miscellaneous_cost>0):
                        shared_passive_cost = shared_passive_cost-Miscellaneous_cost
                    elif dedicated_passive_cost and (dedicated_passive_cost-Miscellaneous_cost>0):
                        dedicated_passive_cost = dedicated_passive_cost-Miscellaneous_cost                    
         

            # ---------- Response ----------
            response = PredictionResponse(
                success=True,
                prediction=round(final_prediction, 2),
                confidence=confidence,
                breakdown=PredictionBreakdown(
                    randomForest=round(rf_pred, 2),
                    xgboost=round(xgb_pred, 2)
                ),
                dedicated=round(dedicated, 2) if dedicated else None,
                shared_cost=round(shared_cost, 2) if shared_cost else None,
                passive_prediction=round(passive_prediction, 2),
                active_prediction=round(active_prediction, 2),
                shared_active_cost=round(shared_active_cost, 2),
                shared_passive_cost=round(shared_passive_cost, 2),
                dedicated_active_cost=round(dedicated_active_cost, 2),
                dedicated_passive_cost=round(dedicated_passive_cost, 2),
                Miscellaneous_cost = round(Miscellaneous_cost,2),
                timestamp=datetime.utcnow()
            )  # updated0817

            response = PredictionResponse(
                success=True,
                prediction=round(final_prediction, 2),
                confidence=confidence,
                breakdown=PredictionBreakdown(
                    randomForest=round(rf_pred, 2),
                    xgboost=round(xgb_pred, 2)
                ),
                dedicated=round(dedicated, 2) if has_dedicated_equipment else 0.0,
                shared_cost=round(shared_cost, 2) if has_shared_equipment else 0.0,
                passive_prediction=round(passive_prediction, 2) if passive_equipment else 0.0,
                active_prediction=round(active_prediction, 2) if active_equipment else 0.0,
                shared_active_cost=round(shared_active_cost, 2) if shared_active_equipment else 0.0,
                shared_passive_cost=round(shared_passive_cost, 2) if shared_passive_equipment else 0.0,
                dedicated_active_cost=round(dedicated_active_cost, 2) if dedicated_active_equipment else 0.0,
                dedicated_passive_cost=round(dedicated_passive_cost, 2) if dedicated_passive_equipment else 0.0,
                Miscellaneous_cost=round(Miscellaneous_cost, 2) if unclassified_equipment else 0.0,
                timestamp=datetime.utcnow()
            )


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
