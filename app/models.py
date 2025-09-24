"""
Pydantic models for T-Mobile Installation Cost Prediction API
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Union
from datetime import datetime
from enum import Enum

class EquipmentType(str, Enum):
    """Enumeration of equipment types."""
    CABLE = "Cable"
    CONNECTOR = "Connector"
    NETWORK_EQUIPMENT = "Network Equipment"
    RADIO_TRANSCEIVER = "Radio Transceiver"
    ANTENNA = "Antenna"
    SPLITTER = "Splitter"
    JUMPER = "Jumper"
    FIBER = "Fiber"
    OTHER = "Other"


# class VenueType(str, Enum):
#     """Enumeration of venue types."""
#     INDUSTRIAL = "industrial"
#     COMMERCIAL = "commercial"
#     RESIDENTIAL = "residential"
#     EDUCATIONAL = "educational"
#     HEALTHCARE = "healthcare"
#     HOSPITALITY = "hospitality"
#     RETAIL = "retail"
#     OFFICE = "office"

class VenueType(str, Enum):
    """Enumeration of venue types."""

    # Existing types
    INDUSTRIAL = "industrial"
    COMMERCIAL = "commercial"
    RESIDENTIAL = "residential"
    EDUCATIONAL = "educational"
    HEALTHCARE = "healthcare"
    HOSPITALITY = "hospitality"
    RETAIL = "retail"
    OFFICE = "office"
    HOSPITAL = "hospital"
    CONVENTION_CENTER = "convention center"
    WAREHOUSE = "warehouse"
    AMPHITHEATER = "amphitheater"
    FACTORY = "factory"
    PARKING = "parking"
    DENSE_INDUSTRIAL = "dense industrial"
    OFFICE_WAREHOUSE = "office / warehouse"
    CAMPUS = "campus"
    STADIUM = "stadium"
    HOTEL = "hotel"
    OFFICE_PARKING = "office / parking"
    RESORT = "resort"
    GOVERNMENT_OFFICE = "government / office"

    
    @classmethod
    def _missing_(cls, value):
        """Allow case-insensitive matching for venue types"""
        if isinstance(value, str):
            value = value.lower()
            for member in cls:
                if member.value.lower() == value:
                    return member
        return None



class SolutionType(str, Enum):
    """Enumeration of solution types."""
    DAS = "das"
    D_RAN = "d-ran"
    ERICSSON_DOT = "ericsson dot"


class EquipmentItem(BaseModel):
    type: str
    componentGroup: str
    manufacturer: str
    model: str
    description: str
    inventory: str
    qty: float
    qtyType: Optional[str]
    shared: Optional[bool]
    numberOfOtherOperators: Optional[int]
    classification: Optional[str]  # "Passive", "Active", or "Unclassified"
    #updated0816


class EquipmentClassification(str, Enum):
    passive = "Passive"
    active = "Active"
    unclassified = "Unclassified"


class Equipment(BaseModel):
    """Equipment item model."""

    type: str = Field(..., description="Type of equipment")
    componentGroup: str = Field(default="", description="Component group classification")
    manufacturer: str = Field(..., description="Equipment manufacturer")
    model: str = Field(..., description="Equipment model")
    description: str = Field(..., description="Equipment description")
    inventory: str = Field(default="N/A", description="Inventory identifier")
    qty: Union[int, float] = Field(..., gt=0, description="Quantity of equipment")
    qtyType: Optional[str] = Field(default=None, description="Unit type for quantity")
    shared: bool = Field(..., description="Whether the equipment is shared")
    numberOfOtherOperators: int = Field(..., ge=1, le=4, description="Number of other operators sharing the equipment")

    classification: Optional[EquipmentClassification] = Field(
        default=None,
        description='Classification of equipment as "Passive", "Active", or "Unclassified"'
    )

    @validator('qty')
    def validate_quantity(cls, v):
        if v <= 0:
            raise ValueError('Quantity must be greater than 0')
        return v
     #updated0816



class Installation(BaseModel):
    """Installation parameters model."""
    venueType: VenueType = Field(..., description="Type of venue")
    solutionType: SolutionType = Field(..., description="Solution type")
    totalCoverageArea: float = Field(..., gt=0, description="Total coverage area in square feet")
    numberOfFloors: int = Field(..., gt=0, description="Number of floors")
    numberOfBuildings: int = Field(..., gt=0, description="Number of buildings")

    @validator('totalCoverageArea')
    def validate_coverage_area(cls, v):
        if v <= 0:
            raise ValueError('Coverage area must be greater than 0')
        return v

    @validator('numberOfFloors')
    def validate_floors(cls, v):
        if v <= 0:
            raise ValueError('Number of floors must be greater than 0')
        return v

    @validator('numberOfBuildings')
    def validate_buildings(cls, v):
        if v <= 0:
            raise ValueError('Number of buildings must be greater than 0')
        return v



class PredictionRequest(BaseModel):
    """Request model for installation cost prediction."""
    name: str = Field(..., description="Project name")
    origin: str = Field(..., description="Source file or origin identifier")
    equipment: List[Equipment] = Field(..., min_items=1, description="List of equipment items")
    installation: Installation = Field(..., description="Installation parameters")

    @validator('equipment')
    def validate_equipment_list(cls, v):
        if not v:
            raise ValueError('Equipment list cannot be empty')
        return v

    class Config:
        schema_extra = {
            "example": {
                "name": "Campus University of Virginia VAYH050A Library",
                "origin": "Campus_University of Virginia_VAYH050A_Library.xlsx",
                "equipment": [
                    {
                        "type": "Cable",
                        "componentGroup": "",
                        "manufacturer": "Generic",
                        "model": "CAT-6",
                        "description": "CAT-6 - 24 AWG min. - 100m Maximum Cable Length",
                        "inventory": "N/A",
                        "qty": 770.35,
                        "qtyType": "feet",
                        "shared": True,
                        "numberOfOtherOperators": 2,
                        "classification": "Passive"
                    }
                ],
                "installation": {
                    "venueType": "industrial",
                    "solutionType": "das",
                    "totalCoverageArea": 10000,
                    "numberOfFloors": 1,
                    "numberOfBuildings": 1
                }
            }
        }


class PredictionBreakdown(BaseModel):
    """Breakdown of prediction results."""
    randomForest: float = Field(..., description="Random Forest model prediction")
    xgboost: float = Field(..., description="XGBoost model prediction")


class ConfidenceLevel(str, Enum):
    """Enumeration of confidence levels."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"



# class PredictionResponse(BaseModel):
#     """Response model for installation cost prediction."""
    
#     success: bool = Field(..., description="Whether prediction was successful")
#     prediction: float = Field(..., description="Final predicted installation cost")
#     confidence: ConfidenceLevel = Field(..., description="Confidence level of prediction")
#     breakdown: PredictionBreakdown = Field(..., description="Breakdown of model predictions")
#     timestamp: datetime = Field(default_factory=datetime.utcnow, description="Prediction timestamp")

#     dedicated: Optional[float] = Field(
#         default=None,
#         description="Predicted cost for shared deployment"
#     )
#     shared_cost: Optional[float] = Field(
#         default=None,
#         description="Cost savings due to sharing"
#     )
    
#     passive_prediction: Optional[float] = Field(
#         default=None,
#         description="Predicted cost contribution from passive components only"
#     )
    
#     active_prediction: Optional[float] = Field(
#         default=None,
#         description="Predicted cost contribution from active components only"
#     )
    

#     class Config:
#         json_schema_extra = {
#             "example": {
#                 "success": True,
#                 "prediction": 332280,
#                 "confidence": "Medium",
#                 "breakdown": {
#                     "randomForest": 315000,
#                     "xgboost": 358200
#                 },
#                 "timestamp": "2025-07-23T10:58:03.900Z",
#                 "dedicated": 280000,
#                 "shared_cost": 52280,
#                 "passive_prediction": 128000,
#                 "active_prediction": 204280
#             }
#         }#updated0816


class PredictionResponse(BaseModel):
    """Response model for installation cost prediction."""
    
    success: bool = Field(..., description="Whether prediction was successful")
    prediction: float = Field(..., description="Final predicted installation cost")
    confidence: ConfidenceLevel = Field(..., description="Confidence level of prediction")
    breakdown: PredictionBreakdown = Field(..., description="Breakdown of model predictions")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Prediction timestamp")


    shared_active_cost: Optional[float] = Field(
        default=None,
        description="Predicted cost contribution from shared active equipment"
    )

    shared_passive_cost: Optional[float] = Field(
        default=None,
        description="Predicted cost contribution from shared passive equipment"
    )

    dedicated_active_cost: Optional[float] = Field(
        default=None,
        description="Predicted cost contribution from dedicated active equipment"
    )

    dedicated_passive_cost: Optional[float] = Field(
        default=None,
        description="Predicted cost contribution from dedicated passive equipment"
    )
    Miscellaneous_cost: Optional[float] = Field(
        default=None,
        description="Miscellaneous_cost from unclassified "
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "prediction": 332280,
                "confidence": "Medium",
                "breakdown": {
                    "randomForest": 315000,
                    "xgboost": 358200
                },
                "timestamp": "2025-07-23T10:58:03.900Z",
                "dedicated": 280000,
                "shared_cost": 52280,
                "passive_prediction": 128000,
                "active_prediction": 204280,
                "shared_active_cost": 50000,
                "shared_passive_cost": 30000,
                "dedicated_active_cost": 120000,
                "dedicated_passive_cost": 132280,
                "Miscellaneous_cost" : 0.0
            }
        } #updated0817








class TokenResponse(BaseModel):
    """Response model for authentication token."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")

    class Config:
        schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 900
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(default=None, description="Error code")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")

    class Config:
        schema_extra = {
            "example": {
                "detail": "Invalid authentication credentials",
                "error_code": "AUTH_001",
                "timestamp": "2025-07-23T10:58:03.900Z"
            }
        }
