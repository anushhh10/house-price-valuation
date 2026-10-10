"""
Domain-Specific Multi-Agent Skeletons for Real Estate Evaluation.

Defines modular agents (Infrastructure, Safety, Environment, Growth)
responsible for evaluating localized societal factors, scoring dimensions,
and providing domain-specific explainable rationale with dynamic buyer profiles
(Family, Investor, Student, Working Professional).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Union
import numpy as np
import pandas as pd

# Supported Buyer Profiles
VALID_BUYER_PROFILES = ["Family", "Investor", "Student", "Working Professional"]

# Global Profile Preferences across the 4 Dimensions (Sums to 1.0)
PROFILE_DIMENSION_WEIGHTS = {
    "Family": {
        "Infrastructure": 0.30,
        "Safety": 0.35,
        "Environment": 0.25,
        "Growth": 0.10,
    },
    "Investor": {
        "Infrastructure": 0.15,
        "Safety": 0.15,
        "Environment": 0.10,
        "Growth": 0.60,
    },
    "Student": {
        "Infrastructure": 0.40,
        "Safety": 0.30,
        "Environment": 0.20,
        "Growth": 0.10,
    },
    "Working Professional": {
        "Infrastructure": 0.35,
        "Safety": 0.20,
        "Environment": 0.20,
        "Growth": 0.25,
    },
}


@dataclass
class AgentEvaluation:
    """Standardized evaluation report returned by each domain agent."""
    agent_name: str
    dimension_score: float  # Normalized score (0.0 to 100.0)
    rating_category: str     # e.g., 'Poor', 'Fair', 'Good', 'Excellent'
    price_impact_factor: float  # Multiplier contribution (e.g. 0.85 to 1.20)
    buyer_profile: str       # Associated profile
    key_drivers: Dict[str, Any] = field(default_factory=dict)
    insights: List[str] = field(default_factory=list)


class BaseAgent(ABC):
    """Abstract base class for all localized domain assessment agents."""

    def __init__(self, name: str, description: str, relevant_features: List[str]):
        self.name = name
        self.description = description
        self.relevant_features = relevant_features

    @abstractmethod
    def evaluate(
        self,
        property_data: Union[pd.Series, Dict[str, Any]],
        buyer_profile: str = "Family",
    ) -> AgentEvaluation:
        """
        Evaluate property metrics relevant to this agent's domain,
        tailored dynamically to the buyer's persona.
        """
        pass

    @abstractmethod
    def get_feature_importances(self, buyer_profile: str = "Family") -> Dict[str, float]:
        """Return the internal domain weight distribution for the given buyer profile."""
        pass

    def _categorize_score(self, score: float) -> str:
        """Helper to convert a 0-100 score into a descriptive category."""
        if score >= 80.0:
            return "Excellent"
        elif score >= 65.0:
            return "Good"
        elif score >= 50.0:
            return "Moderate"
        elif score >= 35.0:
            return "Fair"
        return "Poor"


class InfrastructureAgent(BaseAgent):
    """
    Evaluates accessibility, civic amenities, transit connectivity,
    and educational institution quality with buyer-profile weightings.
    """

    def __init__(self):
        super().__init__(
            name="InfrastructureAgent",
            description="Assesses civic infrastructure, transit links, and educational institutions.",
            relevant_features=["school_score", "hospital_distance_km", "transit_score"],
        )

    def get_feature_importances(self, buyer_profile: str = "Family") -> Dict[str, float]:
        if buyer_profile == "Family":
            return {"school_score": 0.50, "transit_score": 0.25, "hospital_distance_km": 0.25}
        elif buyer_profile == "Investor":
            return {"school_score": 0.25, "transit_score": 0.55, "hospital_distance_km": 0.20}
        elif buyer_profile == "Student":
            return {"school_score": 0.10, "transit_score": 0.70, "hospital_distance_km": 0.20}
        elif buyer_profile == "Working Professional":
            return {"school_score": 0.15, "transit_score": 0.65, "hospital_distance_km": 0.20}
        return {"school_score": 0.40, "transit_score": 0.40, "hospital_distance_km": 0.20}

    def evaluate(
        self,
        property_data: Union[pd.Series, Dict[str, Any]],
        buyer_profile: str = "Family",
    ) -> AgentEvaluation:
        school = float(property_data.get("school_score", 5.0))
        transit = float(property_data.get("transit_score", 5.0))
        hospital_dist = float(property_data.get("hospital_distance_km", 5.0))

        # Normalized to 0-100 scale
        school_norm = np.clip(school * 10.0, 0.0, 100.0)
        transit_norm = np.clip(transit * 10.0, 0.0, 100.0)
        hospital_norm = np.clip(100.0 - (hospital_dist / 12.0) * 100.0, 0.0, 100.0)

        weights = self.get_feature_importances(buyer_profile)
        score = (
            school_norm * weights["school_score"]
            + transit_norm * weights["transit_score"]
            + hospital_norm * weights["hospital_distance_km"]
        )
        score = round(float(np.clip(score, 0.0, 100.0)), 1)
        category = self._categorize_score(score)

        # Dynamic sensitivity per buyer profile
        sensitivity_map = {
            "Family": 0.14,
            "Investor": 0.08,
            "Student": 0.15,
            "Working Professional": 0.13,
        }
        sensitivity = sensitivity_map.get(buyer_profile, 0.12)
        price_impact = round(1.0 + ((score - 50.0) / 50.0) * sensitivity, 3)

        insights = []
        if buyer_profile == "Family":
            if school >= 7.5:
                insights.append(f"High school quality rating ({school}/10) matches prime family settlement criteria.")
            else:
                insights.append(f"Sub-optimal school rating ({school}/10) poses a limitation for school-age children.")

        if buyer_profile in ["Student", "Working Professional"]:
            if transit >= 7.5:
                insights.append(f"Rapid transit access ({transit}/10) dramatically slashes daily commute overhead.")
            elif transit < 5.0:
                insights.append(f"Inferior transit score ({transit}/10) requires reliance on private vehicles.")

        if hospital_dist <= 2.5:
            insights.append(f"Rapid emergency medical access: hospital within {hospital_dist} km.")

        return AgentEvaluation(
            agent_name=self.name,
            dimension_score=score,
            rating_category=category,
            price_impact_factor=price_impact,
            buyer_profile=buyer_profile,
            key_drivers={
                "school_score": school,
                "transit_score": transit,
                "hospital_distance_km": hospital_dist,
            },
            insights=insights,
        )


class SafetyAgent(BaseAgent):
    """
    Evaluates neighborhood crime incidence and law enforcement responsiveness
    with heightened sensitivity for families and students.
    """

    def __init__(self):
        super().__init__(
            name="SafetyAgent",
            description="Assesses neighborhood security, incident rates, and law enforcement proximity.",
            relevant_features=["crime_rate", "police_proximity_km"],
        )

    def get_feature_importances(self, buyer_profile: str = "Family") -> Dict[str, float]:
        if buyer_profile in ["Family", "Student"]:
            return {"crime_rate": 0.75, "police_proximity_km": 0.25}
        elif buyer_profile == "Investor":
            return {"crime_rate": 0.60, "police_proximity_km": 0.40}
        return {"crime_rate": 0.70, "police_proximity_km": 0.30}

    def evaluate(
        self,
        property_data: Union[pd.Series, Dict[str, Any]],
        buyer_profile: str = "Family",
    ) -> AgentEvaluation:
        crime_rate = float(property_data.get("crime_rate", 45.0))
        police_dist = float(property_data.get("police_proximity_km", 4.0))

        crime_safety_norm = np.clip(100.0 - crime_rate, 0.0, 100.0)
        police_norm = np.clip(100.0 - (police_dist / 10.0) * 100.0, 0.0, 100.0)

        weights = self.get_feature_importances(buyer_profile)
        score = (
            crime_safety_norm * weights["crime_rate"]
            + police_norm * weights["police_proximity_km"]
        )
        score = round(float(np.clip(score, 0.0, 100.0)), 1)
        category = self._categorize_score(score)

        # Families and Students are significantly more price-sensitive to safety
        sensitivity_map = {
            "Family": 0.18,
            "Student": 0.16,
            "Working Professional": 0.12,
            "Investor": 0.09,
        }
        sensitivity = sensitivity_map.get(buyer_profile, 0.14)
        price_impact = round(1.0 + ((score - 50.0) / 50.0) * sensitivity, 3)

        insights = []
        if crime_rate <= 25.0:
            insights.append(f"Low crime index ({crime_rate}) delivers peace of mind for {buyer_profile.lower()} occupants.")
        elif crime_rate >= 60.0:
            penalty_note = "Severe barrier" if buyer_profile in ["Family", "Student"] else "Considerable risk factor"
            insights.append(f"{penalty_note}: elevated neighborhood incident frequency ({crime_rate}).")

        if police_dist <= 2.0:
            insights.append(f"Prompt law enforcement dispatch available ({police_dist} km to station).")

        return AgentEvaluation(
            agent_name=self.name,
            dimension_score=score,
            rating_category=category,
            price_impact_factor=price_impact,
            buyer_profile=buyer_profile,
            key_drivers={
                "crime_rate": crime_rate,
                "police_proximity_km": police_dist,
            },
            insights=insights,
        )


class EnvironmentAgent(BaseAgent):
    """
    Evaluates ecological factors, air quality (AQI), tree canopy / green cover,
    and ambient acoustic levels.
    """

    def __init__(self):
        super().__init__(
            name="EnvironmentAgent",
            description="Assesses ambient air quality, surrounding green cover, and acoustic wellness.",
            relevant_features=["aqi", "green_cover_pct", "noise_level_db"],
        )

    def get_feature_importances(self, buyer_profile: str = "Family") -> Dict[str, float]:
        if buyer_profile == "Family":
            return {"aqi": 0.50, "green_cover_pct": 0.35, "noise_level_db": 0.15}
        elif buyer_profile == "Working Professional":
            return {"aqi": 0.35, "green_cover_pct": 0.25, "noise_level_db": 0.40}
        elif buyer_profile == "Student":
            return {"aqi": 0.35, "green_cover_pct": 0.25, "noise_level_db": 0.40}
        return {"aqi": 0.45, "green_cover_pct": 0.35, "noise_level_db": 0.20}

    def evaluate(
        self,
        property_data: Union[pd.Series, Dict[str, Any]],
        buyer_profile: str = "Family",
    ) -> AgentEvaluation:
        aqi = float(property_data.get("aqi", 100.0))
        green_cover = float(property_data.get("green_cover_pct", 25.0))
        noise = float(property_data.get("noise_level_db", 60.0))

        aqi_norm = np.clip(100.0 - ((aqi - 30.0) / 220.0) * 100.0, 0.0, 100.0)
        green_norm = np.clip((green_cover / 50.0) * 100.0, 0.0, 100.0)
        noise_norm = np.clip(100.0 - ((noise - 40.0) / 45.0) * 100.0, 0.0, 100.0)

        weights = self.get_feature_importances(buyer_profile)
        score = (
            aqi_norm * weights["aqi"]
            + green_norm * weights["green_cover_pct"]
            + noise_norm * weights["noise_level_db"]
        )
        score = round(float(np.clip(score, 0.0, 100.0)), 1)
        category = self._categorize_score(score)

        sensitivity_map = {
            "Family": 0.14,
            "Working Professional": 0.12,
            "Student": 0.08,
            "Investor": 0.06,
        }
        sensitivity = sensitivity_map.get(buyer_profile, 0.11)
        price_impact = round(1.0 + ((score - 50.0) / 50.0) * sensitivity, 3)

        insights = []
        if aqi <= 55:
            insights.append(f"Crisp, healthy air quality (AQI {int(aqi)}) promotes longevity and wellness.")
        elif aqi > 150:
            insights.append(f"Substantial atmospheric particulate load (AQI {int(aqi)}).")

        if noise > 70.0 and buyer_profile in ["Working Professional", "Student"]:
            insights.append(f"Acoustic noise ({noise} dB) may disrupt remote work focus or quiet study.")
        elif noise <= 48.0:
            insights.append(f"Serene acoustic profile ({noise} dB) ideal for restorative living.")

        if green_cover >= 35.0:
            insights.append(f"Abundant foliage and parks ({green_cover}%) enhance outdoor recreation.")

        return AgentEvaluation(
            agent_name=self.name,
            dimension_score=score,
            rating_category=category,
            price_impact_factor=price_impact,
            buyer_profile=buyer_profile,
            key_drivers={
                "aqi": aqi,
                "green_cover_pct": green_cover,
                "noise_level_db": noise,
            },
            insights=insights,
        )


class GrowthAgent(BaseAgent):
    """
    Evaluates capital appreciation momentum, upcoming civic/commercial projects,
    and proximity to economic employment hubs. Heavily weighted for Investors.
    """

    def __init__(self):
        super().__init__(
            name="GrowthAgent",
            description="Assesses neighborhood appreciation velocity, commercial expansion, and investment upside.",
            relevant_features=["growth_index", "commercial_hub_distance_km"],
        )

    def get_feature_importances(self, buyer_profile: str = "Family") -> Dict[str, float]:
        if buyer_profile == "Investor":
            return {"growth_index": 0.75, "commercial_hub_distance_km": 0.25}
        elif buyer_profile == "Working Professional":
            return {"growth_index": 0.50, "commercial_hub_distance_km": 0.50}
        return {"growth_index": 0.65, "commercial_hub_distance_km": 0.35}

    def evaluate(
        self,
        property_data: Union[pd.Series, Dict[str, Any]],
        buyer_profile: str = "Family",
    ) -> AgentEvaluation:
        growth_idx = float(property_data.get("growth_index", 50.0))
        hub_dist = float(property_data.get("commercial_hub_distance_km", 7.0))

        growth_norm = np.clip(growth_idx, 0.0, 100.0)
        hub_norm = np.clip(100.0 - (hub_dist / 20.0) * 100.0, 0.0, 100.0)

        weights = self.get_feature_importances(buyer_profile)
        score = (
            growth_norm * weights["growth_index"]
            + hub_norm * weights["commercial_hub_distance_km"]
        )
        score = round(float(np.clip(score, 0.0, 100.0)), 1)
        category = self._categorize_score(score)

        # Investors have high multiplier sensitivity to growth potential
        sensitivity_map = {
            "Investor": 0.25,
            "Working Professional": 0.15,
            "Family": 0.08,
            "Student": 0.05,
        }
        sensitivity = sensitivity_map.get(buyer_profile, 0.12)
        price_impact = round(1.0 + ((score - 50.0) / 50.0) * sensitivity, 3)

        insights = []
        if buyer_profile == "Investor":
            if growth_idx >= 70.0:
                insights.append(f"High-alpha appreciation corridor (Growth Index {growth_idx}) with robust ROI potential.")
            else:
                insights.append(f"Subdued capital growth momentum ({growth_idx}/100) indicates conservative yields.")

        if hub_dist <= 3.5:
            insights.append(f"Direct corridor access to major commercial tech hub ({hub_dist} km).")
        elif hub_dist > 15.0:
            insights.append(f"Peripheral location ({hub_dist} km to CBD/hub) may dampen tenant rental demand.")

        return AgentEvaluation(
            agent_name=self.name,
            dimension_score=score,
            rating_category=category,
            price_impact_factor=price_impact,
            buyer_profile=buyer_profile,
            key_drivers={
                "growth_index": growth_idx,
                "commercial_hub_distance_km": hub_dist,
            },
            insights=insights,
        )


class MultiAgentOrchestrator:
    """
    Coordinates all specialized agents to compute composite locational ratings,
    dynamic profile-adjusted multipliers, and aggregated explainability payloads.
    """

    def __init__(self):
        self.infra_agent = InfrastructureAgent()
        self.safety_agent = SafetyAgent()
        self.env_agent = EnvironmentAgent()
        self.growth_agent = GrowthAgent()
        self.agents = [
            self.infra_agent,
            self.safety_agent,
            self.env_agent,
            self.growth_agent,
        ]

    def evaluate_property(
        self,
        property_data: Union[pd.Series, Dict[str, Any]],
        buyer_profile: str = "Family",
    ) -> Dict[str, Any]:
        """
        Run all domain agents over a given property profile and calibrate
        composite score and valuation multiplier to the buyer persona.

        Parameters:
        -----------
        property_data : pd.Series or Dict[str, Any]
            Property metrics.
        buyer_profile : str
            One of ['Family', 'Investor', 'Student', 'Working Professional'].

        Returns:
        --------
        Dict[str, Any]
            Structured evaluation report with radar metrics, profile multipliers,
            and persona-tailored insights.
        """
        if buyer_profile not in VALID_BUYER_PROFILES:
            buyer_profile = "Family"

        evaluations: Dict[str, AgentEvaluation] = {}
        dim_weights = PROFILE_DIMENSION_WEIGHTS[buyer_profile]
        
        weighted_score_sum = 0.0
        multipliers = []
        all_insights = []

        dim_mapping = {
            "InfrastructureAgent": "Infrastructure",
            "SafetyAgent": "Safety",
            "EnvironmentAgent": "Environment",
            "GrowthAgent": "Growth",
        }

        for agent in self.agents:
            res = agent.evaluate(property_data, buyer_profile=buyer_profile)
            evaluations[agent.name] = res
            dim_key = dim_mapping[agent.name]
            w = dim_weights.get(dim_key, 0.25)
            weighted_score_sum += res.dimension_score * w
            multipliers.append(res.price_impact_factor)
            all_insights.extend(res.insights)

        composite_score = round(float(weighted_score_sum), 1)

        # Profile-adjusted composite multiplier: blend geometric mean with dimension weights
        log_mult = sum(
            np.log(evaluations[agent.name].price_impact_factor) * dim_weights[dim_mapping[agent.name]]
            for agent in self.agents
        )
        composite_multiplier = round(float(np.exp(log_mult)), 3)

        radar_metrics = {
            "Infrastructure": evaluations["InfrastructureAgent"].dimension_score,
            "Safety": evaluations["SafetyAgent"].dimension_score,
            "Environment": evaluations["EnvironmentAgent"].dimension_score,
            "Growth": evaluations["GrowthAgent"].dimension_score,
        }

        agent_multipliers = {
            "Infrastructure": evaluations["InfrastructureAgent"].price_impact_factor,
            "Safety": evaluations["SafetyAgent"].price_impact_factor,
            "Environment": evaluations["EnvironmentAgent"].price_impact_factor,
            "Growth": evaluations["GrowthAgent"].price_impact_factor,
        }

        return {
            "buyer_profile": buyer_profile,
            "composite_score": composite_score,
            "composite_multiplier": composite_multiplier,
            "dimension_weights": dim_weights,
            "radar_metrics": radar_metrics,
            "agent_multipliers": agent_multipliers,
            "agent_evaluations": evaluations,
            "all_insights": all_insights,
        }
