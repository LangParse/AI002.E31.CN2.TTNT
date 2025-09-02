"""
Drug interaction checking utilities for AI Medication Reminder.

Implements basic drug interaction checking according to pipeline specification A.1.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


class DrugInteractionChecker:
    """Checks for drug interactions and contraindications."""

    def __init__(self, interactions_db_path: Optional[Path] = None):
        """
        Initialize drug interaction checker.

        Args:
            interactions_db_path: Path to drug interactions database file
        """
        self.interactions_db = {}
        self.contraindications_db = {}

        if interactions_db_path and interactions_db_path.exists():
            self.load_interactions_db(interactions_db_path)
        else:
            # Initialize with basic mock data for demonstration
            self._initialize_mock_db()

    def _initialize_mock_db(self) -> None:
        """Initialize with mock drug interaction data for demonstration."""
        # Mock drug interactions database
        self.interactions_db = {
            "warfarin": {
                "aspirin": {
                    "severity": "major",
                    "description": "Increased bleeding risk",
                },
                "ibuprofen": {
                    "severity": "moderate",
                    "description": "Increased bleeding risk",
                },
                "acetaminophen": {"severity": "minor", "description": "Monitor INR"},
            },
            "metformin": {
                "alcohol": {
                    "severity": "moderate",
                    "description": "Risk of lactic acidosis",
                },
                "contrast_dye": {
                    "severity": "major",
                    "description": "Risk of kidney damage",
                },
            },
            "lisinopril": {
                "potassium": {
                    "severity": "moderate",
                    "description": "Risk of hyperkalemia",
                },
                "nsaids": {
                    "severity": "moderate",
                    "description": "Reduced effectiveness",
                },
            },
            "simvastatin": {
                "grapefruit": {
                    "severity": "major",
                    "description": "Increased statin levels",
                },
                "cyclosporine": {
                    "severity": "major",
                    "description": "Risk of myopathy",
                },
            },
        }

        # Mock contraindications database
        self.contraindications_db = {
            "aspirin": {
                "conditions": ["active_bleeding", "severe_liver_disease"],
                "age_restrictions": {"min_age": None, "max_age": None},
                "pregnancy": "avoid_third_trimester",
            },
            "warfarin": {
                "conditions": ["active_bleeding", "severe_liver_disease", "pregnancy"],
                "age_restrictions": {"min_age": None, "max_age": None},
                "pregnancy": "contraindicated",
            },
            "metformin": {
                "conditions": ["kidney_disease", "liver_disease", "heart_failure"],
                "age_restrictions": {"min_age": None, "max_age": 80},
                "pregnancy": "category_b",
            },
        }

    def load_interactions_db(self, db_path: Path) -> None:
        """
        Load drug interactions database from file.

        Args:
            db_path: Path to database file
        """
        try:
            with open(db_path, "r") as f:
                db_data = json.load(f)
                self.interactions_db = db_data.get("interactions", {})
                self.contraindications_db = db_data.get("contraindications", {})
            print(f"Loaded drug interactions database from {db_path}")
        except Exception as e:
            print(f"Failed to load database: {str(e)}")
            self._initialize_mock_db()

    def check_drug_interactions(self, medications: List[str]) -> List[Dict]:
        """
        Check for interactions between medications.

        Args:
            medications: List of medication names

        Returns:
            List of interaction dictionaries
        """
        interactions = []

        for i, drug1 in enumerate(medications):
            for drug2 in medications[i + 1 :]:
                interaction = self._check_pair_interaction(drug1.lower(), drug2.lower())
                if interaction:
                    interactions.append(
                        {
                            "drug1": drug1,
                            "drug2": drug2,
                            "severity": interaction["severity"],
                            "description": interaction["description"],
                        }
                    )

        return interactions

    def _check_pair_interaction(self, drug1: str, drug2: str) -> Optional[Dict]:
        """Check interaction between two drugs."""
        # Check both directions
        if drug1 in self.interactions_db and drug2 in self.interactions_db[drug1]:
            return self.interactions_db[drug1][drug2]
        elif drug2 in self.interactions_db and drug1 in self.interactions_db[drug2]:
            return self.interactions_db[drug2][drug1]

        return None

    def check_contraindications(
        self, medication: str, patient_profile: Dict
    ) -> List[Dict]:
        """
        Check for contraindications based on patient profile.

        Args:
            medication: Medication name
            patient_profile: Dictionary with patient information
                           (conditions, age, pregnancy_status, etc.)

        Returns:
            List of contraindication warnings
        """
        warnings = []
        med_lower = medication.lower()

        if med_lower not in self.contraindications_db:
            return warnings

        contraindications = self.contraindications_db[med_lower]

        # Check medical conditions
        patient_conditions = set(patient_profile.get("conditions", []))
        contraindicated_conditions = set(contraindications.get("conditions", []))

        conflicting_conditions = patient_conditions.intersection(
            contraindicated_conditions
        )
        for condition in conflicting_conditions:
            warnings.append(
                {
                    "type": "condition_contraindication",
                    "medication": medication,
                    "condition": condition,
                    "severity": "major",
                    "description": f"{medication} is contraindicated with {condition}",
                }
            )

        # Check age restrictions
        age_restrictions = contraindications.get("age_restrictions", {})
        patient_age = patient_profile.get("age")

        if patient_age:
            min_age = age_restrictions.get("min_age")
            max_age = age_restrictions.get("max_age")

            if min_age and patient_age < min_age:
                warnings.append(
                    {
                        "type": "age_restriction",
                        "medication": medication,
                        "severity": "major",
                        "description": f"{medication} not recommended for age < {min_age}",
                    }
                )

            if max_age and patient_age > max_age:
                warnings.append(
                    {
                        "type": "age_restriction",
                        "medication": medication,
                        "severity": "moderate",
                        "description": f"{medication} requires caution for age > {max_age}",
                    }
                )

        # Check pregnancy status
        pregnancy_status = patient_profile.get("pregnancy_status")
        pregnancy_info = contraindications.get("pregnancy")

        if pregnancy_status and pregnancy_info:
            if pregnancy_info == "contraindicated":
                warnings.append(
                    {
                        "type": "pregnancy_contraindication",
                        "medication": medication,
                        "severity": "major",
                        "description": f"{medication} is contraindicated during pregnancy",
                    }
                )
            elif (
                pregnancy_info == "avoid_third_trimester"
                and pregnancy_status == "third_trimester"
            ):
                warnings.append(
                    {
                        "type": "pregnancy_warning",
                        "medication": medication,
                        "severity": "moderate",
                        "description": f"{medication} should be avoided in third trimester",
                    }
                )

        return warnings
