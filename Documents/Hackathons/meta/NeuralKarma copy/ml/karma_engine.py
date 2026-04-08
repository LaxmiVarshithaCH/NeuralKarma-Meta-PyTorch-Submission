"""
NeuralKarma — Karma Scoring Engine
Core inference engine that computes multi-dimensional karma scores:
  - 5-axis scoring (prosociality, fairness/justice, harm avoidance, virtue, deontology)
  - Temporal decay using exponential half-life model
  - Ripple effect propagation through social graph
  - Aggregate karma with weighted combination
"""

import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np

MODEL_DIR = Path(__file__).parent / "models"

# Axis weights for aggregate score (sum to 1.0)
# Prosociality and harm avoidance weighted higher for social impact
AXIS_WEIGHTS = {
    "prosociality": 0.25,
    "harm_avoidance": 0.20,  # derived from commonsense
    "fairness": 0.20,        # derived from justice
    "virtue": 0.20,          # derived from virtue
    "duty": 0.15,            # derived from deontology
}

# Map model names to display axis names
MODEL_TO_AXIS = {
    "prosociality": "prosociality",
    "commonsense": "harm_avoidance",
    "justice": "fairness",
    "virtue": "virtue",
    "deontology": "duty",
}

# Half-life for temporal decay (in hours)
# After 168 hours (1 week), a score decays to 50%
KARMA_HALF_LIFE_HOURS = 168.0

# Decay constant λ = ln(2) / half_life
DECAY_LAMBDA = math.log(2) / KARMA_HALF_LIFE_HOURS

# Ripple effect damping factor (0-1)
# Each hop in the social graph reduces the ripple by this factor
RIPPLE_DAMPING = 0.6

# Maximum ripple depth
MAX_RIPPLE_DEPTH = 3


class KarmaScorer:
    """
    Multi-axis karma scoring engine. Loads trained ML models and computes
    real ethical impact scores for input text.
    """

    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self._loaded = False

    def load_models(self):
        """Load all trained models from disk."""
        if self._loaded:
            return

        print("Loading NeuralKarma models...")
        model_files = [
            ("prosociality", "prosociality_model.joblib", "prosociality_vectorizer.joblib"),
            ("commonsense", "ethics_commonsense_model.joblib", "ethics_commonsense_vectorizer.joblib"),
            ("deontology", "ethics_deontology_model.joblib", "ethics_deontology_vectorizer.joblib"),
            ("justice", "ethics_justice_model.joblib", "ethics_justice_vectorizer.joblib"),
            ("virtue", "ethics_virtue_model.joblib", "ethics_virtue_vectorizer.joblib"),
            ("utilitarianism", "ethics_utilitarianism_model.joblib", "ethics_utilitarianism_vectorizer.joblib"),
        ]

        for name, model_file, vec_file in model_files:
            model_path = MODEL_DIR / model_file
            vec_path = MODEL_DIR / vec_file
            if model_path.exists() and vec_path.exists():
                self.models[name] = joblib.load(model_path)
                self.vectorizers[name] = joblib.load(vec_path)
                print(f"  [OK] Loaded {name} model")
            else:
                print(f"  [WARNING] Model not found: {name} (will skip this axis)")

        self._loaded = True
        print(f"  Loaded {len(self.models)} models total")

    def score_text(self, text: str) -> Dict[str, float]:
        """
        Score a text input across all ethical axes.

        Returns dict with:
          - Individual axis scores (0-100)
          - Aggregate karma score (0-100)
          - Raw probabilities
          - Confidence level
        """
        if not self._loaded:
            self.load_models()

        axis_scores = {}
        raw_probs = {}
        confidences = []

        for model_name, model in self.models.items():
            if model is None:
                continue

            vectorizer = self.vectorizers.get(model_name)
            if vectorizer is None:
                continue

            try:
                # Transform input text
                X = vectorizer.transform([text])

                # Get probability of positive class (ethical/prosocial)
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X)[0]
                    # prob of positive class (index 1)
                    positive_prob = probs[1] if len(probs) > 1 else probs[0]
                else:
                    # Fallback to decision function
                    decision = model.decision_function(X)[0]
                    positive_prob = 1 / (1 + math.exp(-decision))  # sigmoid

                # Map model name to axis name
                axis_name = MODEL_TO_AXIS.get(model_name, model_name)

                # Scale to 0-100 with non-linear mapping for more spread
                # Apply sigmoid-like stretching to avoid clustering around 50
                stretched = self._stretch_score(positive_prob)
                score = round(stretched * 100, 2)
                score = max(0, min(100, score))

                axis_scores[axis_name] = score
                raw_probs[axis_name] = round(positive_prob, 4)

                # Confidence = distance from 0.5 (higher = more confident)
                confidence = abs(positive_prob - 0.5) * 2
                confidences.append(confidence)

            except Exception as e:
                print(f"  Warning: Error scoring with {model_name}: {e}")
                axis_name = MODEL_TO_AXIS.get(model_name, model_name)
                axis_scores[axis_name] = 50.0  # neutral fallback
                raw_probs[axis_name] = 0.5

        # Calculate aggregate karma score
        aggregate = self.compute_aggregate(axis_scores)

        # Overall confidence
        avg_confidence = np.mean(confidences) if confidences else 0.0

        return {
            "axis_scores": axis_scores,
            "aggregate_karma": aggregate,
            "raw_probabilities": raw_probs,
            "confidence": round(avg_confidence, 4),
            "axes_evaluated": len(axis_scores),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _stretch_score(self, prob: float) -> float:
        """
        Apply non-linear stretching to probability to get better score distribution.
        Uses a modified sigmoid that pushes values away from 0.5.
        """
        # Center around 0
        centered = (prob - 0.5) * 2  # [-1, 1]

        # Apply cube root for spreading while preserving sign
        if centered >= 0:
            stretched = centered ** 0.7
        else:
            stretched = -((-centered) ** 0.7)

        # Map back to [0, 1]
        return (stretched + 1) / 2

    def compute_aggregate(self, axis_scores: Dict[str, float]) -> float:
        """
        Compute weighted aggregate karma score from individual axes.
        Uses the AXIS_WEIGHTS configuration.
        """
        total_weight = 0.0
        weighted_sum = 0.0

        for axis, weight in AXIS_WEIGHTS.items():
            if axis in axis_scores:
                weighted_sum += axis_scores[axis] * weight
                total_weight += weight

        if total_weight == 0:
            return 50.0  # neutral

        return round(weighted_sum / total_weight, 2)

    def apply_temporal_decay(self, score: float, action_time: datetime) -> float:
        """
        Apply exponential temporal decay to a karma score.

        Formula: decayed_score = score × e^(-λ × t)
        Where:
          λ = ln(2) / half_life (decay constant)
          t = elapsed hours since action

        After 1 week (168h), the score decays to 50%.
        After 2 weeks, 25%. After 1 month, ~6%.
        """
        now = datetime.now(timezone.utc)
        if action_time.tzinfo is None:
            action_time = action_time.replace(tzinfo=timezone.utc)

        elapsed_hours = (now - action_time).total_seconds() / 3600.0
        elapsed_hours = max(0, elapsed_hours)

        decay_factor = math.exp(-DECAY_LAMBDA * elapsed_hours)
        return round(score * decay_factor, 2)

    def compute_ripple_effect(
        self,
        source_score: float,
        depth: int = 1,
        connections: int = 3,
    ) -> Dict:
        """
        Simulate how an ethical action's impact propagates through a social network.

        At each hop:
        - Impact is multiplied by RIPPLE_DAMPING
        - Number of affected people grows by `connections` factor
        - Total ripple is the sum of all hop impacts

        Returns: {
            total_impact: float,
            hops: [{depth, impact_per_person, people_affected, cumulative}]
        }
        """
        hops = []
        total_impact = 0.0
        cumulative_people = 0

        for d in range(1, min(depth + 1, MAX_RIPPLE_DEPTH + 1)):
            # Impact decreases with each hop
            impact_per_person = source_score * (RIPPLE_DAMPING ** d)
            # People affected grows exponentially
            people_at_depth = connections ** d
            # Total impact at this depth
            depth_impact = impact_per_person * people_at_depth
            cumulative_people += people_at_depth
            total_impact += depth_impact

            hops.append({
                "depth": d,
                "impact_per_person": round(impact_per_person, 2),
                "people_affected": people_at_depth,
                "depth_total_impact": round(depth_impact, 2),
                "cumulative_people": cumulative_people,
                "cumulative_impact": round(total_impact, 2),
            })

        return {
            "source_score": source_score,
            "total_ripple_impact": round(total_impact, 2),
            "total_people_reached": cumulative_people,
            "damping_factor": RIPPLE_DAMPING,
            "hops": hops,
        }

    def compute_karma_chain(self, actions: List[Dict]) -> List[Dict]:
        """
        Analyze a chain of related actions and compute cascading karma effects.
        Each action in the chain can amplify or attenuate subsequent scores.

        Input: [{text, timestamp, parent_action_id}, ...]
        Output: [{text, score, chain_modifier, effective_score}, ...]
        """
        chain_results = []
        chain_modifier = 1.0

        for i, action in enumerate(actions):
            # Score this action
            result = self.score_text(action["text"])
            base_score = result["aggregate_karma"]

            # Apply chain modifier from previous actions
            effective_score = base_score * chain_modifier
            effective_score = max(0, min(100, effective_score))

            # Update chain modifier based on this action
            # Positive actions (>60) boost subsequent actions slightly
            # Negative actions (<40) dampen subsequent actions
            if base_score > 60:
                chain_modifier = min(1.5, chain_modifier * 1.05)
            elif base_score < 40:
                chain_modifier = max(0.5, chain_modifier * 0.95)
            else:
                chain_modifier = chain_modifier * 0.99  # slight decay to neutral

            chain_results.append({
                "index": i,
                "text": action["text"],
                "base_score": base_score,
                "axis_scores": result["axis_scores"],
                "chain_modifier": round(chain_modifier, 4),
                "effective_score": round(effective_score, 2),
                "timestamp": action.get("timestamp", datetime.now(timezone.utc).isoformat()),
            })

        return chain_results

    def get_karma_tier(self, score: float) -> Dict[str, str]:
        """
        Map aggregate karma score to a tier with label and description.
        """
        if score >= 90:
            return {"tier": "S", "label": "Enlightened", "color": "#FFD700",
                    "description": "Exceptional ethical impact — a true force for good"}
        elif score >= 75:
            return {"tier": "A", "label": "Virtuous", "color": "#00E5FF",
                    "description": "Strong positive ethical presence"}
        elif score >= 60:
            return {"tier": "B", "label": "Benevolent", "color": "#69F0AE",
                    "description": "Consistently positive ethical behavior"}
        elif score >= 45:
            return {"tier": "C", "label": "Neutral", "color": "#B0BEC5",
                    "description": "Balanced — no strong ethical lean"}
        elif score >= 30:
            return {"tier": "D", "label": "Questionable", "color": "#FFB74D",
                    "description": "Actions raise ethical concerns"}
        elif score >= 15:
            return {"tier": "E", "label": "Harmful", "color": "#FF5252",
                    "description": "Significant negative ethical impact"}
        else:
            return {"tier": "F", "label": "Destructive", "color": "#D50000",
                    "description": "Severely harmful — urgent ethical concern"}


# Singleton instance
_scorer = None

def get_scorer() -> KarmaScorer:
    """Get or create the global KarmaScorer instance."""
    global _scorer
    if _scorer is None:
        _scorer = KarmaScorer()
        _scorer.load_models()
    return _scorer
