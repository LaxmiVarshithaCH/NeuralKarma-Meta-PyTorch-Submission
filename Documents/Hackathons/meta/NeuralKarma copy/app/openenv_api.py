import uuid
import math
from typing import Dict, Any, Optional
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException

router = APIRouter()

# Data models
class ResetResponse(BaseModel):
    env_id: str
    observation: Dict[str, Any]

class StepRequest(BaseModel):
    action: Dict[str, Any]

class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]

class StateResponse(BaseModel):
    env_id: str
    state: Dict[str, Any]
    observation: Dict[str, Any]

# In-memory session store
sessions: Dict[str, Dict[str, Any]] = {}

TASKS = ["score_prediction", "axis_classification", "ethical_optimization"]

# Sample scenarios
SCENARIOS = {
    "score_prediction": [
        "I helped an old lady cross the street and carried her groceries.",
        "I stole a candy bar from the local convenience store."
    ],
    "axis_classification": [
        "I fairly divided the remaining budget equally among all team members.",
        "I spread a false rumor about my coworker so they would be fired."
    ],
    "ethical_optimization": [
        "I ignored a colleague who was struggling with a heavy workload."
    ]
}

def generate_observation(session: Dict[str, Any]) -> Dict[str, Any]:
    task = session["current_task"]
    step = session["step"]
    
    if task == "score_prediction":
        scenario = SCENARIOS["score_prediction"][min(step, len(SCENARIOS["score_prediction"])-1)]
    elif task == "axis_classification":
        scenario = SCENARIOS["axis_classification"][min(step, len(SCENARIOS["axis_classification"])-1)]
    else:
        scenario = SCENARIOS["ethical_optimization"][0]

    session["current_scenario"] = scenario
    
    return {
        "scenario": scenario,
        "task_type": task,
        "progress": step / 5.0,
        "metadata": {"step": step}
    }

@router.post("/reset", response_model=ResetResponse)
async def reset_environment(task_type: Optional[str] = "score_prediction"):
    env_id = str(uuid.uuid4())
    if task_type not in TASKS:
        task_type = TASKS[0]
        
    session = {
        "env_id": env_id,
        "current_task": task_type,
        "step": 0,
        "history": []
    }
    
    sessions[env_id] = session
    obs = generate_observation(session)
    return ResetResponse(env_id=env_id, observation=obs)

@router.get("/state/{env_id}", response_model=StateResponse)
async def get_state(env_id: str):
    if env_id not in sessions:
        raise HTTPException(status_code=404, detail="Environment not found")
    
    session = sessions[env_id]
    obs = generate_observation(session)
    return StateResponse(env_id=env_id, state={"step": session["step"], "task": session["current_task"]}, observation=obs)

@router.post("/close/{env_id}")
async def close_environment(env_id: str):
    if env_id in sessions:
        del sessions[env_id]
    return {"status": "closed"}

@router.post("/step/{env_id}", response_model=StepResponse)
async def step_environment(env_id: str, request: StepRequest):
    if env_id not in sessions:
        raise HTTPException(status_code=404, detail="Environment not found")
        
    from ml.karma_engine import get_scorer
    scorer = get_scorer()
    
    session = sessions[env_id]
    task = session["current_task"]
    scenario = session.get("current_scenario", "")
    action = request.action
    
    reward = 0.0
    error = None
    
    try:
        if task == "score_prediction":
            # Compare predicted score with actual model score
            predicted = float(action.get("predicted_score", 0))
            result = scorer.score_text(scenario)
            actual_score = result.get("karma_score", 50)
            
            diff = abs(predicted - actual_score)
            reward = max(0.0, 1.0 - (diff / 20.0)) # 1.0 reward if exactly right, drops to 0 if diff > 20
            
        elif task == "axis_classification":
            predicted_axis = str(action.get("primary_axis", "")).lower()
            result = scorer.score_text(scenario)
            axes = result.get("axis_scores", {})
            # Fix tied values safely
            best_axis = max(axes.items(), key=lambda x: x[1])[0]
            
            if predicted_axis == best_axis:
                reward = 1.0
            else:
                reward = 0.0
                
        elif task == "ethical_optimization":
            rewritten = str(action.get("rewritten_action", ""))
            result = scorer.score_text(rewritten)
            score = result.get("karma_score", 50)
            
            # Map score 50-100 to reward 0-1
            if score >= 80:
                reward = 1.0
            elif score > 50:
                reward = (score - 50) / 30.0
            else:
                reward = 0.0
                
    except Exception as e:
        error = str(e)
        reward = 0.0
        
    session["step"] += 1
    session["history"].append({"action": action, "reward": reward})
    
    # 5 steps per task
    done = session["step"] >= 5 or task == "ethical_optimization"
    
    obs = generate_observation(session)
    
    return StepResponse(
        observation=obs,
        reward=reward,
        done=done,
        info={"error": error}
    )
