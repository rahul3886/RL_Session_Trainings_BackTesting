import torch
import joblib
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
import os

def verify_pentagon_stack():
    print("--- Pentagon Stack Verification (Phase 10) ---")
    base_path = Path("C:/RAHUL BOGI/RL_Session Trainings/xauusd_v3")
    
    # Brain 0: Regime HMM
    hmm_path = base_path / "models/lens0/hmm_model.pkl"
    if hmm_path.exists():
        data = joblib.load(hmm_path)
        hmm = data["model"]
        print(f"[OK] Brain 0 (HMM): Loaded. States={hmm.n_components}")
    else:
        print("[FAIL] Brain 0 (HMM): Missing")

    # Brain 1: Pattern Encoder
    lens1_path = base_path / "models/lens1/lens1_ppo_final.zip"
    if lens1_path.exists():
        lens1 = PPO.load(str(lens1_path))
        print(f"[OK] Brain 1 (Pattern): Loaded. Obs_dim={lens1.observation_space.shape[0]}")
    else:
        print("[FAIL] Brain 1 (Pattern): Missing")

    # Brain 2: NY Classifier (Scaler)
    scaler_path = base_path / "models/lens2/lens2_scaler.pkl"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        print("[OK] Brain 2 (Scaler): Loaded.")
    else:
        print("[FAIL] Brain 2 (Scaler): Missing")

    # Brain 3: NY Classifier (Weights)
    lens2_path = base_path / "models/lens2/lens2_best.pth"
    if lens2_path.exists():
        print("[OK] Brain 3 (Classifier): Weight file exists.")
    else:
        print("[FAIL] Brain 3 (Classifier) weights: Missing")

    # Brain 4: Sniper Agent (Lens 3)
    lens3_path = base_path / "models/lens3/lens3_ppo_final.zip"
    if lens3_path.exists():
        # Check if the file is valid and readable
        try:
            lens3 = MaskablePPO.load(str(lens3_path))
            obs_dim = lens3.observation_space.shape[0]
            print(f"[OK] Brain 4 (Sniper): Loaded. Obs_dim={obs_dim}")
            if obs_dim != 64:
                print(f"[CRITICAL] Obs_dim mismatch: Found {obs_dim}, Expected 64")
            else:
                print("[SUCCESS] Phase 10 Model synchronized with configuration.")
        except Exception as e:
            print(f"[FAIL] Brain 4 (Sniper): Error loading: {e}")
    else:
        print("[FAIL] Brain 4 (Sniper): Missing")

if __name__ == "__main__":
    verify_pentagon_stack()
