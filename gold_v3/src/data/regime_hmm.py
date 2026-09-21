"""
Lens 0: Unsupervised Regime Detection (Hidden Markov Model)
Trains a 3-state HMM to classify market regimes based on Rolling Z-Score features.
Includes Hysteresis Lag to prevent state flickering.
"""

import os
import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

class RegimeHMM:
    def __init__(self, n_components=3, model_path="models/lens0/hmm_model.pkl"):
        self.n_components = n_components
        self.model_path = Path(model_path)
        self.model = None
        
        # State mapping based on structural sorting
        # We will dynamically map 0=Chop, 1=Bull, 2=Bear based on mean returns/EFI
        self.state_map = {} 

    def train(self, df: pd.DataFrame):
        """Fit the GaussianHMM on historical scaled data."""
        from hmmlearn import hmm
        
        # We require hmm_ret_z, hmm_atr_z, hmm_efi_z
        features = ["hmm_ret_z", "hmm_atr_z", "hmm_efi_z"]
        for f in features:
            if f not in df.columns:
                raise ValueError(f"Feature {f} not found in dataframe. Ensure Phase 9 indicators are calculated.")
                
        # Drop warm-up NaNs
        clean_df = df.dropna(subset=features)
        X = clean_df[features].values
        
        logger.info(f"Training Lens 0 HMM on {len(X)} samples with 3 components (Chop, Bull, Bear)...")
        
        self.model = hmm.GaussianHMM(
            n_components=self.n_components, 
            covariance_type="full", 
            n_iter=100, 
            random_state=42
        )
        self.model.fit(X)
        
        # Determine which state is which by analyzing the means of the features
        # X: [Returns, ATR, EFI]
        means = self.model.means_
        
        # State with highest EFI/Returns -> Bull (1)
        # State with lowest EFI/Returns -> Bear (2)
        # Remaining -> Chop (0)
        efi_means = means[:, 2] # hmm_efi_z
        
        sorted_indices = np.argsort(efi_means) # ascending
        bear_state = sorted_indices[0] # lowest EFI
        bull_state = sorted_indices[-1] # highest EFI
        
        remaining = set([0, 1, 2]) - {bear_state, bull_state}
        chop_state = list(remaining)[0]
        
        self.state_map = {
            chop_state: 0, # Chop
            bull_state: 1, # Bull
            bear_state: 2  # Bear
        }
        
        logger.info(f"HMM Training Complete.")
        logger.info(f"  Mapped State {chop_state} -> Chop (0)")
        logger.info(f"  Mapped State {bull_state} -> Bull (1)")
        logger.info(f"  Mapped State {bear_state} -> Bear (2)")
        
        # Save model and mapping
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "state_map": self.state_map}, str(self.model_path))
        logger.info(f"HMM Model saved to {self.model_path}")
        
    def load(self):
        """Load the pre-trained HMM."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"HMM Model not found at {self.model_path}. Run --mode train-hmm first.")
        
        data = joblib.load(str(self.model_path))
        self.model = data["model"]
        if "state_map" in data:
            self.state_map = data["state_map"]
        else:
            # Fallback legacy
            self.state_map = {0: 0, 1: 1, 2: 2}

    def predict_live(self, df: pd.DataFrame, lookback=30) -> int:
        """
        Predict the current regime using a hysteresis lag.
        Requires the state probability to be > 0.70 for 2 consecutive bars.
        Returns the mapped state (0=Chop, 1=Bull, 2=Bear).
        """
        if self.model is None:
            self.load()
            
        features = ["hmm_ret_z", "hmm_atr_z", "hmm_efi_z"]
        if len(df) < lookback:
            return 0 # Default to chop if insufficient history
            
        # Extract recent window
        recent_df = df.iloc[-lookback:].copy()
        # forward fill any single nans that might exist at live edge
        recent_df = recent_df.ffill().bfill()
        
        X = recent_df[features].values
        
        # Viterbi decoding gives the hard path, but we want probabilities for hysteresis
        # predict_proba returns array of shape (n_samples, n_components)
        try:
            probs = self.model.predict_proba(X)
        except Exception as e:
            logger.warning(f"Failed HMM predict_proba, defaulting to Chop (0). Error: {e}")
            return 0
            
        if len(probs) < 2:
            return 0
            
        # Get the highest probability class for the last 2 bars (t and t-1)
        prob_t = probs[-1]
        prob_t_minus_1 = probs[-2]
        
        state_t = np.argmax(prob_t)
        state_t_minus_1 = np.argmax(prob_t_minus_1)
        
        confidence_t = prob_t[state_t]
        confidence_t_minus_1 = prob_t_minus_1[state_t_minus_1]
        
        # Phase 9: Hysteresis Lag (Requires P > 0.70 for 2 consecutive matching bars)
        if (state_t == state_t_minus_1) and (confidence_t > 0.70) and (confidence_t_minus_1 > 0.70):
            # Confirmed regime shift!
            return self.state_map.get(state_t, 0)
            
        # If not confirmed, we fallback to Chop (safety first)
        return 0

    def predict_with_velocity(self, df_m15: pd.DataFrame, df_m5: pd.DataFrame = None, lookback=30) -> tuple[int, float, float]:
        """
        Predict regime state with institutional Velocity (Rate of Change).
        Returns: (state, velocity, current_confidence)
        Velocity = (P(t) - P(t-1))
        Since periods are fixed M15, Time_Delta is effectively 1 unit.
        """
        if self.model is None:
            self.load()
            
        features = ["hmm_ret_z", "hmm_atr_z", "hmm_efi_z"]
        if len(df_m15) < lookback:
            return 0, 0.0, 0.0
            
        recent_df = df_m15.iloc[-lookback:].copy().ffill().bfill()
        X = recent_df[features].values
        
        try:
            probs = self.model.predict_proba(X)
        except Exception as e:
            return 0, 0.0, 0.0
            
        if len(probs) < 2:
            return 0, 0.0, 0.0
            
        prob_t = probs[-1]
        prob_t_minus_1 = probs[-2]
        
        state_t = np.argmax(prob_t)
        
        confidence_t = prob_t[state_t]
        confidence_t_minus_1 = prob_t_minus_1[state_t] # Prob of the SAME state in the previous bar
        
        velocity = confidence_t - confidence_t_minus_1
        
        mapped_state = self.state_map.get(state_t, 0)
        
        # M5 Slope Secondary Confirmation
        if df_m5 is not None and len(df_m5) >= 3:
            # Check price slope over last 3 M5 bars
            m5_slope = (df_m5["close"].iloc[-1] - df_m5["close"].iloc[-3])
            
            # If we are jumping into a Bull Trend (1), slope must be positive
            if mapped_state == 1 and m5_slope < 0:
                velocity = 0.0 # Dilute velocity, it's a lazy/divergent transition
            # If Bear Trend (2), slope must be negative
            elif mapped_state == 2 and m5_slope > 0:
                velocity = 0.0
        
        return mapped_state, velocity, confidence_t

def run_hmm_training(settings, project_root):
    from src.pipeline import run_data_pipeline
    data = run_data_pipeline(settings, project_root)
    
    import pandas as pd
    logger.info("Merging data splits for comprehensive HMM Regime Clustering...")
    df_all = pd.concat([data["train"], data["validate"], data["test"]]).sort_index()
    
    hmm_engine = RegimeHMM(n_components=3, model_path=str(project_root / "models/lens0/hmm_model.pkl"))
    hmm_engine.train(df_all)
    return hmm_engine
