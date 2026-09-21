import logging
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.session.profiler import get_session_pair, get_ny_after_anchor, get_trading_days
from src.lens1.geometry import LondonSessionProfiler
from src.lens1.labeler import HistoricalScenarioLabeler, S_UNDEFINED

logger = logging.getLogger(__name__)

def generate_golden_dataset(df: pd.DataFrame, settings) -> pd.DataFrame:
    """
    Generates the Golden Dataset for Lens 2 supervised training.
    Maps London Geometry -> Ground Truth Scenario Label.
    """
    logger.info("Generating Golden Dataset for Lens 2 Predictor...")
    trading_days = get_trading_days(df)
    
    profiler = LondonSessionProfiler(settings)
    labeler = HistoricalScenarioLabeler(settings)
    
    dataset = []
    
    for date in tqdm(trading_days, desc="Labeling Scenarios"):
        # Extract London Profile
        london, _, lp, _ = get_session_pair(df, date, settings.sessions)
        if lp is None or len(london) < 5:
            continue
            
        # Get London Geometry
        geometry = profiler.profile_geometry(london)
        if not geometry:
            continue
            
        # Extract NY after Lens 1 Anchor (12:15)
        ny_after_anchor = get_ny_after_anchor(df, date, settings.sessions.lens1_anchor)
        if len(ny_after_anchor) < 5:
            continue
            
        # Get Ground Truth Scenario Label
        scenario_id = labeler.label_day(df, lp, ny_after_anchor)
        
        # We also need the base feature values that Lens 1 would have seen
        # to ensure the predictor has access to core ATR tracking etc.
        features = {
            "date": date.date(),
            "scenario_id": scenario_id,
            "london_range": lp.range_points,
            "london_high": lp.high,
            "london_low": lp.low,
            "atr": ny_after_anchor["atr"].iloc[0] if "atr" in ny_after_anchor.columns else 1.0,
        }
        
        # Merge dictionaries
        features.update(geometry)
        dataset.append(features)
        
    result_df = pd.DataFrame(dataset)
    
    # Save Dataset
    save_path = Path(settings.project_root) / settings.lens2.dataset_save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(save_path, index=False)
    
    logger.info(f"Generated Golden Dataset with {len(result_df)} samples.")
    # Log class distribution
    if "scenario_id" in result_df.columns:
        dist = result_df["scenario_id"].value_counts().sort_index()
        logger.info(f"Class Distribution: \\n{dist}")
        
    return result_df
