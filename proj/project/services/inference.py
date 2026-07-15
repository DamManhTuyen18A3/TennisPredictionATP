"""
Enhanced Inference Service

Handles model loading, prediction, and SHAP-based explainability.
This is the core prediction engine for the application.

Features:
- Model caching and versioning
- Batch and single predictions
- SHAP value computation
- Confidence scoring
- Error handling and logging
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import streamlit as st

from project.utils.logger import get_logger
from project.utils.formatters import ProbabilityFormatter, NumberFormatter

logger = get_logger(__name__)


class ModelLoader:
    """Load and manage ML models."""
    
    _instance = None
    _models_cache: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, models_dir: str = None):
        # LỖI ĐÃ SỬA: trước đây mặc định `models_dir="models"` — đây là
        # đường dẫn TƯƠNG ĐỐI theo thư mục làm việc hiện tại (cwd) lúc chạy
        # lệnh, KHÔNG phải theo vị trí file này. Nếu người dùng chạy
        # `streamlit run project/main.py` từ một thư mục khác thư mục gốc dự
        # án (vd. từ trong `project/` hoặc từ ổ đĩa khác trên Windows), Python
        # sẽ tìm nhầm thư mục `models/` và báo "không tìm thấy model". Nay
        # tính đường dẫn tuyệt đối dựa trên vị trí file này (giống cách
        # data_fetcher.py và Home.py/Analytics.py đã làm), nên luôn đúng bất
        # kể lệnh được chạy từ thư mục nào.
        if models_dir is None:
            project_root = Path(__file__).resolve().parents[2]
            models_dir = project_root / "models"
        self.models_dir = Path(models_dir)
        self.available_models = [
            "final/final_model.joblib",
            "tuned/CatBoost_tuned.joblib",
            "LightGBM_baseline.joblib",
            "XGBoost_baseline.joblib",
            "RandomForest_baseline.joblib",
        ]
    
    @st.cache_resource
    def _load_model_cached(_self, model_name: str) -> Optional[Any]:
        """Hàm cache thực sự — tham số đầu có dấu `_` để Streamlit không cố
        băm (hash) đối tượng `self`, đây chính là nguyên nhân khiến trang
        Prediction crash ngay khi mở với các bản Streamlit mới hơn."""
        model_path = _self.models_dir / model_name

        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return None

        try:
            model = joblib.load(model_path)
            logger.info(f"Model loaded successfully: {model_name}")
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            return None

    def load_model(self, model_name: str = "tuned/CatBoost_tuned.joblib") -> Optional[Any]:
        """Load a model with caching."""
        return self._load_model_cached(model_name)
    
    def get_best_model(self) -> Tuple[Optional[Any], str]:
        """Get the best performing model."""
        for model_path in ["final/final_model.joblib", "tuned/CatBoost_tuned.joblib"]:
            model = self.load_model(model_path)
            if model is not None:
                return model, model_path
        return None, "No model found"


class PredictionEngine:
    """
    Core prediction engine with explainability.
    """
    
    def __init__(self):
        self.model_loader = ModelLoader()
        self.model, self.model_name = self.model_loader.get_best_model()
        
        if self.model is None:
            logger.error("Failed to load any model")
            # Create a fallback model that returns 0.5
            self.model = None
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on input features.
        
        Args:
            features: Input features DataFrame
        
        Returns:
            Prediction probabilities
        """
        try:
            if self.model is None:
                return np.full(len(features), 0.5)
            
            probabilities = self.model.predict_proba(features)
            # Return probability of class 1 (player A wins)
            return probabilities[:, 1]
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return np.full(len(features), 0.5)
    
    def predict_single(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a single prediction with confidence score.
        
        Args:
            features: Dictionary of feature values
        
        Returns:
            Prediction result dictionary
            
        Example:
            {
                'probability': 0.75,
                'confidence': 'HIGH',
                'confidence_emoji': '🟢',
                'winner': 'Player A',
                'odds_implied': 1.33,
            }
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame([features])
            
            # Get prediction
            probability = self.predict(df)[0]
            
            # Calculate confidence
            confidence = ProbabilityFormatter.confidence_level(probability)
            confidence_emoji = ProbabilityFormatter.confidence_emoji(probability)
            
            # Determine winner
            winner = "Player A" if probability > 0.5 else "Player B"
            
            # Implied odds
            odds_implied = 1 / probability if probability > 0 else 0
            
            result = {
                'probability': float(probability),
                'probability_formatted': NumberFormatter.percentage(probability),
                'probability_opponent': float(1 - probability),
                'confidence': confidence,
                'confidence_emoji': confidence_emoji,
                'winner': winner,
                'odds_implied': float(odds_implied),
                'expected_value': float(probability - 0.5),
            }
            
            logger.info(f"Prediction made: {winner} with {probability:.1%}")
            return result
        
        except Exception as e:
            logger.error(f"Single prediction error: {str(e)}")
            return {'error': str(e)}
    
    def predict_batch(self, features_list: List[Dict]) -> List[Dict]:
        """
        Make batch predictions.
        
        Args:
            features_list: List of feature dictionaries
        
        Returns:
            List of prediction results
        """
        results = []
        for features in features_list:
            result = self.predict_single(features)
            results.append(result)
        return results
    
    def get_shap_explanation(self, features: pd.DataFrame, top_n: int = 8) -> Dict[str, float]:
        """SHAP THẬT cho một dự đoán cụ thể.

        Dùng `model.get_feature_importance(..., type='ShapValues')` — API SHAP
        NỘI BỘ của chính CatBoost — thay vì thư viện `shap` bên ngoài
        (`shap.TreeExplainer`). Lý do: trong quá trình kiểm thử, tổ hợp
        `shap.TreeExplainer` + một số giá trị đầu vào cực trị (ví dụ hồ sơ có
        Elo/rank_points rất cao) từng gây crash tiến trình (segmentation
        fault) khi chạy lồng trong nhiều tab/form của Streamlit. API nội bộ
        của CatBoost cho kết quả SHAP giống hệt về mặt toán học nhưng ổn định
        hơn vì không phụ thuộc thư viện ngoài."""
        try:
            if self.model is None:
                return {}
            from catboost import Pool

            cat_idx = self.model.get_cat_feature_indices()
            pool = Pool(features, cat_features=cat_idx)
            shap_matrix = self.model.get_feature_importance(pool, type="ShapValues")
            # Cột cuối cùng là giá trị base (expected value), không phải 1 feature
            values = shap_matrix[0][:-1]
            pairs = sorted(
                zip(features.columns.tolist(), values),
                key=lambda x: -abs(float(x[1])),
            )[:top_n]
            return {name: float(val) for name, val in pairs}
        except Exception as e:
            logger.error(f"SHAP explanation error: {str(e)}")
            return {}

    def get_multi_model_consensus(self, base_row: pd.DataFrame) -> Dict[str, Optional[float]]:
        """
        So sánh dự đoán của CẢ 4 mô hình đã huấn luyện cho cùng 1 trận đấu
        (TÍNH NĂNG MỚI — tăng tính khoa học: thay vì chỉ tin 1 mô hình,
        người dùng thấy được mức độ đồng thuận giữa các thuật toán khác nhau).

        Mỗi thư viện (CatBoost/LightGBM/XGBoost/RandomForest) yêu cầu định
        dạng cột phân loại khác nhau nên phải chuẩn bị riêng cho từng model.
        Model nào lỗi (vd. XGBoost đôi khi kén category chưa từng thấy khi
        train) sẽ trả về None và được hiển thị là "Không khả dụng" thay vì
        làm sập cả trang.
        """
        results: Dict[str, Optional[float]] = {}
        cat_cols = [
            "surface", "tourney_level", "p1_entry", "p2_entry",
            "p1_hand", "p2_hand", "p1_ioc", "p2_ioc", "round", "match_category",
        ]

        # 1) CatBoost (tuned) — model chính, đã load sẵn ở self.model
        try:
            results["CatBoost (đã tối ưu)"] = float(self.model.predict_proba(base_row)[:, 1][0])
        except Exception as e:
            logger.error(f"Consensus CatBoost error: {e}")
            results["CatBoost (đã tối ưu)"] = None

        # 2) CatBoost baseline (chưa tối ưu) — cùng định dạng string
        try:
            cbb = self.model_loader.load_model("CatBoost_baseline.joblib")
            results["CatBoost (baseline)"] = float(cbb.predict_proba(base_row)[:, 1][0]) if cbb else None
        except Exception as e:
            logger.error(f"Consensus CatBoost baseline error: {e}")
            results["CatBoost (baseline)"] = None

        # 3) LightGBM — cần dtype 'category' thay vì 'string'
        try:
            lgb_model = self.model_loader.load_model("LightGBM_baseline.joblib")
            if lgb_model is not None:
                row_cat = base_row.copy()
                for c in cat_cols:
                    if c in row_cat.columns:
                        row_cat[c] = row_cat[c].astype("category")
                results["LightGBM"] = float(lgb_model.predict_proba(row_cat)[:, 1][0])
            else:
                results["LightGBM"] = None
        except Exception as e:
            logger.error(f"Consensus LightGBM error: {e}")
            results["LightGBM"] = None

        # 4) XGBoost — ĐÃ LOẠI BỎ khỏi bảng đồng thuận.
        # Lý do: trong quá trình kiểm thử, khi cột phân loại (vd. p1_entry) có
        # giá trị NaN được gán chuỗi "nan" (không nằm trong tập category lúc
        # train XGBoost), một số trường hợp gây segmentation fault ở tầng
        # native C++ của XGBoost KHÔNG THỂ bắt bằng try/except Python — nhất
        # là khi gọi lồng sâu trong nhiều tab/form của Streamlit. Để đảm bảo
        # ứng dụng không bao giờ crash, XGBoost được loại khỏi bảng đồng
        # thuận; CatBoost (2 phiên bản) + LightGBM + Random Forest vẫn cho
        # đủ góc nhìn đa mô hình một cách an toàn.
        results["XGBoost"] = None

        # 5) Random Forest — chỉ dùng tập con 38 feature số (không có cột phân loại)
        try:
            rf_model = self.model_loader.load_model("RandomForest_baseline.joblib")
            if rf_model is not None:
                rf_cols = list(rf_model.feature_names_in_)
                results["Random Forest"] = float(rf_model.predict_proba(base_row[rf_cols])[:, 1][0])
            else:
                results["Random Forest"] = None
        except Exception as e:
            logger.error(f"Consensus RandomForest error: {e}")
            results["Random Forest"] = None

        return results

    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """
        Get feature importance from model.
        
        Args:
            top_n: Number of top features
        
        Returns:
            Dictionary of feature importances
        """
        try:
            if self.model is None or not hasattr(self.model, 'feature_importances_'):
                return {}
                
            importances = self.model.feature_importances_
            feature_names = getattr(self.model, 'feature_names_in_', 
                                   [f"Feature_{i}" for i in range(len(importances))])
            
            # Create DataFrame and sort
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(top_n)
            
            return dict(zip(importance_df['feature'], importance_df['importance']))
        
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {}
    
    def explain_prediction(self, features: Dict[str, Any], 
                          player_names: Tuple[str, str] = ("Player A", "Player B"),
                          num_features: int = 10) -> Dict[str, Any]:
        """
        Generate explanation for a prediction.
        
        Args:
            features: Input features
            player_names: Tuple of (player_a, player_b) names
            num_features: Number of top features to explain
        
        Returns:
            Explanation dictionary
        """
        try:
            df = pd.DataFrame([features])
            prediction = self.predict(df)[0]
            
            explanation = {
                'prediction': float(prediction),
                'player_a': player_names[0],
                'player_b': player_names[1],
                'top_features': self.get_feature_importance(num_features),
            }
            
            # Generate text explanation
            if prediction > 0.7:
                confidence_text = "Very Strong"
            elif prediction > 0.6:
                confidence_text = "Strong"
            elif prediction > 0.55:
                confidence_text = "Moderate"
            elif prediction > 0.5:
                confidence_text = "Slight"
            else:
                confidence_text = "Slight Advantage"
                
            winner = player_names[0] if prediction > 0.5 else player_names[1]
            explanation['summary'] = (
                f"Model predicts {winner} to win with {confidence_text} confidence "
                f"({prediction:.1%} probability). "
                f"Key factors: {', '.join(list(explanation['top_features'].keys())[:3])}"
            )
            
            logger.info(f"Prediction explained: {explanation['summary']}")
            return explanation
        
        except Exception as e:
            logger.error(f"Explanation error: {str(e)}")
            return {
                'error': str(e),
                'prediction': None,
            }
    
    def monte_carlo_simulation(self, features: Dict[str, Any], 
                              num_simulations: int = 10000) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation for outcome probability.
        
        Args:
            features: Input features
            num_simulations: Number of simulations
        
        Returns:
            Simulation results
        """
        try:
            df = pd.DataFrame([features])
            base_probability = self.predict(df)[0]
            
            # Simulate outcomes with slight noise
            np.random.seed(42)
            noise = np.random.normal(0, 0.05, num_simulations)
            simulated_probs = np.clip(base_probability + noise, 0, 1)
            
            wins = np.sum(simulated_probs > 0.5)
            win_percentage = wins / num_simulations
            
            result = {
                'simulations': num_simulations,
                'mean_probability': float(np.mean(simulated_probs)),
                'std_probability': float(np.std(simulated_probs)),
                'win_percentage': float(win_percentage),
                'simulated_probabilities': simulated_probs.tolist(),
                'confidence_interval': {
                    'lower': float(np.percentile(simulated_probs, 2.5)),
                    'upper': float(np.percentile(simulated_probs, 97.5)),
                }
            }
            
            logger.info(f"Monte Carlo simulation complete: {win_percentage:.1%} win rate")
            return result
        
        except Exception as e:
            logger.error(f"Simulation error: {str(e)}")
            return {'error': str(e)}


@st.cache_resource
def get_prediction_engine() -> PredictionEngine:
    """Get prediction engine instance (cached)."""
    return PredictionEngine()


# Backwards compatibility functions
@st.cache_resource
def load_model():
    """Load model (legacy function for compatibility)."""
    engine = get_prediction_engine()
    return engine.model


def predict_match(model, X_input):
    """
    Predict match outcome (legacy function for compatibility).
    
    Args:
        model: Model object
        X_input: Input features DataFrame
        
    Returns:
        Probability of Player A winning
    """
    if model is None:
        return 0.5
    
    try:
        proba = model.predict_proba(X_input)[0][1]
        return proba
    except Exception as e:
        logger.error(f"Legacy predict error: {str(e)}")
        return 0.5


__all__ = [
    "ModelLoader",
    "PredictionEngine",
    "get_prediction_engine",
    "load_model",
    "predict_match",
]



