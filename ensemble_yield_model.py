from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

# Example of the Ensemble approach: Random Forest & XGBoost
# as highlighted in the VIT Machine Learning Project Report

def train_yield_models(X_train, y_train):
    # Random Forest: Robust for non-linear relationships [cite: 4042, 4043]
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # XGBoost: High accuracy on tabular environmental data [cite: 4044, 4096]
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
    xgb_model.fit(X_train, y_train)
    
    return rf_model, xgb_model
