# In save_model_files.py, add this before saving the scaler:
# Force the scaler to have the right feature names
if hasattr(scaler, 'feature_names_in_'):
    print(f"✅ Scaler feature names: {scaler.feature_names_in_}")
else:
    # For older scikit-learn versions, we'll manually set it
    print("⚠️  Scaler doesn't have feature_names_in_ attribute")
    
# Create a wrapper that ensures feature order
class FeatureOrderScaler:
    def __init__(self, scaler, feature_names):
        self.scaler = scaler
        self.feature_names = feature_names
        
    def transform(self, X):
        # Ensure X has the right feature order
        if hasattr(X, 'columns'):
            X = X[self.feature_names]
        return self.scaler.transform(X)
    
    def __getattr__(self, name):
        return getattr(self.scaler, name)

# Wrap the scaler
scaler = FeatureOrderScaler(scaler, all_feature_columns)