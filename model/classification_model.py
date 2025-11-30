import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# --- Configuration ---
DATA_FILE = '../data/startup_data_processed.csv'  # Assumed file name for the cleaned dataset
MODEL_OUTPUT_FILE = '../model/random_forest_model.joblib'
TARGET_COLUMN = 'labels'

def load_data(file_path):
    """Loads the cleaned data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please ensure the cleaned data is available.")
        # Exit or return None to prevent further execution
        return None

def train_and_save_model(df):
    """
    Splits data, defines the Random Forest pipeline, trains the model, 
    evaluates it, and saves the trained pipeline using joblib.
    """
    
    # Define Features (X) and Target (y)
    # The columns to be dropped are derived directly from your original snippet.
    columns_to_drop = [
        TARGET_COLUMN, 'founded_at', 'name', 'has_roundA', 'has_roundB', 
        'has_roundC', 'has_VC', 'has_angel',
    ]
    
    # Ensure only existing columns are dropped
    actual_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    
    X = df.drop(columns=actual_columns_to_drop)
    y = df[TARGET_COLUMN]

    print(f"Features selected: {list(X.columns)}")
    print(f"Target distribution:\n{y.value_counts()}")

    # Identify categorical and numeric columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns

    print(f"Categorical columns identified: {list(cat_cols)}")
    print(f"Numerical columns identified: {list(num_cols)}")

    # Define Preprocessing Pipeline
    # Only using the 'no_scaling' approach as Random Forest is not sensitive to feature scaling.
    preprocess_no_scaling = ColumnTransformer(
        transformers=[
            # One-Hot Encode categorical features, ignoring unknown categories during inference
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            # Pass-through numerical features without scaling
            ("num", "passthrough", num_cols),
        ],
        remainder='drop' # Drop any columns not explicitly handled
    )

    # Define the Final Random Forest Pipeline
    rf_pipeline = Pipeline([
        ("prep", preprocess_no_scaling),
        # Using the specified hyperparameters for the classifier
        ("clf", RandomForestClassifier(max_depth=None, min_samples_split=2, random_state=22))
    ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("\nData split into 80% training and 20% testing.")

    # Train the model
    print("Starting Random Forest model training...")
    rf_pipeline.fit(X_train, y_train)
    print("Training complete.")

    # Evaluate the model
    y_pred = rf_pipeline.predict(X_test)
    print("\n=========== Random Forest Classification Report ===========")
    print(classification_report(y_test, y_pred))
    print("========================================================")

    # Save the trained pipeline
    joblib.dump(rf_pipeline, MODEL_OUTPUT_FILE)
    print(f"\nSuccessfully saved the trained Random Forest pipeline to: {MODEL_OUTPUT_FILE}")


if __name__ == "__main__":
    data = load_data(DATA_FILE)
    if data is not None and TARGET_COLUMN in data.columns:
        train_and_save_model(data)
    elif data is not None:
        print(f"Error: Target column '{TARGET_COLUMN}' not found in the loaded data.")