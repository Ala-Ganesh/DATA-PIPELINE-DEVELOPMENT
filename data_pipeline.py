import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# ============================
# Step 1: Extract (Load Raw Data)
# ============================
def extract_data(file_path):
    """Load dataset using pandas."""
    return pd.read_csv(file_path)

# ============================
# Step 2: Transform (Clean & Preprocess)
# ============================
def transform_data(df):
    """Clean and preprocess dataset using Scikit-learn pipeline."""

    # Identify columns
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()

    # Numeric pipeline: handle missing values + scale
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline: handle missing values + encode
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine both pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ]
    )

    # Apply transformations
    transformed_data = preprocessor.fit_transform(df)

    return transformed_data, preprocessor

# ============================
# Step 3: Load (Save Processed Data)
# ============================
def load_data(transformed_data, output_path):
    """Save preprocessed data to a CSV file."""
    processed_df = pd.DataFrame(transformed_data.toarray() if hasattr(transformed_data, 'toarray') else transformed_data)
    processed_df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

# ============================
# Main ETL Pipeline
# ============================
def main():
    input_file = "raw_data.csv"   # Replace with your dataset path
    output_file = "processed_data.csv"

    # Extract
    df = extract_data(input_file)
    print(f"Loaded data with shape: {df.shape}")

    # Transform
    transformed_data, preprocessor = transform_data(df)
    print(f"Transformed data shape: {transformed_data.shape}")

    # Load
    load_data(transformed_data, output_file)

if __name__ == "__main__":
    main()
