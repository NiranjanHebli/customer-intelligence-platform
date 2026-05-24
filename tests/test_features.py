import pandas as pd
import numpy as np
from src.data_pipeline.features import add_derived_features, build_preprocessor

def test_add_derived_features():
    # Construct dummy DataFrame
    data = {
        "pdays": [999, 10, 999, 3],
        "previous": [0, 1, 0, 2]
    }
    df = pd.DataFrame(data)
    
    df_feat = add_derived_features(df)
    
    assert "pdays_contacted" in df_feat.columns
    assert "has_previous_contact" in df_feat.columns
    
    # Assert values
    # pdays_contacted: 1 if pdays != 999 else 0
    np.testing.assert_array_equal(df_feat["pdays_contacted"].values, [0, 1, 0, 1])
    
    # has_previous_contact: 1 if previous > 0 else 0
    np.testing.assert_array_equal(df_feat["has_previous_contact"].values, [0, 1, 0, 1])


def test_build_preprocessor_unknown_category():
    # Construct train data
    train_data = {
        "job": ["admin.", "services", "admin."],
        "age": [30, 40, 50]
    }
    df_train = pd.DataFrame(train_data)
    
    categorical_cols = ["job"]
    numerical_cols = ["age"]
    
    preprocessor = build_preprocessor(categorical_cols, numerical_cols)
    
    # Fit on training data
    preprocessor.fit(df_train)
    
    # Test data has an unknown category: "astronaut"
    test_data = {
        "job": ["services", "astronaut"],
        "age": [35, 45]
    }
    df_test = pd.DataFrame(test_data)
    
    # Transform test data
    # Should work without raising errors because handle_unknown='ignore' is set
    transformed = preprocessor.transform(df_test)
    
    # Check shape
    # "job" has 2 unique values in train ("admin.", "services") -> 2 columns
    # "age" has 1 numerical column -> 1 column
    # Total shape should be 2 rows by 3 columns
    assert transformed.shape == (2, 3)
