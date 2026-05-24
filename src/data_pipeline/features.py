import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates derived features from existing raw features.
    
    1. pdays_contacted: 1 if pdays != 999 (was previously contacted), 0 otherwise.
    2. has_previous_contact: 1 if previous > 0, 0 otherwise.
    """
    df_copy = df.copy()
    
    # pdays == 999 means client was not previously contacted
    if 'pdays' in df_copy.columns:
        df_copy['pdays_contacted'] = (df_copy['pdays'] != 999).astype(int)
    
    # previous is the number of contacts performed before this campaign
    if 'previous' in df_copy.columns:
        df_copy['has_previous_contact'] = (df_copy['previous'] > 0).astype(int)
        
    return df_copy


def build_preprocessor(categorical_cols: list, numerical_cols: list) -> ColumnTransformer:
    """
    Builds a scikit-learn ColumnTransformer preprocessor for encoding and scaling.
    Uses handle_unknown='ignore' for OneHotEncoder to ensure robustness at serving time.
    """
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    numerical_transformer = StandardScaler()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_cols),
            ('num', numerical_transformer, numerical_cols)
        ],
        remainder='passthrough'
    )
    
    return preprocessor
