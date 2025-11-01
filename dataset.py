import numpy as np
import pandas as pd
import torch 
from torch.utils.data import Dataset, DataLoader

from dataclasses import dataclass
from typing import List
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


@dataclass
class FeatureDescription:
    # Time tag
    time: str
    # Identifier tag, separating different time series
    id: str
    # Tag of y 
    target: str
    # Tags of static features, either categorical or continuous
    static_categorical: List[str]
    static_continuous: List[str]
    # Tags of known features
    known_categorical: List[str]
    known_continuous: List[str]
    # Tags of observed (unknown) features
    observed_continuous: List[str]
    observed_categorical: List[str]


class TimeSeriesDataset(Dataset):
    def __init__(
        self, 
        df, 
        feature_description, 
        encoder_length, 
        decoder_length,
        categorical_encoder=None
    ):
        super().__init__()
        self.df = df
        self.features = feature_description
        self.enc_len = encoder_length
        self.dec_len = decoder_length
        self.categorical_encoder = categorical_encoder
        self.time_steps = self.enc_len + self.dec_len

        # Preprocess the dataframe
        df = df.copy()
        df[self.features.time] = pd.to_datetime(df[self.features.time], utc=False)
        df.sort_values([self.features.id, self.features.time], inplace=True)

        # Group categorical features
        self.categorical_features = (
            self.features.static_categorical + 
            self.features.known_categorical + 
            self.features.observed_categorical
        )
        # Group continuous features
        self.continuous_features = (
            self.features.static_continuous + 
            self.features.known_continuous + 
            self.features.observed_continuous
        )

        # == Encode categorical features ==
        if self.categorical_features:
            # Ensure they are strings
            X_cat = df[self.categorical_features].astype("string")
            
            # Define categorical encoder and fit if not provided
            if self.categorical_encoder is None:
                self.categorical_encoder = OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1
                )
                self.categorical_encoder.fit(X_cat)
  
            # Shift to start from 0
            X_cat_encoded = self.categorical_encoder.transform(X_cat).astype(np.int64) + 1
            df[self.categorical_features] = X_cat_encoded
        else:
            self.categorical_encoder = None

        self.df = df    

        # Build samples
        self.build_samples()


    def build_samples(self):
        # == Create samples and group data ==
        self.groups = {}          # Dictionary mapping id to group dataframe
        self.samples = []         # List of (id, t) tuples indicating samples feature
        for identifier, group in self.df.groupby(self.features.id, sort=False):
            group = group.reset_index(drop=True)
            self.groups[identifier] = group

            # Generates all possible valid combinations of (id, t)
            last_t = len(group) - self.dec_len
            for t in range(self.enc_len, last_t + 1):
                self.samples.append((identifier, t))


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        identifier, t = self.samples[idx]
        group = self.groups[identifier]
        # Get the window of data
        enc_df = group.iloc[t - self.enc_len:t]
        dec_df = group.iloc[t:t + self.dec_len]

        # Get static features only once
        if self.features.static_categorical:
            static_categorial = torch.tensor(
                [int(group[c].iloc[0]) for c in self.features.static_categorical],
                dtype=torch.long
            )
        else:
            static_categorial = torch.zeros(0, dtype=torch.long)

        if self.features.static_continuous:
            static_continuous = torch.tensor(
                group[self.features.static_continuous].iloc[0].to_numpy(dtype=np.float32),
                dtype=torch.float32
            )
        else:
            static_continuous = torch.zeros(0, dtype=torch.float32)

        # Categorical features (known and observed) for encoder
        enc_cat_cols = self.features.known_categorical + self.features.observed_categorical
        if enc_cat_cols:
            enc_temp_cats = torch.from_numpy(
                enc_df[enc_cat_cols].to_numpy(dtype=np.int64)
            )
        else:
            enc_temp_cats = torch.zeros((self.enc_len, 0), dtype=torch.long)

        # Categorical features (just known) for decoder
        dec_cat_cols = self.features.known_categorical
        if dec_cat_cols:
            dec_temp_cats = torch.from_numpy(
                dec_df[dec_cat_cols].to_numpy(dtype=np.int64)
            )
        else:
            dec_temp_cats = torch.zeros((self.dec_len, 0), dtype=torch.long)

        # Continuous features (known and observed) for encoder
        enc_cont_cols = self.features.known_continuous + self.features.observed_continuous
        if len(enc_cont_cols) == 0:
            enc_temp_cont = torch.empty((self.enc_len, 0), dtype=torch.float32)
        else:
            enc_temp_cont = torch.tensor(
                enc_df[enc_cont_cols].to_numpy(dtype=np.float32),
                dtype=torch.float32
            )

        # Continuous features (just known) for decoder
        dec_cont_cols = self.features.known_continuous
        if len(dec_cont_cols) == 0:
            dec_temp_cont = torch.empty((self.dec_len, 0), dtype=torch.float32)
        else:
            dec_temp_cont = torch.tensor(
                dec_df[dec_cont_cols].to_numpy(dtype=np.float32),
                dtype=torch.float32
            )

        # Target values
        target = torch.tensor(
            dec_df[[self.features.target]].to_numpy(dtype=np.float32),
            dtype=torch.float32
        )

        return {
            "model_inputs": {
                "static_cats": static_categorial,
                "static_cont": static_continuous,
                "enc_temp_cats": enc_temp_cats,
                "enc_temp_cont": enc_temp_cont,
                "dec_temp_cats": dec_temp_cats,
                "dec_temp_cont": dec_temp_cont,
            },
            "target": target,
            "id": str(identifier),
            "cut_time": group[self.features.time].iloc[t].isoformat(),
        }
    
    @staticmethod
    def get_scalers(dataset):
        """Compute scalers for each time series in using training set."""
        real_scalers = {}    # Dictionary mapping id to StandardScaler for input continuous features
        target_scalers = {}  # Dictionary mapping id to StandardScaler for target
        for identifier, sliced in dataset.df.groupby(dataset.features.id, sort=False):
            real_scalers[identifier] = StandardScaler().fit(sliced[dataset.continuous_features].values)
            target_scalers[identifier] = StandardScaler().fit(sliced[[dataset.features.target]].values)
        
        return real_scalers, target_scalers
    

    def apply_scalers(
        self,
        real_scalers,
        target_scalers
    ):
        """Apply scalers to dataset."""
        for identifier, group in self.df.groupby(self.features.id, sort=False):
            if identifier in real_scalers:
                real_scaler = real_scalers[identifier]
                self.df.loc[group.index, self.continuous_features] = real_scaler.transform(
                    group[self.continuous_features].values
                )
            if identifier in target_scalers:
                target_scaler = target_scalers[identifier]
                self.df.loc[group.index, [self.features.target]] = target_scaler.transform(
                    group[[self.features.target]].values
                )

        # Rebuild samples after scaling
        self.build_samples()


# Test code
if __name__ == "__main__":
    
    # Create feature description for electricity dataset
    feature_description = FeatureDescription(
        id="categorical_id",
        time="date",
        target="power_usage",
        known_continuous=["hour", "day", "day_of_week", "month", "days_from_start", "hours_from_start","t"],
        known_categorical=["categorical_hour", "categorical_day_of_week"],
        static_categorical=[],
        static_continuous=[],
        observed_continuous=[],
        observed_categorical=[],
    )

    # Load dataset
    df = pd.read_csv("processed_power_usage.csv")
    
    # Split into train, val, test
    valid_boundary = 1315
    test_boundary  = 1339

    df_train = df[df["days_from_start"] < valid_boundary]
    df_val   = df[(df["days_from_start"] >= valid_boundary - 7) & (df["days_from_start"] < test_boundary)]
    df_test  = df[df["days_from_start"] >= test_boundary - 7]

    # Create datasets
    train_dataset = TimeSeriesDataset(
        df=df_train,
        feature_description=feature_description,
        encoder_length=168,
        decoder_length=24
    )
    # Get categorical encoder and scalers from training set
    categorical_encoder = train_dataset.categorical_encoder
    real_scalers, target_scalers = TimeSeriesDataset.get_scalers(train_dataset)

    val_dataset = TimeSeriesDataset(
        df=df_val,
        feature_description=feature_description,
        encoder_length=168,
        decoder_length=24,
        categorical_encoder=categorical_encoder
    )

    test_dataset = TimeSeriesDataset(
        df=df_test,
        feature_description=feature_description,
        encoder_length=168,
        decoder_length=24,
        categorical_encoder=categorical_encoder
    )

    # Apply scalers
    train_dataset.apply_scalers(real_scalers, target_scalers)
    val_dataset.apply_scalers(real_scalers, target_scalers)
    test_dataset.apply_scalers(real_scalers, target_scalers)

    dl = DataLoader(train_dataset, batch_size=32, shuffle=True)
    batch = next(iter(dl))

    print("Debug")
