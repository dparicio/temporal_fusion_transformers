import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import FeatureDescription, TimeSeriesDataset


class GLU(nn.Module):
    """Gated Linear Unit (GLU)"""
    # Inputs can be of shape [B, H] or [B, T, H]
    def __init__(self, input_size, hidden_size, dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.fc = nn.Linear(input_size, hidden_size * 2, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)

        # Split into activations and gates (pre-sigmoid)
        a, gate_pre = torch.split(x, self.hidden_size, dim=-1)

        gate = torch.sigmoid(gate_pre)
        y = a * gate
        return y, gate
    

class GRN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=None, dropout=0.0):
        super().__init__()
        output_size = output_size if output_size is not None else hidden_size

        # Dense layers main path (same notation as in paper)
        self.fc2 = nn.Linear(input_size, hidden_size, bias=True) 
        self.fc3 = nn.Linear(hidden_size, hidden_size, bias=False) # for context
        self.fc1 = nn.Linear(hidden_size, hidden_size, bias=True)

        self.glu = GLU(input_size=hidden_size, hidden_size=output_size, dropout=dropout)
        self.elu = nn.ELU()
        self.skip = nn.Identity() if input_size == output_size else nn.Linear(input_size, output_size, bias=False)
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, x, context=None, return_gate=False):
        x0 = x  # for skip connection

        # Compute nu2
        x = self.fc2(x)
        if context is not None:
            x = x + self.fc3(context)
        x = self.elu(x)

        # Compute nu1
        x = self.fc1(x)

        # Compute GRN as GLU + skip + layer norm
        y, gate = self.glu(x)
        out = self.layer_norm(self.skip(x0) + y)

        # Returns the GLU gate for diagnostic purposes (in the paper)
        return (out, gate) if return_gate else out
    

class MyModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.enc_len = params['encoder_length']
        self.dec_len = params['decoder_length']
        self.time_steps = params['time_steps']
        self.feature_description = params['feature_description']
        # Map number of embeddings per categorical variable
        self.embed_per_cat = params['embed_per_cat']
        # Hidden state size (common across TFT)
        self.d_model = params['d_model']
        self.dropout = params['dropout']

        # Input features
        self.static_categorical_inputs = len(self.feature_description.static_categorical)
        self.static_continuous_inputs = len(self.feature_description.static_continuous)
        self.known_categorical_inputs = len(self.feature_description.known_categorical)
        self.known_continuous_inputs = len(self.feature_description.known_continuous)
        self.obs_categorical_inputs = len(self.feature_description.observed_categorical)
        self.obs_continuous_inputs = len(self.feature_description.observed_continuous)
        self.n_cat = (self.static_categorical_inputs + 
                      self.known_categorical_inputs + 
                      self.obs_categorical_inputs)
        self.n_cont = (self.static_continuous_inputs + 
                       self.known_continuous_inputs + 
                       self.obs_continuous_inputs)

        # Embeddings
        self.linear_projections = nn.ModuleList([nn.Linear(1, self.d_model) for _ in range(self.n_cont)])
        self.embeddings = nn.ModuleList([nn.Embedding(num_embeddings=self.embed_per_cat[i], 
                                                      embedding_dim=self.d_model, 
                                                      padding_idx=0) for i in range(self.n_cat)])
        
        
        self.n_static = self.static_categorical_inputs + self.static_continuous_inputs
        if self.n_static > 0:
            # Static variable selection GRNs
            self.static_scorer = GRN(
                input_size=self.d_model * self.n_static,
                hidden_size=self.d_model,
                output_size=self.n_static,
                dropout=self.dropout
            )
            self.static_var_grns = nn.ModuleList([
                GRN(input_size=self.d_model, hidden_size=self.d_model, output_size=self.d_model, dropout=self.dropout)
                for _ in range(self.n_static)
            ])

            # Static covariate encoders GRNs (4 in total)
            self.static_context_grns = nn.ModuleList([
                GRN(input_size=self.d_model, hidden_size=self.d_model, output_size=self.d_model, dropout=self.dropout)
                for _ in range(4)
            ])

        # Temporal variable selection GRNs
        # Number of historical and future features
        self.n_hist = (self.obs_categorical_inputs + self.obs_continuous_inputs) + (self.known_categorical_inputs + self.known_continuous_inputs)
        self.n_fut  = (self.known_categorical_inputs + self.known_continuous_inputs)

        if self.n_hist > 0:
            self.temporal_scorer_hist = GRN(
                input_size=self.d_model * self.n_hist,   
                hidden_size=self.d_model,
                output_size=self.n_hist,                 
                dropout=self.dropout
            )
            self.temporal_var_grns_hist = nn.ModuleList([
                GRN(input_size=self.d_model, hidden_size=self.d_model, output_size=self.d_model, dropout=self.dropout)
                for _ in range(self.n_hist)              
            ])

        if self.n_fut > 0:
            self.temporal_scorer_fut = GRN(
                input_size=self.d_model * self.n_fut,    
                hidden_size=self.d_model,
                output_size=self.n_fut,                  
                dropout=self.dropout
            )
            self.temporal_var_grns_fut = nn.ModuleList([
                GRN(input_size=self.d_model, hidden_size=self.d_model, output_size=self.d_model, dropout=self.dropout)
                for _ in range(self.n_fut)               
            ])

        # Define LSTM encoder and decoder (NOTE: 1 layer only)
        self.lstm_enc = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )
        self.lstm_dec = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )

        # Final LSTM gating and output layer (NOTE: no dropout)
        self.temporal_gate = GLU(input_size=self.d_model, hidden_size=self.d_model, dropout=self.dropout)
        self.temporal_ln   = nn.LayerNorm(self.d_model)

    def input2embedding(
        self, 
        static_categorical_inputs, 
        static_continuous_inputs, 
        known_categorical_inputs, 
        known_continuous_inputs, 
        obs_categorical_inputs, 
        obs_continuous_inputs
    ):
        n_static_cat = self.static_categorical_inputs
        n_static_cont = self.static_continuous_inputs
        n_known_cat = self.known_categorical_inputs
        n_known_cont = self.known_continuous_inputs
        n_obs_cat = self.obs_categorical_inputs
        n_obs_cont = self.obs_continuous_inputs

        # == CATEGORICAL VARIABLES ==
        static_cat_embeddings = []
        for i in range(n_static_cat):
            static_cat_embeddings.append(self.embeddings[i](static_categorical_inputs[..., i]))

        obs_cat_embeddings = []
        for i in range(n_obs_cat):
            obs_cat_embeddings.append(self.embeddings[n_static_cat + i](obs_categorical_inputs[..., i]))

        known_cat_embeddings = []
        for i in range(n_known_cat):
            known_cat_embeddings.append(self.embeddings[n_static_cat + n_obs_cat + i](known_categorical_inputs[..., i]))

        # == CONTINUOUS VARIABLES ==
        static_cont_embeddings = []
        for i in range(n_static_cont):
            static_cont_embeddings.append(self.linear_projections[i](static_continuous_inputs[..., i:i+1]))    

        obs_cont_embeddings = []
        for i in range(n_obs_cont):
            obs_cont_embeddings.append(self.linear_projections[n_static_cont + i](obs_continuous_inputs[..., i:i+1]))

        known_cont_embeddings = []
        for i in range(n_known_cont):
            known_cont_embeddings.append(self.linear_projections[n_static_cont + n_obs_cont + i](known_continuous_inputs[..., i:i+1]))

        # Stack all embeddings along a new dimension
        static_embeddings = None
        obs_embeddings    = None
        known_embeddings  = None

        if static_cat_embeddings or static_cont_embeddings:
            static_embeddings = torch.stack(static_cat_embeddings + static_cont_embeddings, dim=-1) # [B, H, n_static]
        if obs_cat_embeddings or obs_cont_embeddings:
            obs_embeddings    = torch.stack(obs_cat_embeddings + obs_cont_embeddings, dim=-1) # [B, T_enc, H, n_obs]
        if known_cat_embeddings or known_cont_embeddings:
            known_embeddings  = torch.stack(known_cat_embeddings + known_cont_embeddings, dim=-1) # [B, T_dec, H, n_known]

        return static_embeddings, obs_embeddings, known_embeddings


    def static_variable_selection(self, static_embeddings):
        #TODO: handled when no static inputs (static_embeddings is None)

        B, H, n_static = static_embeddings.shape

        # Flatten inputs
        flat = static_embeddings.reshape(B, H*n_static)

        # GRN to compute variable weights
        scorer = self.static_scorer
        static_weights = torch.softmax(scorer(flat), dim=-1).unsqueeze(1) # shape [B, 1, n_static]

        # Apply per-variable GRN
        transfomed_list = []
        for i in range(n_static):
            ti = self.static_var_grns[i](static_embeddings[:, :, i]).unsqueeze(-1)
            transfomed_list.append(ti)
        trans_static_embeddings = torch.cat(transfomed_list, dim=-1) # shape [B, H, n_static]

        # Apply weights to static embeddings
        static_vec = (static_weights * trans_static_embeddings).sum(dim=-1) # shape [B, H]

        return static_vec, static_weights


    def static_covariate_encoders(self, static_vec):
        """Computes the static covariate encoders"""
        static_context_variable_selection = self.static_context_grns[0](static_vec)
        static_context_enrichment = self.static_context_grns[1](static_vec)
        static_context_state_h = self.static_context_grns[2](static_vec)
        static_context_state_c = self.static_context_grns[3](static_vec)

        # shape [B, H]
        return (static_context_variable_selection, 
                static_context_enrichment, 
                static_context_state_h, 
                static_context_state_c)


    def temporal_variable_selection(
            self, 
            temporal_embeddings, 
            static_context_variable_selection,
            mode):
        
        # Retrieve correct scorer and var_grns
        if mode == 'hist':
            scorer = self.temporal_scorer_hist
            var_grns = self.temporal_var_grns_hist
        elif mode == 'fut':
            scorer = self.temporal_scorer_fut
            var_grns = self.temporal_var_grns_fut

        # Flatten inputs
        B, T, H, N = temporal_embeddings.shape
        flat = temporal_embeddings.reshape(B, T, H*N)

        # Convert static context to correct dimension
        # Context dim: [B, H] -> [B, 1, H] -> [B, T, H]
        context = static_context_variable_selection.unsqueeze(1).expand(B, T, H)

        # Compute variable weights (#NOTE return_gate=True like in paper)
        score, static_gate = scorer(flat, context=context, return_gate=True)  # shape: [B, T, N]
        temporal_weights = torch.softmax(score, dim=-1).unsqueeze(2)  # shape: [B, T, 1, N]

        # Apply per-variable GRN. shape [B, T, H, N]
        trans_temporal_embeddings = torch.stack([var_grns[i](temporal_embeddings[..., i]) for i in range(N)], dim=-1)

        # Apply weights to temporal embeddings. shape [B, T, H]
        temporal_vec = (temporal_weights * trans_temporal_embeddings).sum(dim=-1) 

        return temporal_vec, temporal_weights, static_gate
    

    def lstm_encoder_decoder(self, hist_inputs, fut_inputs, static_context_state_h, static_context_state_c):
        # Static context initialization shape: [B, H] -> [1, B, H]
        h0 = static_context_state_h.unsqueeze(0)   
        c0 = static_context_state_c.unsqueeze(0)  

        # LSTM encoder
        enc_out, (h, c) = self.lstm_enc(hist_inputs, (h0, c0))  

        # LSTM decoder
        dec_out, _ = self.lstm_dec(fut_inputs, (h, c))  

        lstm_layer = torch.cat([enc_out, dec_out], dim=1) 
        return lstm_layer


    def forward(self, batch):
        # Get inputs
        stat_cats = batch["model_inputs"]["static_cats"]
        stat_cont = batch["model_inputs"]["static_cont"]
        obs_cats  = batch["model_inputs"]["obs_cats"]
        obs_cont  = batch["model_inputs"]["obs_cont"]
        know_cats = batch["model_inputs"]["known_cats"]
        know_cont = batch["model_inputs"]["known_cont"]

        # Get embeddings shapes: [B, H, n_static], [B, T, H, n_obs], [B, T, H, n_known]
        static_embeddings, obs_embeddings, known_embeddings = self.input2embedding(
            static_categorical_inputs=stat_cats,
            static_continuous_inputs=stat_cont,
            known_categorical_inputs=know_cats,
            known_continuous_inputs=know_cont,
            obs_categorical_inputs=obs_cats,
            obs_continuous_inputs=obs_cont
        )  

        # Split temporal embeddings into historical and future
        if obs_embeddings is None:
            historical_inputs = known_embeddings[:, :self.enc_len, :, :]
        elif known_embeddings is None:
            historical_inputs = obs_embeddings[:, :self.enc_len, :, :]
        else:
            historical_inputs = torch.cat([obs_embeddings, known_embeddings[:, :self.enc_len, :, :]], dim=-1)
        future_inputs     = known_embeddings[:, self.enc_len:, :, :]     

        # Apply variable selection network to static inputs
        static_vec, static_weights = self.static_variable_selection(static_embeddings)

        # Get static covariate encoders
        static_context_variable_selection, static_context_enrichment, static_context_state_h, static_context_state_c = self.static_covariate_encoders(static_vec)

        # Apply temporal variable selection to historical and future inputs
        hist_features, hist_flags, _ = self.temporal_variable_selection(
            historical_inputs, static_context_variable_selection, mode="hist"
        )
        fut_features, fut_flags, _ = self.temporal_variable_selection(
            future_inputs, static_context_variable_selection, mode="fut"
        )

        # LSTM encoder-decoder
        lstm_layer = self.lstm_encoder_decoder(
            hist_features, 
            fut_features, 
            static_context_state_h, 
            static_context_state_c
        )
        # Pass through final gating layer
        lstm_gated, _ = self.temporal_gate(lstm_layer)
        
        # Create input embeddings for final gating layer
        input_embeddings = torch.cat([hist_features, fut_features], dim=1)
        # Add and layer norm
        temporal_feature_layer = self.temporal_ln(lstm_gated + input_embeddings)
    
        print("Debug")


        # _, (h, c) = self.enc(enc_in)                     # h,c: [num_layers, B, H]
        # dec_out, _ = self.dec(dec_in, (h, c))            # [B, Td, H]
        # yhat = self.head(dec_out)                        # [B, Td, 1]
        return None



# Test code
if __name__ == "__main__":
    
    # Create feature description for electricity dataset
    feature_description = FeatureDescription(
        id="categorical_id",
        time="date",
        target="power_usage",
        known_continuous=["hour", "day", "day_of_week", "month", "days_from_start", "hours_from_start","t"],
        known_categorical=["categorical_hour", "categorical_day_of_week"],
        static_categorical=["categorical_id"],
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create params
    params = {
        "encoder_length": train_dataset.enc_len,
        "decoder_length": train_dataset.dec_len,
        "time_steps": train_dataset.time_steps,
        "feature_description": feature_description,
        "embed_per_cat": train_dataset.get_embedding_per_cat(),
        "d_model": 64,
        "dropout": 0.1,
    }

    model = MyModel(params=params).to(device)

    # move batch to device
    for k, v in batch["model_inputs"].items():
        if isinstance(v, torch.Tensor):
            batch["model_inputs"][k] = v.to(device)
    batch["target"] = batch["target"].to(device)

    opt = optim.Adam(model.parameters(), lr=1e-3)
    for step in range(500):
        opt.zero_grad()
        pred = model(batch)
        loss = F.mse_loss(pred, batch["target"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step+1) % 50 == 0:
            print(f"step {step+1}: loss={loss.item():.6f}")

