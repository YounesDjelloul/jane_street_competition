class FinancialDataset(Dataset):
    def __init__(self, df):
        columns_to_drop = ['feature_21', 'feature_26', 'feature_27', 'feature_31']
        df = df.drop(columns=columns_to_drop)

        self.feature_cols = ([f'feature_{i:02d}' for i in range(79) if f'feature_{i:02d}' not in columns_to_drop] +
                             ['responder_0', 'responder_1', 'responder_2', 'responder_3',
                              'responder_4', 'responder_5', 'responder_7', 'responder_8'])

        for col in self.feature_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

        self.features = df[self.feature_cols].values
        self.targets = df['responder_6'].values

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.FloatTensor([self.targets[idx]])


class FinancialNN(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.BatchNorm1d(input_size),
            torch.nn.Linear(input_size, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.network(x)