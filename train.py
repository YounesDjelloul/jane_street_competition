class FinancialDataset(Dataset):
    def __init__(self, parquet_files, feature_cols=None):
        self.lazy_df = pl.scan_parquet(parquet_files)
        if feature_cols is None:
            self.feature_cols = [f'feature_{i:02d}' for i in range(79)]
        else:
            self.feature_cols = feature_cols

        self.length = self.lazy_df.select(pl.count()).collect().item()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        row = self.lazy_df.slice(idx, 1).collect()

        features = row.select(self.feature_cols).to_numpy().flatten()
        target = row.select('responder_6').to_numpy().flatten()

        return torch.FloatTensor(features), torch.FloatTensor(target)


class FinancialNN(torch.nn.Module):
    def __init__(self, input_size=79):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)


def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_features, batch_targets in train_loader:
            batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
                outputs = model(batch_features)
                val_loss += criterion(outputs, batch_targets).item()

        print(
            f'Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}')

    return model