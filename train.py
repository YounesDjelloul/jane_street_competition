def train_model(model, train_loader, val_loader, epochs=10, lr=0.0005):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_features, batch_targets in train_loader:
            batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0
        predictions = []
        actuals = []

        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
                outputs = model(batch_features)
                val_loss += criterion(outputs, batch_targets).item()

                predictions.extend(outputs.cpu().numpy())
                actuals.extend(batch_targets.cpu().numpy())

        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)

        predictions = np.array(predictions)
        actuals = np.array(actuals)
        r2 = 1 - np.sum((predictions - actuals) ** 2) / np.sum((actuals - actuals.mean()) ** 2)

        print(f'Epoch {epoch + 1}:')
        print(f'  Train Loss: {train_loss:.6f}')
        print(f'  Val Loss: {val_loss:.6f}')
        print(f'  R2 Score: {r2:.6f}')

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

    return model
