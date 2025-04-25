import torch
import torch.nn as nn
from tqdm import tqdm
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class Transformer_EEG(nn.Module):
    def __init__(self, input_channels, n_classes, d_model=128, n_heads=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        # x: (batch, channels, time) → (batch, time, channels)
        x = x.permute(0, 2, 1)

        # Project to model dimension
        x = self.input_proj(x)  # (batch, time, d_model)
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer_encoder(x)

        # Mean pooling across time
        x = x.mean(dim=1)  # (batch, d_model)

        return self.classifier(x)

    def train_model(self, 
                    train_loader, 
                    val_loader, 
                    class_weights, 
                    epochs=10,
                    lr=1e-3,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    checkpoint_path="best_eeg_transformer.pt",
                    save_latest=False, latest_path="latest_transformer.pt"):

        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_val_acc = 0.0

        for epoch in range(epochs):
            self.train()
            train_loss, correct, total = 0, 0, 0
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

            for x, y in loop:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = self(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * x.size(0)
                pred = torch.argmax(logits, dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

            epoch_loss = train_loss / total
            epoch_acc = 100. * correct / total
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc)
            print(f"[Train] Epoch {epoch+1}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%")

            # Validation
            if val_loader:
                self.eval()
                val_loss, val_correct, val_total = 0, 0, 0
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(device), y.to(device)
                        logits = self(x)
                        loss = criterion(logits, y)
                        val_loss += loss.item() * x.size(0)
                        val_pred = torch.argmax(logits, dim=1)
                        val_correct += (val_pred == y).sum().item()
                        val_total += y.size(0)

                val_loss /= val_total
                val_acc = 100. * val_correct / val_total
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                print(f"[Val] Epoch {epoch+1}: Loss={val_loss:.4f}, Acc={val_acc:.2f}%")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(self.state_dict(), checkpoint_path)
                    print(f"[V] Best model saved to {checkpoint_path}")

            if save_latest:
                torch.save(self.state_dict(), latest_path)
                print(f"[S] Latest model saved to {latest_path}")

        return history