import torch
import torch.nn as nn
from tqdm import tqdm

class CNN2D_EEG(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),

            nn.Conv2d(16, 32, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # (B, 64, 1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),          # (B, 64)
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add fake channel dim: (B, 1, C, T)
        x = self.encoder(x)
        x = self.classifier(x)
        return x

    def train_model(self, 
                    train_loader, 
                    val_loader,
                    class_weights,  
                    epochs=10, 
                    lr=1e-3, 
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    checkpoint_path="best_model_2d.pt",
                    save_latest=False,
                    latest_path="latest_model_2d.pt"):

        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }

        best_val_acc = 0.0

        for epoch in range(epochs):
            self.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_x, batch_y in loop:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = self(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(outputs, dim=1)
                train_correct += (preds == batch_y).sum().item()
                train_total += batch_y.size(0)

                loop.set_postfix(loss=loss.item(), acc=100. * train_correct / train_total)

            epoch_train_loss = train_loss / train_total
            epoch_train_acc = 100. * train_correct / train_total

            history['train_loss'].append(epoch_train_loss)
            history['train_acc'].append(epoch_train_acc)

            print(f"[Train] Epoch {epoch+1}: Loss = {epoch_train_loss:.4f}, Acc = {epoch_train_acc:.2f}%")

            # Validation
            if val_loader:
                self.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for val_x, val_y in val_loader:
                        val_x, val_y = val_x.to(device), val_y.to(device)
                        val_outputs = self(val_x)
                        loss = criterion(val_outputs, val_y)

                        val_loss += loss.item() * val_x.size(0)
                        val_preds = torch.argmax(val_outputs, dim=1)
                        val_correct += (val_preds == val_y).sum().item()
                        val_total += val_y.size(0)

                epoch_val_loss = val_loss / val_total
                epoch_val_acc = 100. * val_correct / val_total

                history['val_loss'].append(epoch_val_loss)
                history['val_acc'].append(epoch_val_acc)

                print(f"[Val]   Epoch {epoch+1}: Loss = {epoch_val_loss:.4f}, Acc = {epoch_val_acc:.2f}%")

                if epoch_val_acc > best_val_acc:
                    best_val_acc = epoch_val_acc
                    torch.save(self.state_dict(), checkpoint_path)
                    print(f"[V] Saved best model to {checkpoint_path} (Acc: {epoch_val_acc:.2f}%)")

            if save_latest:
                torch.save(self.state_dict(), latest_path)
                print(f"[S] Saved latest model to {latest_path}")

        return history