from pathlib import Path

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split

script_dir = Path(__file__).resolve().parent
text_path = Path("data/shakespeare.txt")
unknown_key = "<UNK>"
seq_len = 50
loss_curve_path = script_dir / "loss_curve.png"

class LSTMModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.patience = 10

    def forward(self, x):
        x = self.embedding(x)
        lstm_output, (_, _) = self.lstm(x)
        output = self.fc(lstm_output)
        return output

    def fit(self, train_loader, val_loader, epochs=50):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005)
        loss_fn = nn.CrossEntropyLoss()
        trigger_times = 0
        best_val_loss = float('inf')

        train_loss = []
        val_loss = []
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                outputs = self(batch_X)
                loss = loss_fn(outputs.view(-1, outputs.size(-1)), batch_y.view(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_X.shape[0]

            epoch_val_loss = self._compute_validation_loss(val_loader, loss_fn)
            if epoch_val_loss < best_val_loss:
                trigger_times = 0
                best_val_loss = epoch_val_loss
            else:
                trigger_times += 1
                if trigger_times >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            epoch_loss = epoch_loss / len(train_loader.dataset)
            train_loss.append(epoch_loss)
            val_loss.append(epoch_val_loss)
            print(f"Epoch {epoch}: Train Loss: {epoch_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}")
        _plot_loss_curve(train_loss, val_loss)

    def predict(self, x):
        self.eval()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            outputs = self(x)
            last_logits = outputs[:, -1, :]
            probs = torch.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        return next_token.item()

    def _compute_validation_loss(self, val_loader, loss_fn):
        total_loss = 0
        self.eval()
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = self(batch_X)
                loss = loss_fn(outputs.view(-1, outputs.size(-1)), batch_y.view(-1))
                total_loss += loss.item() * batch_X.shape[0]
        return total_loss / len(val_loader.dataset)

def _plot_loss_curve(train_loss, val_loss):
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss, label='Train Loss', marker='o')
    plt.plot(val_loss, label='Validation Loss', marker='o')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(loss_curve_path)
    plt.close()
    print(f"Saved loss curve to {loss_curve_path}")

def _create_vocabulary(text):
    vocabulary = {"<PAD>": 0, "<UNK>": 1}
    next_index = 2
    for char in text:
        if char not in vocabulary:
            vocabulary[char] = next_index
            next_index += 1
    reversed_vocabulary = {v: k for k, v in vocabulary.items()}
    return vocabulary, reversed_vocabulary

def _tokenize(text, vocabulary):
    tokenized_text = []
    for char in text:
        if char not in vocabulary:
            tokenized_text.append(vocabulary.get(unknown_key))
        else:
            tokenized_text.append(vocabulary.get(char))
    return tokenized_text

def _generate_training_data(tokenized_text):
    x, y = [], []
    step = 3
    for i in range(0, len(tokenized_text) - seq_len, step):
        x.append(tokenized_text[i: i + seq_len])
        y.append(tokenized_text[i+1: i + seq_len + 1])

    X_tensor = torch.LongTensor(x)
    y_tensor = torch.LongTensor(y)

    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False)
    return train_loader, val_loader

def generate_text(seed, length, vocabulary, reversed_vocabulary, model):
    tokenized_seed = _tokenize(seed, vocabulary)
    tokenized_seed_tensor = torch.LongTensor(tokenized_seed).unsqueeze(0)
    for i in range(length):
        prediction = model.predict(tokenized_seed_tensor)
        prediction_tensor = torch.LongTensor([prediction]).unsqueeze(0)
        tokenized_seed_tensor = torch.cat((tokenized_seed_tensor, prediction_tensor), dim=1)
        tokenized_seed_tensor = tokenized_seed_tensor[:, -seq_len:] # keep last seq_len tokens

    token_list = tokenized_seed_tensor[0].tolist()
    result_text = [reversed_vocabulary.get(idx, "<UNK>") for idx in token_list]

    return "".join(result_text)

def main():
    text = text_path.read_text(encoding="utf-8")
    text = text[:100_000]  # only first 100k characters
    vocabulary, reversed_vocabulary = _create_vocabulary(text)
    tokenized_text = _tokenize(text, vocabulary)
    train_loader, val_loader = _generate_training_data(tokenized_text)
    model = LSTMModel(num_embeddings=len(vocabulary), embedding_dim=32, hidden_size=32, num_layers=1, output_size=len(vocabulary))

    model.fit(train_loader, val_loader)

    custom_text = "They fell together al"
    result_text = generate_text(custom_text, 100, vocabulary, reversed_vocabulary, model)
    print(result_text)

if __name__ == '__main__':
    main()