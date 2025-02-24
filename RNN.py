import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Загрузка данных
data = pd.read_csv('AAPL1.csv')  
print("Доступные столбцы:", data.columns)  # Выведем названия столбцов для проверки

# Используем числовые значения из второго столбца (цены)
data = data.iloc[:, 1].values  # Берем второй столбец с ценами

# Нормализация данных
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data.reshape(-1, 1)).squeeze()

# Разделение на обучающую, валидационную и тестовую выборки
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.15)
train_data = data[:train_size]
val_data = data[train_size:train_size+val_size]
test_data = data[train_size+val_size:]

# Функция для создания последовательностей

def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:(i + seq_length)])
        labels.append(data[i + seq_length])
    return np.array(sequences), np.array(labels)

# Создание последовательностей для всех наборов данных
seq_length = 20  # Вернем прежнее значение
X_train, y_train = create_sequences(train_data, seq_length)
X_val, y_val = create_sequences(val_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Преобразование в формат PyTorch с правильной размерностью
X_train = torch.FloatTensor(X_train).unsqueeze(2)
y_train = torch.FloatTensor(y_train)
X_val = torch.FloatTensor(X_val).unsqueeze(2)
y_val = torch.FloatTensor(y_val)
X_test = torch.FloatTensor(X_test).unsqueeze(2)
y_test = torch.FloatTensor(y_test)

# Создание DataLoader для батч-обработки
batch_size = 32
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train, y_train), 
    batch_size=batch_size, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_val, y_val), 
    batch_size=batch_size
)

# Определение модели
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=256, output_size=1, num_layers=2):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out.squeeze()

model = LSTM(input_size=1, hidden_size=256, output_size=1, num_layers=2)

# Обучение модели
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 200

# Обучение модели с валидацией
best_val_loss = float('inf')
patience = 10
no_improve = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        # Добавим клиппинг градиентов для стабильности
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
    
    # Валидация
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            val_loss += criterion(outputs, batch_y).item()
    
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    # Ранняя остановка
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

# Загрузка лучшей модели для тестирования
model.load_state_dict(torch.load('best_model.pth'))

# Тестирование модели
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    predictions = predictions.numpy()
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    y_test_inverse = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))
    
    # Вычисляем и выводим метрики
    mse = np.mean((predictions - y_test_inverse) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - y_test_inverse))
    
    print("\nРезультаты тестирования:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")

# Визуализация результата
plt.figure(figsize=(14, 5))
plt.plot(y_test_inverse, label='Фактические цены')
plt.plot(predictions, label='Предсказанные цены', color='red')
plt.xlabel('Время')
plt.ylabel('Цена акции')
plt.title('Предсказание цены акции')
plt.legend()
plt.savefig('prediction_plot.png')
plt.close()

print("\nГрафик сохранен в файл 'prediction_plot.png'")
