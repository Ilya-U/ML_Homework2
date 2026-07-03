import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import make_regression_data, mse, log_epoch, RegressionDataset, ClassificationDataset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


class LinearRegression(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)


class LogisticRegression(nn.Module):
    def __init__(self, in_features, num_classes=1):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        return self.linear(x)


def train_with_l1_l2_regularization_and_early_stopping(dataloader, l1_lambda=0, weight_decay=1e-2, epochs=500, patience=100,
                                                       epsilon=1e-3):
    """Тренирует модель линейной регрессии"""
    first_batch = next(iter(dataloader))
    in_features = first_batch[0].shape[1]
    # Создаём модель, функцию потерь и оптимизатор
    model = LinearRegression(in_features=in_features)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01,
                          weight_decay=weight_decay)  # Инициализируем оптимизатор с weight_decay

    # Параметры для early stopping
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # Обучаем модель

    for epoch in range(1, epochs + 1):
        total_loss = 0

        for i, (batch_X, batch_y) in enumerate(dataloader):
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)
            l1_norm = sum(p.abs().sum() for p in model.parameters())  # Получим сумму всех параметров модели
            loss += l1_lambda * l1_norm  # Добавим к функции потерь
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (i + 1)

        # Если потери меньше лучших потерь, обновляем счетчик patience
        if avg_loss < best_loss - epsilon:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        # Если счетчик поднялся выше patience, завершаем обучение
        if patience_counter >= patience:
            print(f'Early stopping на эпохе {epoch}')
            model.load_state_dict(best_model_state)
            break

        if epoch % 10 == 0:
            log_epoch(epoch, avg_loss)

    return model

def make_classification_data(n=100, source='multiclass'):
    """Генерирует данные для классификации"""
    if source == 'binary':
        X = torch.rand(n, 2)
        w = torch.tensor([2.0, -3.0])
        b = 0.5
        logits = X @ w + b
        print(logits)
        y = (logits > 0).float().unsqueeze(1)
        return X, y
    elif source == 'multiclass':
        X = torch.rand(n, 2)
        w = torch.tensor([1.5, -1.0])
        b = 1.0
        logits = X @ w + b
        y = torch.zeros(n, dtype=torch.long)
        y[logits >= 0.5] = 1
        y[logits >= 1] = 2
        return X, y
    else:
        raise ValueError('Unknown source')

def accuracy(y_pred, y_true, task='binary'):
    """Вычисяет accuracy"""
    if task == 'binary':
        y_pred_bin = (y_pred > 0.5).float()
        return (y_pred_bin == y_true).float().mean().item()
    elif task == 'multiclass':
        y_pred_class = y_pred.argmax(dim=1)
        return (y_pred_class == y_true).float().mean().item()


def calculate_metrics(y_true, y_pred, y_prob=None, task='binary'):
    """Вычисление метрик precision, recall, F1-score, ROC-AUC"""
    y_true = y_true.cpu().numpy()

    if task == 'binary':
        y_pred_bin = (y_pred > 0.5).cpu().numpy().astype(int)
        y_true = y_true.astype(int)

        precision = precision_score(y_true, y_pred_bin, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred_bin, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred_bin, average='binary', zero_division=0)

        if y_prob is not None:
            y_prob = y_prob.detach().numpy()
            if len(np.unique(y_true)) > 1:
                roc_auc = roc_auc_score(y_true, y_prob)
            else:
                roc_auc = 0.5
        else:
            roc_auc = 0.0

    elif task == 'multiclass':
        y_pred_class = y_pred.argmax(dim=1).cpu().numpy()

        precision = precision_score(y_true, y_pred_class, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred_class, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred_class, average='macro', zero_division=0)

        if y_prob is not None:
            y_prob = y_prob.detach().numpy()
            if len(np.unique(y_true)) > 1:
                try:
                    roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
                except:
                    roc_auc = 0.5
            else:
                roc_auc = 0.5
        else:
            roc_auc = 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }


def logistic_regression(dataloader, task_type='multiclass', epochs=800):
    """Тренирует модель логистической регрессии"""
    print(f'Размер датасета: {len(dataloader.dataset)}')
    print(f'Количество батчей: {len(dataloader)}')
    print(f'Тип задачи: {task_type}')
    first_batch = next(iter(dataloader))
    in_features = first_batch[0].shape[1]
    # Создаём модель, функцию потерь и оптимизатор
    if task_type == 'binary':
        model = LogisticRegression(in_features=in_features, num_classes=1)
        criterion = nn.BCEWithLogitsLoss()
    else:
        model = LogisticRegression(in_features=in_features, num_classes=3)
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Обучаем модель
    for epoch in range(1, epochs + 1):
        total_loss = 0
        total_acc = 0
        all_y_true = []
        all_y_pred = []
        all_y_prob = []

        for i, (batch_X, batch_y) in enumerate(dataloader):
            optimizer.zero_grad()
            logits = model(batch_X)

            if task_type == 'binary':
                loss = criterion(logits, batch_y)
                y_pred = torch.sigmoid(logits)
                acc = accuracy(y_pred, batch_y, task='binary')
            else:
                loss = criterion(logits, batch_y.long())
                y_pred = torch.softmax(logits, dim=1)
                acc = accuracy(logits, batch_y, task='multiclass')

            loss.backward()
            optimizer.step()

            # Сохраняем предсказания для вычисления метрик
            all_y_true.append(batch_y)
            all_y_pred.append(y_pred)

            total_loss += loss.item()
            total_acc += acc

        avg_loss = total_loss / (i + 1)
        avg_acc = total_acc / (i + 1)

        # Вычисляем метрики для всей эпохи
        y_true_all = torch.cat(all_y_true)
        y_pred_all = torch.cat(all_y_pred)

        metrics = calculate_metrics(y_true_all, y_pred_all, y_pred_all, task=task_type)

        if epoch % 10 == 0:
            log_epoch(epoch, avg_loss, acc=avg_acc)
            print(f'  Precision: {metrics["precision"]:.4f}, Recall: {metrics["recall"]:.4f}, '
                  f'F1: {metrics["f1"]:.4f}, ROC-AUC: {metrics["roc_auc"]:.4f}')

    return model


if __name__ == "__main__":
    X, y = make_classification_data(n=200, source='multiclass')

    # Создаём датасет и даталоадер
    dataset = ClassificationDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Вызываем функцию
    model = logistic_regression(dataloader, task_type='multiclass', epochs=500)
    print(model.state_dict()['linear.weight'])
    print(model.state_dict()['linear.bias'])