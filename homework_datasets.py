import torch
from torch.utils.data import Dataset, DataLoader
import csv
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from typing import List, Optional, Tuple, Dict
from homework_model_modification import train_with_l1_l2_regularization_and_early_stopping, logistic_regression

class CSVDataset(Dataset):
    """
    Кастомный датасет для CSV файлов с предобработкой для PyTorch.
    """

    def __init__(
            self,
            filepath: str,
            target_column = None,
            numerical_columns = None,
            categorical_columns = None,
            binary_columns = None,
            normalize = True
    ):
        self.filepath = filepath
        self.numerical_columns = numerical_columns or []
        self.categorical_columns = categorical_columns or []
        self.binary_columns = binary_columns or []
        self.target_column = target_column

        # Загрузка и парсинг CSV
        self.raw_data, self.headers = self._load_csv()

        # Индексы колонок по типам
        self.header_to_idx = {h: i for i, h in enumerate(self.headers)}
        self._validate_columns()

        # Извлечение данных по типам
        self._extract_features()

        # Предобработка
        self._handle_missing_values()
        self._encode_binary()
        self._encode_categorical()

        if normalize and len(self.numerical_columns) > 0:
            num_data = self._get_numerical_data()
            if num_data.shape[1] > 0:
                self.scaler = StandardScaler()
                self.numerical_data = self.scaler.fit_transform(num_data)

        # Нормализация целевой переменной
        if normalize and self.target_data is not None:
            self.target_scaler = StandardScaler()
            self.target_data = self.target_scaler.fit_transform(self.target_data.reshape(-1, 1)).flatten()

        # Формирование финальных тензоров
        self._build_tensors()

    def _load_csv(self) -> Tuple[List[List], List[str]]:
        """Загрузка CSV файла."""
        with open(self.filepath, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            data = list(reader)
        return data, headers

    def _validate_columns(self):
        """Проверка существования колонок."""
        all_specified = (self.numerical_columns + self.categorical_columns +
                         self.binary_columns + ([self.target_column] if self.target_column else []))
        for col in all_specified:
            if col not in self.header_to_idx:
                raise ValueError(f"Колонка '{col}' не найдена в CSV файле")

    def _get_column_data(self, col_name: str) -> List:
        """Извлечение данных конкретной колонки."""
        idx = self.header_to_idx[col_name]
        return [row[idx] for row in self.raw_data]

    def _extract_features(self):
        """Извлечение признаков по типам."""
        self.numerical_data = np.array([
            [float(row[self.header_to_idx[col]]) if row[self.header_to_idx[col]] else np.nan
             for col in self.numerical_columns]
            for row in self.raw_data
        ]) if self.numerical_columns else np.array([])

        self.categorical_data = np.array([
            [row[self.header_to_idx[col]] for col in self.categorical_columns]
            for row in self.raw_data
        ]) if self.categorical_columns else np.array([])

        self.binary_data = np.array([
            [row[self.header_to_idx[col]] for col in self.binary_columns]
            for row in self.raw_data
        ]) if self.binary_columns else np.array([])

        if self.target_column:
            target_idx = self.header_to_idx[self.target_column]
            self.target_data = np.array([
                float(row[target_idx]) if row[target_idx] else np.nan
                for row in self.raw_data
            ])
        else:
            self.target_data = None

    def _handle_missing_values(self):
        """Заполнение пропусков."""
        if len(self.numerical_data) > 0:
            col_means = np.nanmean(self.numerical_data, axis=0)
            nan_mask = np.isnan(self.numerical_data)
            self.numerical_data[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

        if len(self.categorical_data) > 0:
            for col in range(self.categorical_data.shape[1]):
                mask = self.categorical_data[:, col] == ''
                if mask.any():
                    values, counts = np.unique(
                        self.categorical_data[~mask, col], return_counts=True
                    )
                    mode = values[np.argmax(counts)]
                    self.categorical_data[mask, col] = mode

    def _encode_binary(self):
        """Label Encoding для бинарных признаков."""
        if len(self.binary_data) == 0:
            return

        self.label_encoders = []
        encoded = np.zeros_like(self.binary_data, dtype=np.float32)

        for col in range(self.binary_data.shape[1]):
            le = LabelEncoder()
            encoded[:, col] = le.fit_transform(self.binary_data[:, col])
            self.label_encoders.append(le)

        self.binary_data = encoded

    def _encode_categorical(self):
        """OneHot Encoding для категориальных признаков."""
        if len(self.categorical_data) == 0:
            return

        self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.categorical_data = self.onehot_encoder.fit_transform(self.categorical_data)
        self.categorical_columns = []  # После one-hot становятся числовыми

    def _get_numerical_data(self) -> np.ndarray:
        """Объединение всех числовых данных."""
        parts = []
        if len(self.numerical_data) > 0:
            parts.append(self.numerical_data)
        if len(self.categorical_data) > 0:
            parts.append(self.categorical_data)
        if len(self.binary_data) > 0:
            parts.append(self.binary_data)
        return np.hstack(parts) if parts else np.array([])

    def _build_tensors(self):
        """Построение итоговых тензоров."""
        features = self._get_numerical_data()
        self.X = torch.FloatTensor(features)
        self.y = torch.FloatTensor(self.target_data).reshape(-1, 1) if self.target_data is not None else None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

    def get_dataloader(self, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    @property
    def input_dim(self) -> int:
        return self.X.shape[1]


def train_housing_model():
    dataset = CSVDataset(
        filepath='data/BostonHousing.csv',
        target_column='medv',
        numerical_columns=['crim', 'zn', 'indus', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'b', 'lstat'],
        categorical_columns=[],
        binary_columns=['chas'],
        normalize=True
    )
    dataloader = dataset.get_dataloader()
    model = train_with_l1_l2_regularization_and_early_stopping(dataloader, patience=1000000, l1_lambda=0, weight_decay=0, epochs=1000)
    print(model.state_dict()['linear.weight'])
    print(model.state_dict()['linear.bias'])

def train_titanic_model():
    dataset = CSVDataset(
        filepath='data/Titanic.csv',
        target_column='Survived',
        numerical_columns=['PassengerId', 'Pclass', "Age", "SibSp", "Parch","Fare", ],
        categorical_columns=["Name", "Cabin", "Embarked", "Ticket"],
        binary_columns=['Sex'],
        normalize=True
    )
    dataloader = dataset.get_dataloader()
    model = logistic_regression(dataloader, task_type="binary")
    print(model.state_dict()['linear.weight'])
    print(model.state_dict()['linear.bias'])

train_titanic_model()