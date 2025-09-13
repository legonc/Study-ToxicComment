# Study-ToxicComment

Проект **Study-ToxicComment** посвящен анализу токсичности комментариев на английском языке. В рамках проекта реализован полный пайплайн: от предобработки текста до обучения моделей машинного обучения и трансформеров, включая извлечение эмбеддингов и fine-tuning MiniLM.

---

## 📌 Цель проекта

- Определять, является ли комментарий токсичным или нет.
- Сравнивать простые модели (TF-IDF + Logistic Regression) с трансформерами (MiniLM).
- Изучать методы обработки текста, балансировки классов и извлечения эмбеддингов.

---

## 📁 Структура репозитория

```
Study-ToxicComment/
│
├─ notebooks/               # Jupyter ноутбуки с анализом и обучением
├─ src/                     # Скрипты и утилиты для обработки данных и обучения
├─ .gitignore               # Файл для исключения ненужных файлов из Git
├─ .python-version          # Версия Python для проекта
├─ LICENSE                  # Лицензия проекта
├─ pyproject.toml           # Конфигурация зависимостей и проекта
├─ README.md                # Описание проекта
├─ uv.lock                  # Файл блокировки зависимостей для uv
```

---

## ⚙️ Используемые технологии

- Python 3.11+
- Pandas, NumPy, Scikit-learn
- PyTorch, HuggingFace Transformers
- Matplotlib, Seaborn
- Kaggle для экспериментов и хранения моделей
- uv для управления зависимостями

---

## 🛠 Установка и запуск

### Клонирование репозитория

1. Склонируйте репозиторий:

   ```bash
   git clone https://github.com/legonc/Study-ToxicComment.git
   cd Study-ToxicComment
   ```

### Настройка окружения с помощью uv

1. Убедитесь, что у вас установлен `uv`. Если нет, установите его:

   ```bash
   pip install uv
   ```

2. Установите зависимости, указанные в `uv.lock`:

   ```bash
   uv sync
   ```

3. (Опционально) Если вы хотите создать новое виртуальное окружение:

   ```bash
   uv venv
   source .venv/bin/activate  # Для Linux/MacOS
   .venv\Scripts\activate      # Для Windows
   ```

4. Проверьте, что все зависимости установлены:

   ```bash
   uv pip list
   ```

### Запуск проекта

- Для экспериментов используйте Jupyter ноутбуки в папке `notebooks/`:

  ```bash
  jupyter notebook
  ```
- Для запуска скриптов используйте Python:

  ```bash
  python src/<script_name>.py
  ```

---

## 🧩 Предобработка данных

- Очистка текста от лишних символов и опечаток.
- Вычисление признаков:
  - `caps_ratio` — доля заглавных букв.
  - `toxic_word_count` — количество токсичных слов.
  - `clean_text_length` — длина очищенного текста.
- Удаление неинформативных колонок.
- Разделение на train/test с сохранением стратификации по классу.

---

## 🔹 Baseline модель

- **TF-IDF + Logistic Regression**
- Результаты на тестовой выборке:
  - F1-score (weighted): 0.9311
  - ROC-AUC: 0.9610
  - Precision (toxic): 0.5922
  - Recall (toxic): 0.8514

> Baseline модель дает нижнюю границу метрики и позволяет сравнить результаты с трансформером.

---

## 🔹 Fine-tuning MiniLM

- **Модель:** `microsoft/MiniLM-L12-H384-uncased`
- Используется weighted sampler для балансировки классов.
- Результаты на тестовой выборке:
  - F1-score (weighted): 0.9560
  - Precision (toxic): 0.7260
  - Recall (toxic): 0.8801
  - ROC-AUC: 0.9213
- Размер эмбеддингов: 384
- Скорость инференса: \~24 ms на текст

> Модель показывает высокое качество предсказаний при минимальном переобучении.

---

## 💡 Рекомендации по улучшению

- Использовать более мощные трансформеры (RoBERTa, DeBERTa, BERT-large).
- Подбирать гиперпараметры через grid-search или Optuna.
- Решать проблему дисбаланса классов: oversampling, undersampling, Focal Loss.
- Устранять слипшиеся слова, опечатки и шум в тексте.
- Применять аугментацию текста (back-translation, synonym replacement).
- Энсамблировать модели для повышения качества.
- Оптимизировать инференс с помощью ONNX или TorchScript.

---

## 🛠 Пример использования модели

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Загрузка модели и токенизатора
model_path = "models/minilm_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# Пример предсказания
text = "I can't believe how terrible this product is!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
with torch.no_grad():
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
print("Toxic" if pred == 1 else "Non-toxic")
```

---

## 🔗 Полезные ссылки

- Репозиторий GitHub: https://github.com/legonc/Study-ToxicComment
- Kaggle Notebook с полным процессом: https://www.kaggle.com/code/legonc/study-toxiccomment

---

## ⚠️ Примечания

- Для обучения MiniLM требуется GPU с CUDA.
- Для быстрого эксперимента можно использовать TF-IDF + Logistic Regression без GPU.
- Все результаты, модели и эмбеддинги сохранены в папках `models/` и `outputs/`.
- **За файлами проекта (обученная модель, эмбеддинги и т.д., пишите в тг).