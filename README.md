# Study-ToxicComment

Проект **Study-ToxicComment** посвящен бинарной классификации токсичных комментариев на английском языке. Реализован полный пайплайн: от предобработки текста до обучения моделей, включая baseline (TF-IDF + Logistic Regression) и fine-tuning трансформера `unitary/toxic-bert`. Основной акцент — на достижение метрики F1 ≥ 0.75 для интернет-магазина "Викишоп" с целью автоматизации модерации контента.

---

## 📌 Изменения в новой версии

**Версия 2.0** (по сравнению с предыдущей версией):  
- **Модель**: Заменена `microsoft/MiniLM-L12-H384-uncased` на `unitary/toxic-bert`, оптимизированную для классификации токсичности (обучена на Jigsaw датасетах).  
- **Метрики**: Удален `weighted` F1-score, использован стандартный F1-score для корректной оценки (0.8403 на тесте).  
- **Предобработка**: Улучшена обработка спама (сжатие повторов, удаление текстов < 3 символов), что сократило долю обрезаемых текстов для BERT.  
- **Обучение**: Использована подвыборка 50k для ускорения, добавлены веса классов в CrossEntropyLoss ([0.1, 0.9]).  
- **Инференс**: Проверена работа на текстах с орфографическими ошибками (F1=0.8403, время инференса ~34 мс).  
- **Рекомендации**: Добавлены конкретные предложения по снижению переобучения (early stopping, dropout) и оптимизации продакшна (TorchScript, порог классификации).  
- **Документация**: README обновлен с акцентом на результаты, рекомендации и структуру. Исключены упоминания эмбеддингов, так как они не использовались.  

---

## 📌 Цель проекта

- Классифицировать комментарии на токсичные (1) и нетоксичные (0) с метрикой F1 ≥ 0.75.  
- Сравнить baseline-модель (TF-IDF + Logistic Regression) с fine-tuning трансформера `unitary/toxic-bert`.  
- Подготовить решение для продакшна с учетом скорости инференса и качества классификации.  

---

## 📁 Структура репозитория

```
Study-ToxicComment/
│
├─ notebooks/               # Jupyter ноутбуки с EDA, обучением и инференсом
├─ src/                    # Скрипты для предобработки, обучения и оценки
├─ models/                 # Сохраненные модели (toxic-bert, токенизатор)
├─ outputs/                # Результаты обучения и метрики
├─ .gitignore              # Игнорируемые файлы
├─ .python-version         # Версия Python (3.11+)
├─ LICENSE                 # Лицензия (MIT)
├─ pyproject.toml          # Зависимости проекта
├─ uv.lock                 # Фиксация зависимостей
├─ README.md               # Описание проекта
```

---

## ⚙️ Используемые технологии

- Python 3.11+  
- Pandas, NumPy, Scikit-learn  
- PyTorch, HuggingFace Transformers  
- Matplotlib, Seaborn, WordCloud  
- Kaggle для обучения на GPU  
- uv для управления зависимостей  

**Изменения**:  
- Добавлен WordCloud для визуализации частотных слов.  

---

## 🛠 Установка и запуск

### Клонирование репозитория

```bash
git clone https://github.com/legonc/Study-ToxicComment.git
cd Study-ToxicComment
```

### Настройка окружения с помощью uv

1. Установите `uv`, если не установлен:  
   ```bash
   pip install uv
   ```

2. Синхронизируйте зависимости:  
   ```bash
   uv sync
   ```

3. (Опционально) Создайте виртуальное окружение:  
   ```bash
   uv venv
   source .venv/bin/activate  # Linux/MacOS
   .venv\Scripts\activate      # Windows
   ```

4. Проверьте установленные зависимости:  
   ```bash
   uv pip list
   ```

### Запуск проекта

- Для анализа и экспериментов:  
  ```bash
  jupyter notebook notebooks/
  ```
- Для запуска скриптов:  
  ```bash
  python src/<script_name>.py
  ```

---

## 🧩 Предобработка данных

- **Очистка текста**: Приведение к нижнему регистру, удаление не-ASCII символов (~10.8%), пунктуации, стоп-слов.  
- **Обработка спама**: Сжатие чередующихся повторов (max 2), удаление текстов короче 3 символов (807 строк).  
- **Лемматизация**: Приведение слов к базовой форме (NLTK WordNetLemmatizer).  
- **Признаки**:  
  - `caps_ratio`: Доля заглавных букв (r=0.221 с toxic).  
  - `toxic_word_count`: Количество токсичных слов (mean: 1.07 для токсичных, 0.15 для нетоксичных).  
  - `clean_text_length`: Длина очищенного текста.  
- **Разделение данных**: Train (50k строк, стратификация), Test (31 859 строк).  
- **Токенизация**: AutoTokenizer (`unitary/toxic-bert`), max_length=512, паддинг и усечение.   

---

## 🔹 Baseline модель (TF-IDF + Logistic Regression)

- **Описание**: Логистическая регрессия с TF-IDF (max_features=5000, ngram_range=(1,1)) и числовыми признаками (`caps_ratio`, `toxic_word_count`, `clean_text_length`).  
- **Гиперпараметры**: Подбор через RandomizedSearchCV (C=10, l2-пенализация).  
- **Результаты**:  
  | Метрика       | Train  | Test  |  
  |---------------|--------|-------|  
  | F1-score      | 0.801  | 0.763 |  
  | Precision     | 0.904  | 0.874 |  
  | Recall        | 0.724  | 0.677 |  
  | ROC-AUC       | 0.979  | 0.958 |  

- **Вывод**: Надежная модель, но слабее на сложных текстах (например, с ошибками). Переобучение минимальное (ΔF1=0.038).  

**Изменения**:  
- Удален `weighted` F1-score, добавлен стандартный F1-score.  
- Уточнены метрики и выводы о переобучении.

---

## 🔹 Fine-tuning Toxic-BERT

- **Модель**: `unitary/toxic-bert` (обучена на Jigsaw датасетах для токсичности).  
- **Обучение**:  
  - Подвыборка: 50k строк (train).  
  - Эпохи: 2, батчи: 2084 (train), 1328 (test).  
  - Оптимизатор: AdamW (lr=2e-5), CrossEntropyLoss с весами [0.1, 0.9].  
  - WeightedRandomSampler для балансировки классов.  
  - Смешанная точность (GradScaler) для ускорения.  
- **Результаты**:  
  | Метрика       | Train  | Test  |  
  |---------------|--------|-------|  
  | F1-score      | 0.9911 | 0.8403|  
  | Precision     | 0.9830 | 0.7826|  
  | Recall        | 0.9994 | 0.9073|  
  | ROC-AUC       | 0.9988 | 0.9856|  

- **Инференс**: ~34 мс/текст на GPU, успешно классифицирует тексты с ошибками (например, "fuckng" → toxic).  
- **Вывод**: Превосходит baseline (F1=0.8403 > 0.75), высокая обобщаемость (ROC-AUC=0.9856), но умеренное переобучение (ΔF1=0.151).  

**Изменения**:  
- Замена MiniLM на `unitary/toxic-bert`.  
- Уточнены параметры обучения и метрики.  

---

## 💡 Рекомендации по улучшению

1. **Снижение переобучения**:  
   - Добавить dropout (0.1–0.3), weight decay (0.01).  
   - Использовать early stopping по val_loss.  
   - Увеличить эпохи до 3–4 с мониторингом F1.  
2. **Балансировка классов**:  
   - Oversampling токсичных примеров или Focal Loss.  
3. **Аугментация данных**:  
   - Back-translation, synonym replacement, исправление опечаток.  
4. **Оптимизация порога**:  
   - Подобрать threshold (>0.5) для максимизации F1 (цель: >0.85).  
5. **Продакшн**:  
   - Использовать TorchScript/ONNX для ускорения инференса.  
   - Логировать false positives/negatives, retrain ежемесячно.  
   - Для ограниченных ресурсов: TF-IDF + эмбеддинги Toxic-BERT (~2k примеров).  
6. **Масштабирование**:  
   - Рассмотреть multilingual Toxic-BERT для мультиязычности.  


---

## 🛠 Пример использования модели

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Загрузка модели и токенизатора
model_path = "models/minilm_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval().to("cuda")

# Пример текста
text = "I cn't beleive how terribl this product is!!! Evrything broke in 2 days, fuckng custmer service is uselesss and i regret ever buying it. Awful experiance."
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
inputs = {k: v.to("cuda") for k, v in inputs.items()}

# Предсказание
with torch.no_grad():
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
print("Toxic" if pred == 1 else "Non-toxic")  # Вывод: Toxic
```


---

## 🔗 Полезные ссылки

- Репозиторий GitHub: [https://github.com/legonc/Study-ToxicComment](https://github.com/legonc/Study-ToxicComment)  
- Kaggle Notebook: [https://www.kaggle.com/code/legonc/study-toxiccomment](https://www.kaggle.com/code/legonc/study-toxiccomment)  
 

---

## ⚠️ Примечания

- Для fine-tuning Toxic-BERT требуется GPU с CUDA (обучение на Kaggle).  
- Baseline (TF-IDF + Logistic Regression) работает без GPU, подходит для быстрого тестирования.  
- Модель и токенизатор сохранены в `models/minilm_model/`.  
- Результаты (метрики, визуализации) в `outputs/`.  
- **За файлами проекта (обученная модель, метрики и т.д., пишите в тг).**

---

## 📊 Итоговые результаты

- **Baseline (TF-IDF + Logistic Regression)**: F1=0.763, ROC-AUC=0.958, минимальное переобучение.  
- **Toxic-BERT**: F1=0.8403, ROC-AUC=0.9856, умеренное переобучение, но высокая точность на сложных текстах.  
- Модель готова для продакшна: высокая скорость инференса (~34 мс), точность выше целевой (F1>0.75).  


---