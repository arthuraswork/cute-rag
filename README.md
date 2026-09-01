# RAG sdk на ChromaDB + Ollama

## Описание

SDK для семантического поиска и генерации ответов с использованием **Retrieval-Augmented Generation (RAG)**.  
Реализована на Python с использованием:
- **ChromaDB** — векторное хранилище
- **Sentence-Transformers** — эмбеддинги текста
- **Ollama** — локальные LLM для генерации ответов
- **uv** — менеджер зависимостей

---

## Структура проекта

```
.
├── src/
│   ├── stor.py          # класс Stor для работы с ChromaDB
│   ├── ollamas.py       # перечисление доступных моделей Ollama
│   └── llm.py           # класс OllamaRAG для генерации
├── testdocs/            # папка с документами для загрузки
├── chroma_data/         # папка с постоянным хранилищем Chroma
├── pyproject.toml       # зависимости и метаданные (uv)
└── README.md
```

---

## Установка

### 1. Установите `uv`

```bash
pip install uv
```

### 2. Инициализируйте виртуальное окружение и установите зависимости

```bash
uv init
uv add chromadb sentence-transformers langchain-text-splitters langchain-community ollama
```

### 3. Установите Ollama и загрузите модель

```bash
ollama pull gemma3:4b
# или другую модель из списка OllamaRAGModels
```

---

## Использование

### 1. Импорт библиотек и создание экземпляра хранилища

```python
from src.stor import Stor
from src.llm import OllamaRAG, OllamaRAGModels, format_context, to_text

# Создаём хранилище
s = Stor()
```

### 2. Добавление документа в коллекцию

```python
# Добавляем документ в коллекцию 'altlinux_late'
s.append('altlinux_late', './testdocs/documentation.txt')
s.append('altlinux_late', './testdocs/license.txt')
```

### 3. Поиск по тексту

```python
prompt = 'несут ли ответственность правообладатели'

# Выполняем запрос к векторному хранилищу
result = s.text_query('altlinux_late', prompt, k=4)

# Форматируем контекст
context = format_context(result)
```

### 4. Генерация ответа через LLM

```python
# Создаём модель
model = OllamaRAG(OllamaRAGModels.LIGHT_GEMMA_3_4B)

# Генерируем ответ на основе контекста
response = to_text(model.generate(prompt, context))

print(response)
```

---

## Класс `Stor`

### Параметры конструктора

| Параметр | Значение по умолчанию | Описание |
|----------|------------------------|----------|
| `db_path` | `'./chroma_data'` | Путь к папке с векторным хранилищем |
| `model_name` | `'all-MiniLM-L6-v2'` | Модель для эмбеддингов |
| `chunk_size` | `512` | Размер чанка при разбивке текста |
| `chunk_overlap` | `64` | Перекрытие чанков |
| `batch_size` | `256` | Размер батча при добавлении |

### Методы

| Метод | Описание |
|-------|----------|
| `add(collection_name, path_to_file, chunks)` | Добавляет чанки в коллекцию с метаданными |
| `append(collection_name, path_to_file)` | Загружает файл, разбивает и добавляет в коллекцию |
| `get_collection(collection_name)` | Возвращает или создаёт коллекцию |
| `list_collections()` | Возвращает список всех коллекций |
| `text_query(collection_name, text, k=3)` | Поиск ближайших чанков по тексту |
| `dump_collection(collection_name)` | Выгружает все данные из коллекции |
| `load_file(path_to_file)` | Загружает текстовый файл |
| `create_chunks(text)` | Разбивает текст на чанки |
| `vectorise(text)` | Преобразует текст в вектор |

---

## Класс `OllamaRAG`

### Параметры

| Параметр | Значение по умолчанию | Описание |
|----------|------------------------|----------|
| `model_name` | — | Название модели Ollama |
| `guidelines` | `'Отвечай строго по документу...'` | Системный промпт |
| `template` | `"Документ: {content}\nВопрос: {prompt}..."` | Шаблон для промпта |

### Методы

| Метод | Описание |
|-------|----------|
| `generate(prompt, context)` | Генерирует ответ на основе промпта и контекста |
| `_format_prompt(content, user_prompt)` | Подставляет контекст и вопрос в шаблон |

---

## Функции-помощники

```python
def format_prompt(template, content, user_prompt)  # форматирует шаблон
def format_context(content: QueryResult) -> str   # извлекает документы из результата
def to_text(model_response) -> str               # извлекает текст из ответа Ollama
```

---

## Пример полного пайплайна

```python
from src.stor import Stor
from src.llm import OllamaRAG, OllamaRAGModels, format_context, to_text

# 1. Инициализация
stor = Stor()

# 2. Загрузка документов
stor.append('legal', './docs/contract.txt')

# 3. Запрос
user_question = 'Какие обязательства у сторон?'
result = stor.text_query('legal', user_question, k=3)
context = format_context(result)

# 4. Генерация ответа
rag = OllamaRAG(OllamaRAGModels.LIGHT_GEMMA_3_4B)
response = to_text(rag.generate(user_question, context))

print(response)
```

## Лицензия

MIT