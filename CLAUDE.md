# agents-and-vibecode

Коллекция AI-агентов на Python. Каждый агент наследует базовый класс и регистрирует инструменты через декоратор `@tool`.

## Структура проекта

```
agents-and-vibecode/
├── agents/
│   ├── base.py           — BaseAgent (Anthropic/Claude API)
│   ├── ollama_base.py    — OllamaAgent (локальная LLM, без API-ключей)
│   ├── photo_agent.py    — PhotoAgent: Ollama + Stable Diffusion Forge
│   ├── photo_runner.py   — PhotoRunner: батч-генерация без LLM
│   └── youtuber.py       — YoutuberAgent: автоматизация YouTube
├── examples/
│   └── run_photo_agent.py — CLI-запуск PhotoAgent
├── prompts/
│   └── example_prompts.txt — пример файла промптов
├── outputs/              — сгенерированные PNG (в .gitignore)
└── .env.example          — шаблон переменных окружения
```

---

## PhotoAgent — Ollama + Stable Diffusion Forge

### Что делает

1. Читает промпты из текстового файла (по одному на строку)
2. Отправляет каждый в Stable Diffusion Forge через REST API
3. Ждёт завершения генерации
4. Сохраняет PNG в `outputs/` с именем `20250607_143022_a_fantasy_landscape.png`
5. Переходит к следующему промпту

Логика (когда вызывать инструменты и в каком порядке) управляется **Ollama** — локальной LLM. Никаких API-ключей не требуется.

---

### Быстрый старт

#### 1. Установить зависимости

```bash
pip install -e ".[dev]"
# или
pip install openai requests python-dotenv anthropic
```

#### 2. Запустить Ollama

```bash
ollama serve
ollama pull llama3.1   # модель с поддержкой tool calling
```

Другие совместимые модели: `llama3.2`, `llama3.3`, `mistral-nemo`, `qwen2.5`, `qwen2.5-coder`

#### 3. Запустить Stable Diffusion Forge с API

В командной строке или bat-файле добавь флаг `--api`:

```bash
python launch.py --api
# или в WebUI bat-файле добавить --api в COMMANDLINE_ARGS
```

Forge будет доступен на `http://127.0.0.1:7860` (по умолчанию).

#### 4. Создать `.env`

```bash
cp .env.example .env
```

Минимальный `.env` для PhotoAgent:

```env
OLLAMA_MODEL=llama3.1
OLLAMA_URL=http://localhost:11434
PHOTO_BACKEND=forge
PHOTO_API_URL=http://127.0.0.1:7860
PHOTO_OUTPUT_DIR=./outputs
```

#### 5. Написать промпты

Файл `prompts/example_prompts.txt` — по одному промпту на строку. Строки с `#` и пустые игнорируются:

```
# Мои промпты
a fantasy landscape at sunset, oil painting, 8k
portrait of a cyberpunk samurai, neon lights, photorealistic
<lora:add_detail:0.8> cute robot, studio lighting
```

#### 6. Запустить

```bash
python examples/run_photo_agent.py prompts/example_prompts.txt
```

---

### Все параметры CLI

```bash
python examples/run_photo_agent.py <файл_промптов> [опции]

Основные:
  --steps INT        Шагов сэмплирования         (default: 20)
  --cfg FLOAT        CFG scale                    (default: 7.0)
  --width INT        Ширина изображения           (default: 512)
  --height INT       Высота изображения           (default: 512)
  --seed INT         Seed, -1 = случайный         (default: -1)
  --sampler STR      Сэмплер                      (default: "DPM++ 2M")
  --scheduler STR    Scheduler                    (default: "Karras")
  --negative STR     Negative prompt
  --checkpoint STR   Имя чекпоинта (.safetensors)

HR Fix (upscale после генерации):
  --hr               Включить HR Fix
  --hr-scale FLOAT   Масштаб апскейла             (default: 2.0)
  --hr-steps INT     Шагов второго прохода        (default: 10)
  --denoise FLOAT    Denoising strength           (default: 0.4)
```

#### Примеры

```bash
# Простой запуск
python examples/run_photo_agent.py prompts/example_prompts.txt

# HD с HR Fix
python examples/run_photo_agent.py prompts/example_prompts.txt \
    --width 768 --height 768 --steps 30 \
    --hr --hr-scale 2.0 --denoise 0.35

# SDXL чекпоинт
python examples/run_photo_agent.py prompts/example_prompts.txt \
    --checkpoint juggernautXL_v9.safetensors \
    --width 1024 --height 1024 --steps 30 --cfg 5.0

# Портреты с LoRA (LoRA указывается прямо в промпте)
# В файле prompts.txt: <lora:beautifulDetailedEyes:0.7> woman portrait, 8k
python examples/run_photo_agent.py prompts/my_prompts.txt \
    --width 512 --height 768
```

---

### PhotoRunner — без LLM

Если Ollama не нужен — используй `PhotoRunner`: чистый Python-цикл, никакой LLM.

```python
from agents.photo_runner import PhotoRunner, GenerationSettings

runner = PhotoRunner(settings=GenerationSettings(steps=30, width=768, height=768))
results = runner.run_batch("prompts/example_prompts.txt")
```

Тот же CLI (`run_photo_agent.py`) автоматически использует `PhotoRunner` если не нужен агент — или импортируй напрямую.

---

### Переменные окружения — полный список

| Переменная | Default | Описание |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.1` | Модель Ollama для оркестрации |
| `OLLAMA_URL` | `http://localhost:11434` | URL Ollama |
| `PHOTO_BACKEND` | `forge` | Бэкенд генерации (`forge` или `automatic1111`) |
| `PHOTO_API_URL` | `http://127.0.0.1:7860` | URL Forge / A1111 |
| `PHOTO_OUTPUT_DIR` | `./outputs` | Папка для сохранения PNG |
| `ANTHROPIC_API_KEY` | — | Только для `BaseAgent`/Claude-агентов |

---

### Инструменты агента (доступны Ollama)

| Инструмент | Что делает |
|---|---|
| `load_prompts(file_path)` | Загружает промпты из файла в очередь |
| `get_next_prompt()` | Возвращает следующий промпт или `"DONE"` |
| `set_generation_settings(...)` | Меняет steps, cfg, size, HR Fix, checkpoint и др. |
| `list_models()` | Список доступных чекпоинтов в Forge |
| `generate_image(prompt)` | Генерирует изображение, сохраняет PNG, возвращает путь |

---

### Troubleshooting

**`Error: cannot reach forge @ http://127.0.0.1:7860`**
Forge не запущен или запущен без `--api`. Добавь флаг `--api` при запуске.

**`Connection refused` на Ollama**
Запусти `ollama serve` в отдельном терминале.

**Ollama не вызывает инструменты**
Смени модель на другую с поддержкой tool calling: `ollama pull qwen2.5`.

**Изображения сохраняются, но пустые**
Forge вернул пустой base64. Проверь логи Forge — скорее всего закончилась VRAM.

**Долгая генерация без ответа**
Таймаут по умолчанию 600 сек (для SDXL/FLUX). Для SD 1.5 можно уменьшить в `photo_agent.py`: `_GENERATION_TIMEOUT = 120`.

---

## BaseAgent — Claude (Anthropic API)

Для агентов с Claude (не локальная LLM):

```python
from agents.base import BaseAgent, AgentConfig, tool

class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__(config=AgentConfig(system_prompt="You are ..."))

    @tool(description="...", input_schema={...})
    def my_tool(self, arg: str) -> str:
        return "result"
```

Требует `ANTHROPIC_API_KEY` в `.env`.

---

## OllamaAgent — локальная LLM

Базовый класс для любых агентов на Ollama. Идентичный интерфейс с `BaseAgent`:

```python
from agents.ollama_base import OllamaAgent, OllamaConfig, tool

class MyLocalAgent(OllamaAgent):
    def __init__(self):
        super().__init__(config=OllamaConfig(
            model="llama3.1",
            system_prompt="You are ...",
        ))

    @tool(description="...", input_schema={...})
    def my_tool(self, x: int) -> str:
        return str(x * 2)

agent = MyLocalAgent()
print(agent.run("вычисли 21 * 2"))
```

---

## YoutuberAgent

Автоматизация YouTube через Playwright (браузер). Требует `YOUTUBE_EMAIL`, `YOUTUBE_PASSWORD` и установленного Chromium:

```bash
playwright install chromium
python -m examples.youtuber_run
```

Подробнее — в `dependencies.txt`.
