# Local AI Photo Agent — Plan

## Цель

Агент, который читает промпты из файла по очереди, отправляет каждый в локальный
AI-генератор изображений (Automatic1111 / ComfyUI), ждёт результата и переходит к
следующему.

---

## Поддерживаемые бэкенды

| Бэкенд | API | По умолчанию |
|---|---|---|
| **Automatic1111 SDUI** | REST `POST /sdapi/v1/txt2img` | `http://127.0.0.1:7860` |
| **ComfyUI** | REST + WebSocket | `http://127.0.0.1:8188` |

Выбор бэкенда задаётся через переменную окружения `PHOTO_BACKEND`.

---

## Архитектура

```
PhotoAgent (agents/photo_agent.py)
│
├── Инструменты (tools) — вызываются Claude:
│   ├── load_prompts(file_path)          — читает промпты из файла
│   ├── get_next_prompt()                — возвращает следующий промпт из очереди
│   ├── set_generation_settings(...)     — применяет настройки (model, steps, cfg, …)
│   ├── generate_image(prompt)           — запускает генерацию, ждёт результата
│   └── save_result(image_b64, path)     — сохраняет изображение на диск
│
└── Основной цикл (run_batch):
    for prompt in prompts → generate_image → save → next
```

---

## Структура файлов

```
agents-and-vibecode/
├── agents/
│   ├── base.py              (существующий)
│   ├── youtuber.py          (существующий)
│   └── photo_agent.py       ← НОВЫЙ
├── examples/
│   └── run_photo_agent.py   ← пример запуска
├── prompts/
│   └── example_prompts.txt  ← файл с промптами (по одному на строку)
├── outputs/                 ← куда сохраняются изображения
│   └── .gitkeep
└── LOCAL_AI_PHOTO_AGENT_PLAN.md
```

---

## Шаги реализации

### Шаг 1 — Файл промптов

`prompts/example_prompts.txt` — обычный текстовый файл, по одному промпту на строку.
Пустые строки и строки начинающиеся с `#` игнорируются.

```
# Пример файла промптов
a fantasy landscape at sunset, oil painting style, 8k
portrait of a cyberpunk samurai, neon lights, rain, ultra detailed
cute robot drinking coffee, studio lighting, photorealistic
```

### Шаг 2 — Настройки генерации

Хранятся в словаре `self._settings` внутри агента.
Параметры соответствуют API Automatic1111:

```python
{
    "steps": 25,
    "cfg_scale": 7.0,
    "width": 512,
    "height": 512,
    "sampler_name": "DPM++ 2M Karras",
    "model": None,          # если None — использует текущую модель
    "negative_prompt": "ugly, blurry, low quality",
    "seed": -1,             # -1 = случайный
    "batch_size": 1,
}
```

### Шаг 3 — Инструмент `generate_image`

Automatic1111 (основной путь):
1. `POST /sdapi/v1/txt2img` → получаем JSON с `images: [base64]`
2. Декодируем base64 → PNG
3. Сохраняем в `outputs/`

ComfyUI (альтернативный путь):
1. Строим `workflow` (JSON) с нужными нодами
2. `POST /prompt` → получаем `prompt_id`
3. Подключаемся к WebSocket `/ws?clientId=…`
4. Ждём сообщение `execution_complete` для нашего `prompt_id`
5. `GET /history/{prompt_id}` → URL файла
6. `GET /view?filename=…` → скачиваем изображение

### Шаг 4 — Основной цикл агента

Claude получает системный промпт с инструкцией:

```
1. load_prompts(file_path)
2. Для каждого промпта:
   a. get_next_prompt()
   b. generate_image(prompt)
   c. Сообщи результат (путь к файлу, seed, время)
3. Когда промпты закончились — завершить работу.
```

### Шаг 5 — Переменные окружения (`.env`)

```
ANTHROPIC_API_KEY=sk-ant-...
PHOTO_BACKEND=automatic1111         # или comfyui
PHOTO_API_URL=http://127.0.0.1:7860 # URL локального сервера
PHOTO_OUTPUT_DIR=./outputs
```

### Шаг 6 — Пример запуска

```bash
# Запустить Automatic1111 заранее:
# python launch.py --api --listen

# Запустить агента:
python examples/run_photo_agent.py prompts/example_prompts.txt
```

---

## Обработка ошибок

| Ситуация | Поведение |
|---|---|
| Бэкенд недоступен | Агент повторяет 3 раза с паузой 10 сек, затем сообщает об ошибке |
| Промпт пустой | Пропускается, переходим к следующему |
| Ошибка декодирования base64 | Логируем, продолжаем |
| Все промпты обработаны | Агент выводит итоговую сводку и завершается |

---

## Расширения (не в первой версии)

- **img2img** — передать начальное изображение вместе с промптом
- **ControlNet** — дополнительные параметры управления позой/глубиной
- **Upscale** после генерации через `/sdapi/v1/extra-single-image`
- **Параллельная генерация** — несколько запросов одновременно (если GPU позволяет)
- **Web UI** для управления очередью в реальном времени
