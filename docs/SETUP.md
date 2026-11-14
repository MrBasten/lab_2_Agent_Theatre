# Подключение Yandex Cloud и ElevenLabs

Инструкция проверена по официальной документации 03.11.2026. Рабочая папка:
`C:\Users\MrBasten\Desktop\Agent_Theatre`.

## 1. Что уже подготовлено

В `.env` оставлены пустые поля для доступов. Этот файл исключён из Git. В чат ключи присылать не нужно.
`.env.sample` — публичный образец без секретов. Основной локальный персонаж использует
`Qwen/Qwen3-4B-Instruct-2507`: около 8 ГБ скачиваемых весов, облачный ключ не нужен.
Модель уже загружена на этом компьютере. Для RTX 5070 12 ГБ задано
`LOCAL_QUANTIZATION=nf4`: веса квантизуются локально при загрузке, вычисления выполняются в BF16.
Нужен extra `quantized` с bitsandbytes. Режим `none` оставляет обычные веса;
в этом Windows-окружении он прошёл короткий тест, но на длинном контексте ушёл в общую RAM.
[Описание 4-битной загрузки](https://huggingface.co/docs/transformers/main/quantization/bitsandbytes).

Откройте PowerShell в рабочей папке. Проверка заполнения без сетевых запросов:

```powershell
.\.venv\Scripts\python.exe scripts\check_services.py
```

Она показывает только SET/UNSET, наличие зависимостей и FFmpeg. Если вы запускаете проект
на другом компьютере, сначала выполните команды установки из раздела 5.

## 2. Yandex Cloud / AI Studio

1. Войдите в [AI Studio](https://aistudio.yandex.cloud/) с Яндекс ID. Выберите или создайте облако и каталог; подключите платёжный аккаунт, если этого требует интерфейс. API в проекте работает по потреблению: подписка в веб-интерфейсе сама по себе не оплачивает эти запросы. [Документация базового запроса](https://aistudio.yandex.ru/ru/docs/ai-studio/operations/generation/create-prompt).
2. В верхней части AI Studio нажмите **Создать API-ключ**. Укажите понятное описание `agent-theatre` и срок действия. При создании с новым сервисным аккаунтом платформа назначает доступы автоматически. Сохраните **секретное значение**, показанное один раз, а не ID ключа. Для создания ключа у вас должна быть роль администратора каталога. [Создание API-ключа](https://aistudio.yandex.ru/en/docs/ai-studio/operations/get-api-key).
3. Скопируйте **ID каталога**: в AI Studio наведите указатель на название каталога вверху и используйте копирование. Это не ID облака, платёжного аккаунта или сервисного аккаунта. [Аутентификация и каталог](https://aistudio.yandex.ru/ru/docs/ai-studio/api-ref/authentication).
4. Откройте `.env` в редакторе и заполните:

```dotenv
folder_id=ВАШ_ID_КАТАЛОГА
api_key=ВАША_СЕКРЕТНАЯ_ЧАСТЬ_API_КЛЮЧА
LLM_BASE_URL=https://ai.api.cloud.yandex.net/v1
LLM_API_KIND=responses
LLM_MODEL=
```

При пустом `LLM_MODEL` используется Alice AI LLM Flash (`gpt://<folder_id>/aliceai-llm-flash`).
Для другой модели укажите имя или полный URI из Model Gallery в `LLM_MODEL`.
Для Марка можно отдельно указать, например, `MARK_MODEL=yandexgpt-5.1`.
В режиме `cloud` Эхо тоже исполняется в Yandex Cloud.
В гибридном режиме `live` Эхо исполняется локальной Qwen.

5. Если используете **существующий** сервисный аккаунт, проверьте его роли и scope ключа. Для используемого Responses API нужен доступ к моделям; официальный пример указывает роль `ai.languageModels.user` и scope **`yc.ai.foundationModels.execute`**. При 403 проверьте также требования `ai.assistants.editor` для используемой операции Responses в разделе управления доступом. Scope `yc.ai.languageModels.execute`, достаточный для другого Text Generation API, не следует автоматически переносить на Responses. [Базовый запрос](https://aistudio.yandex.ru/ru/docs/ai-studio/operations/generation/create-prompt), [требования к API-операциям](https://yandex.cloud/en/docs/security/standard-ai/all).
6. Сохраните `.env`. Запустите сначала один короткий запрос:

```powershell
.\.venv\Scripts\python.exe scripts\check_services.py --cloud
```

Короткий облачный прогон и полный эксперимент без локальной модели:

```powershell
python -m theatre --mode cloud --acts 1 --turns 3 --output outputs/cloud/smoke
python -m theatre --mode cloud
```

Для режима `cloud` достаточно Python с установленным `python-dotenv`. Полный результат
сохраняется в `outputs/cloud/`. Для озвучки передайте `outputs/cloud/play.json` в `--audio-from`.

### Озвучка через Piper без оплаты за реплики

Установите Piper и один раз скачайте четыре русских голоса:

```powershell
python -m pip install piper-tts
python -m piper.download_voices --data-dir voices ru_RU-irina-medium ru_RU-dmitri-medium ru_RU-ruslan-medium ru_RU-denis-medium
python -m theatre --audio-from outputs/cloud/play.json --piper-model-dir voices
```

После скачивания синтез выполняется на CPU. Используются открытые голоса из каталога
[Piper voices](https://huggingface.co/rhasspy/piper-voices/tree/main/ru/ru_RU).
Ключ ElevenLabs нужен только для облачного варианта озвучки.

Для короткого гибридного прогона с локальной Qwen используйте:

```powershell
.\.venv\Scripts\python.exe -m theatre --mode live --acts 1 --turns 3 --output outputs/live/smoke
```

И полный эксперимент для отчёта:

```powershell
.\.venv\Scripts\python.exe -m theatre --mode live
```

Результаты: `outputs/live/play.md`, `play.html`, `play.json`, `stats.json`, `events.jsonl`.
Последний файл пишется по мере выполнения и помогает найти этап ошибки. Журнал содержит
выбор говорящих и найденные документы. Повтор команды генерации создаёт новую пьесу и новые
платные запросы; ошибки не заменяются заранее написанными репликами.

## 3. ElevenLabs: ключ и четыре голоса

1. Создайте аккаунт на [ElevenLabs](https://elevenlabs.io/). В интерфейсе откройте **Developers → API Keys → Create API Key**. Назовите ключ `agent-theatre`, разрешите **Text to Speech** и **Voices: Read**. При желании задайте лимит использования. Скопируйте полный секрет сразу после создания. [Официальная инструкция](https://elevenlabs.io/docs/help-center/technical/how-do-i-authorize-myself-using-an-api-key).
2. В `.env` заполните `ELEVENLABS_API_KEY`. У аккаунта должны быть доступ к выбранным голосам и достаточная квота. Тариф и доступность смотрите в своём аккаунте; покупать определённый тариф заранее для этой инструкции не требуется.
3. В разделе Voices прослушайте варианты на русском и добавьте понравившиеся голоса в свою библиотеку. Для пьесы нужны четыре **разных** существующих голоса: Вера, Марк, Эхо и рассказчик. Клонирование голоса не используется.
4. Выведите доступные вашему ключу ID:

```powershell
.\.venv\Scripts\python.exe scripts\check_services.py --voices
```

Скрипт читает официальный endpoint `GET /v2/voices`; показывает имя, `voice_id` и метки.
Выбирайте голоса после прослушивания, а не по одному имени. [API списка голосов](https://elevenlabs.io/docs/api-reference/voices/search).

5. Скопируйте выбранные значения в `.env`:

```dotenv
ELEVENLABS_API_KEY=ВАШ_СЕКРЕТНЫЙ_КЛЮЧ
ELEVENLABS_VOICE_VERA=ID_ГОЛОСА_ВЕРЫ
ELEVENLABS_VOICE_MARK=ID_ГОЛОСА_МАРКА
ELEVENLABS_VOICE_ECHO=ID_ГОЛОСА_ЭХО
ELEVENLABS_VOICE_NARRATOR=ID_ГОЛОСА_РАССКАЗЧИКА
```

6. Проверьте короткую озвучку Веры и рассказчика, прежде чем синтезировать всю пьесу:

```powershell
.\.venv\Scripts\python.exe scripts\check_services.py --audio-smoke
```

Она создаст настоящий звук в `outputs/audio_smoke/`. Для этого теста нужны только голоса
Веры и рассказчика. Для всей пьесы нужны все четыре.

7. Озвучьте **уже сохранённую** настоящую пьесу без повторных запросов к LLM:

```powershell
.\.venv\Scripts\python.exe -m theatre --audio-from outputs/live/play.json
```

Используется `eleven_multilingual_v2`. Готовый файл: `outputs/live/audio/play.mp3`;
там же сохраняются отдельные реплики, `manifest.json` и плейлист `play.m3u`.
FFmpeg уже найден на этом компьютере. При сбое озвучки повторите ту же команду:
завершённые реплики с совпадающей контрольной суммой будут использованы из кеша.
[Метод синтеза ElevenLabs](https://elevenlabs.io/docs/api-reference/text-to-speech/convert).

## 4. Типовые ошибки

| Симптом | Что проверить |
|---|---|
| Ключ всё ещё UNSET | Файл называется `.env`, находится рядом с README и сохранён; перезапустите ядро ноутбука после изменения переменных |
| YC 401 | Вставлена секретная часть ключа, срок действия не истёк |
| YC 403 | Роли, scope Responses, правильный каталог, доступ к модели |
| YC 404 | URI модели из Model Gallery, endpoint и `LLM_API_KIND=responses` |
| YC 429 | Лимиты, квота и платёжный аккаунт; автоматические повторы ограничены |
| ElevenLabs 401/403 | Ключ, права Text to Speech / Voices Read, доступ к выбранному голосу |
| ElevenLabs 429 | Квота/частота; завершённые аудиофайлы сохранены |
| Нет общего MP3 | Отдельные MP3 и M3U пригодны для прослушивания; проверьте `ffmpeg -version` |
| Ошибка CUDA | Используйте `.venv` и CUDA-сборку PyTorch из команды ниже |

## 5. Установка на другом компьютере

Команды выполняйте **последовательно**: одновременные установки в одну `.venv` могут
создать конфликт сборок. GPU-вариант для RTX 5070:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -e ".[notebook,dev]"
uv pip install --python .venv\Scripts\python.exe "torch==2.11.0+cu128" --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -e ".[local,quantized]"
.\.venv\Scripts\python.exe scripts\check_local_model.py
.\.venv\Scripts\python.exe -m pytest -q
```

Обычный учебный запуск без API и GPU:

```powershell
.\.venv\Scripts\python.exe -m theatre --mode demo
.\.venv\Scripts\python.exe scripts\execute_notebook.py
```

`execute_notebook.py` намеренно выполняет notebook в demo, чтобы проверка не расходовала квоты.
Для live откройте notebook, выберите ядро `.venv`, установите `MODE = "live"` в первой
кодовой ячейке и выполните все ячейки. Перед сдачей сохраните выводы настоящего прогона.
