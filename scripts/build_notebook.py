"""Mirror complete implementation into the graded notebook; preserve no stale outputs.

The %%writefile cells contain readable, complete Python source, not just imports.
The notebook can rebuild the package and fictional corpus in an empty folder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build():
    cells = []

    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text.strip()))

    def code(text):
        cells.append(nbf.v4.new_code_cell(text.strip()))

    md("""
# Архив последнего света
## Лабораторная работа «Агентский театр»

**Автор:** заполните ФИО перед сдачей. **Реализация и проверка:** 03–04.11.2026.

В затопляемом архиве три героя решают, какую память спасать и у кого есть право на забвение.
Вера говорит образами и защищает личные свидетельства, Марк проверяет инженерные допущения,
локальный хранитель Эхо учится отличать сохранённую запись от согласия человека.
Документы дают героям основания менять позицию; финальное решение имеет цену.

Этот ноутбук содержит **полный код**, объяснения, вымышленный корпус, эксперимент и результаты.
Исходники в `theatre/` зеркалируются в ячейки, чтобы их было удобно тестировать отдельно.
Ячейки `%%writefile` сохраняют опубликованный здесь код в пакет перед запуском.
Не редактируйте одновременно обе копии: после изменения исходников выполните
`python scripts/build_notebook.py`.

По умолчанию выполняется **demo** — детерминированная сценарная фикстура для проверки механики.
Её текст подготовлен заранее и не выдаётся за результат работы LLM.
В режиме **cloud** все роли используют Yandex Cloud. Для Марка можно указать вторую
облачную модель в `MARK_MODEL`. Режим **live** поручает Эхо локальной Qwen, а
**local** запускает все роли на локальной модели.
""")
    md("""
### 1. Окружение и режим эксперимента

В PowerShell из рабочей папки:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\\Scripts\\python.exe -e ".[notebook,dev]"
# Только для локальной Qwen на RTX 5070: PyTorch с поддержкой Blackwell (CUDA 12.8+).
uv pip install --python .venv\\Scripts\\python.exe torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\\Scripts\\python.exe -e ".[local,quantized]"
```

Выберите ядро `.venv`. Заполните `.env` по `.env.sample`; ключи не выводятся.
Первое локальное выполнение скачивает около 8 ГБ весов Qwen3-4B-Instruct-2507.
Для RTX 5070 12 ГБ в `.env` задано `LOCAL_QUANTIZATION=nf4`: веса хранятся в четырёх
битах через bitsandbytes, вычисления — BF16. `none` выбирает обычные веса без квантизации.
Переключение на `MODE = "cloud"` или `MODE = "live"` выполняет платные запросы вашего API. Количество действий
и реплик задаётся явно; скрытого перехода на фикстуру при ошибке нет.
""")
    code("""from pathlib import Path
import os
import sys
import json
from dotenv import load_dotenv
from IPython.display import display, Markdown, HTML, Audio

ROOT = Path.cwd()
(ROOT / "theatre").mkdir(exist_ok=True)
(ROOT / "data" / "archive").mkdir(parents=True, exist_ok=True)
load_dotenv(ROOT / ".env", override=False)
MODE = os.getenv("THEATRE_MODE", "demo")  # demo | cloud | live | local
ACTS = 3
TURNS_PER_ACT = 4
SEED = 42
SYNTHESIZE_AUDIO = False
PIPER_MODEL_DIR = os.getenv("PIPER_MODEL_DIR")
print(f"Режим: {MODE}; действий: {ACTS}; реплик в действии: {TURNS_PER_ACT}")""")

    descriptions = {
        "__init__.py": "### 2. Пакет и воспроизводимость\n\nИмпорт пакета не обращается к API и не скачивает веса.",
        "providers.py": "### 3. Настоящие модельные провайдеры\n\nОблачный адаптер поддерживает Responses и Chat Completions, ограничивает повторы и не печатает ключи. Локальный адаптер загружает открытые веса Hugging Face, применяет chat template и декодирует только новые токены. Qwen3 работает с отключённым thinking; инференс выполняется на CUDA при её наличии. Seed не делает облачные ответы воспроизводимыми.",
        "tools.py": "### 4. Инструменты и границы доверия\n\nЛексический поиск по UTF-8 файлам возвращает фрагменты и стабильные ID D1–D5. Чтение ограничено каталогом корпуса; обход пути и символические ссылки запрещены. Смета — прозрачный численный инструмент вымышленного мира. Текст документа передаётся модели как недоверенные данные.",
        "prompts.py": "### 5. Персонажи, тайны и few-shot\n\nКаждый актёр получает только собственную тайну, цель и пример голоса. Служебные роли разделены: драматург строит общий план, режиссёр выбирает следующего участника, рассказчик ставит сцену. Персонажи видят развитие диалога, но не чужие системные промпты.",
        "engine.py": "### 6. Оркестрация и журнал\n\nСначала LLM строит план действий. На каждом шаге отдельный режиссёр предлагает говорящего; правила запрещают немедленный повтор, ограничивают выпадение героя и обеспечивают участие всех троих в каждом действии. До первой реплики актёр должен получить ответ поиска/чтения. На ход разрешено до двух вызовов инструментов, на неверный JSON, ссылку или дословный повтор — одна попытка исправления. Недостающий жест дописывает настоящий рассказчик. Ссылки допускаются только на источники, реально полученные этим актёром. Это проверка происхождения ссылки, а не автоматическая проверка истинности художественной интерпретации. Счётчики точных повторов также не заменяют литературную оценку.",
        "audio.py": "### 7. Озвучка разными голосами\n\nElevenLabs озвучивает полученный диалог: каждый персонаж и рассказчик получают отдельный существующий голос. Сохраняются MP3 по репликам, манифест, плейлист и общий MP3 через FFmpeg. Повторный запуск использует проверенный кеш. Без ключа звуковой результат не имитируется.",
        "__main__.py": "### 8. Тот же эксперимент из командной строки\n\nCLI использует те же провайдеры, корпус и движок, что и этот ноутбук.",
    }
    for name in descriptions:
        md(descriptions[name])
        source = (ROOT / "theatre" / name).read_text(encoding="utf-8")
        code(f"%%writefile theatre/{name}\n" + source)

    md("""
### 9. Корпус свидетельств

Все документы ниже придуманы для пьесы. Это художественный архив, а не реальные исторические источники.
Противоречие между ранней сметой, техническим приложением, письмом и протоколом создаёт повод
для действий агентов. Код поиска не подставляет заранее выбранный документ в ответ.
""")
    for path in sorted((ROOT / "data" / "archive").glob("*.md")):
        code(f"%%writefile data/archive/{path.name}\n" + path.read_text(encoding="utf-8"))

    md("""
### 10. Запуск

`demo` полностью проверяет путь план → режиссёр → инструмент → реплика → экспорт без ключей.
Для настоящей пьесы выберите `cloud` (Yandex Cloud) или `live` (Yandex Cloud и Qwen).
В журнал попадут фактические идентификаторы моделей.
`local` полезен для отдельного эксперимента, но все роли в нём исполняет одна модель.
""")
    code("""import importlib
importlib.invalidate_caches()
# Повторный запуск ячеек использует только что записанный код.
for module_name in list(sys.modules):
    if module_name == "theatre" or module_name.startswith("theatre."):
        del sys.modules[module_name]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from theatre.__main__ import build_providers
from theatre.engine import run_play, save_result
from theatre.tools import ToolBox

tools = ToolBox(ROOT / "data" / "archive")
providers = build_providers(MODE, SEED)
print(json.dumps({role: p.name for role, p in providers.items()}, ensure_ascii=False, indent=2))
result = run_play(providers, tools=tools, turns_per_act=TURNS_PER_ACT,
                  acts=ACTS, mode=MODE, seed=SEED)
output_dir = ROOT / "outputs" / MODE
paths = save_result(result, output_dir)
display(Markdown(f"**{result['metadata']['warning']}**"))
display(Markdown(Path(paths["markdown"]).read_text(encoding="utf-8")))""")

    md("""
### 11. Метрики и проверяемые свидетельства

Количество реплик и ссылок не оценивает художественное качество автоматически.
Таблица и журнал позволяют проверить участие героев, реальные запросы инструментов
и обоснование каждого выбора говорящего. Осмысленность, выразительность и изменение
позиций нужно оценить при чтении настоящего прогона.
""")
    code("""print(json.dumps(result["stats"], ensure_ascii=False, indent=2))
selection_events = [e for e in result["events"] if e["type"] == "selection"]
tool_events = [e for e in result["events"] if e["type"] == "tool"]
display(Markdown("| Действие | Говорящий | Основание |\\n|---|---|---|\\n" + "\\n".join(
    f"| {e['act']} | {e['speaker']} | {e.get('nomination', {}).get('reason', '')} / {e['guard']} |"
    for e in selection_events)))
print("Пример реального вызова инструмента в текущем режиме:")
print(json.dumps(tool_events[0] if tool_events else {"notice": "В этом прогоне инструменты не использовались"},
                 ensure_ascii=False, indent=2))
assert result["stats"]["actor_turns"] == ACTS * TURNS_PER_ACT
for act in range(1, ACTS + 1):
    assert {t["speaker"] for t in result["turns"] if t["act"] == act} >= {"vera", "mark", "echo"}
print("Артефакты:", json.dumps(paths, ensure_ascii=False, indent=2))""")
    code("""# Проверяем происхождение результата отдельно от числа реплик.
print(json.dumps(result["metadata"], ensure_ascii=False, indent=2))
smoke_path = ROOT / "outputs" / "local_smoke_latest.json"
if not smoke_path.exists():
    smoke_path = ROOT / "outputs" / "local_smoke_nf4.json"
if smoke_path.exists():
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    display(Markdown("**Отдельный настоящий тест основной локальной HF-модели 4B; точность указана в quantization:**"))
    print(json.dumps(smoke, ensure_ascii=False, indent=2))
else:
    print("Отдельный локальный прогон ещё не сохранён: python scripts/check_local_model.py")""")

    md("""
### 12. Сохранённый настоящий эксперимент

Если в репозитории есть `outputs/local/play.json` или `outputs/local_4b/play.json`, ниже показываются **ранее выполненные**
локальные эксперименты с реальными весами. Эта ячейка не генерирует новую пьесу и не использует
облачные квоты. В local все роли играются одной моделью; для разных LLM нужен live.
Это разведочные проверки: между запусками менялись и модель, и промпты с валидаторами.
Поэтому таблица не является контролируемым сравнением моделей при прочих равных.
""")
    code("""saved_local_paths = [ROOT / "outputs" / folder / "play.json" for folder in ("local", "local_4b")]
found_saved_experiment = False
comparison = ["| Модель | Режим весов | Реплики | Уникальных текстов | Точные повторы | Успешные инструменты | Секунды |",
              "|---|---|---:|---:|---:|---:|---:|"]
for saved_local_path in saved_local_paths:
    if saved_local_path.exists():
        found_saved_experiment = True
        saved_local = json.loads(saved_local_path.read_text(encoding="utf-8"))
        assert not saved_local["metadata"]["fixture"]
        display(Markdown(f"**Сохранённый настоящий эксперимент {saved_local_path.parent.name}; не demo и не новая генерация этой ячейки.**"))
        review_path = saved_local_path.parent / "quality_review.json"
        if review_path.exists():
            print(review_path.read_text(encoding="utf-8"))
        print(json.dumps(saved_local["metadata"], ensure_ascii=False, indent=2))
        print(json.dumps(saved_local["stats"], ensure_ascii=False, indent=2))
        stats = saved_local["stats"]
        quality = stats.get("quality", {})
        precision = saved_local["metadata"].get("local_models", {}).get("echo", {}).get("quantization", "не записан")
        comparison.append(f"| {saved_local['metadata']['models']['echo']} | {precision} | {stats['actor_turns']} | "
                          f"{quality.get('unique_actor_texts', '—')} | {quality.get('duplicate_actor_turns', '—')} | "
                          f"{stats['successful_tool_calls']} | {saved_local['metadata']['elapsed_seconds']} |")
        if saved_local_path.parent.name == "local_4b":
            display(Markdown(saved_local_path.with_suffix(".md").read_text(encoding="utf-8")))
        else:
            display(Markdown("Базовый эксперимент 1.7B не прошёл содержательную проверку; полный текст — в outputs/local/play.md."))
if found_saved_experiment:
    display(Markdown("\\n".join(comparison)))
if not found_saved_experiment:
    print("Полный локальный результат пока отсутствует. Для запуска: python -m theatre --mode local")""")
    code("""diagnostic_path = ROOT / "outputs" / "diagnostics" / "local_4b_nf4_duplicate_2026-11-04.review.json"
if diagnostic_path.exists():
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    display(Markdown("**Настоящий NF4-эксперимент остановлен проверкой повторов. Это незавершённая пьеса.**"))
    print(json.dumps(diagnostic, ensure_ascii=False, indent=2))
else:
    print("Архивный диагностический отчёт NF4 отсутствует в этой копии проекта.")""")
    md("""
### 13. Синтез речи и прослушивание

Сохранённый облачный прогон можно прочитать без новых запросов.
Для новой озвучки включите `SYNTHESIZE_AUDIO` и задайте `PIPER_MODEL_DIR` для локальных
голосов Piper или заполните ключ и ID голосов ElevenLabs в `.env`.
""")
    code("""published = ROOT / "outputs" / "cloud"
if (published / "play.md").is_file():
    display(Markdown((published / "play.md").read_text(encoding="utf-8")))
    if (published / "audio_piper" / "play.wav").is_file():
        display(Audio(filename=str(published / "audio_piper" / "play.wav")))

if SYNTHESIZE_AUDIO:
    if PIPER_MODEL_DIR:
        from theatre.audio import render_local_audio
        audio_result = render_local_audio(result, output_dir / "audio_piper", model_dir=PIPER_MODEL_DIR)
    else:
        from theatre.audio import render_audio
        audio_result = render_audio(result, output_dir / "audio")
    print(json.dumps(audio_result, ensure_ascii=False, indent=2))
    audio_path = audio_result.get("audio")
    if audio_path and Path(audio_path).exists():
        display(Audio(filename=str(audio_path)))
else:
    print("Озвучка выключена. Аудиофайлы не имитируются.")""")
    md("""
### 14. Выводы и ограничения

Разделение планировщика, режиссёра и актёров делает управление драматургией наблюдаемым.
Различимые голоса требуют собственных целей и речевых примеров; смена имени в одном промпте
этого не обеспечивает. Файловый поиск полезен, когда документы меняют решение, а проверка ID
не подменяет проверку содержания. Мягкое решение режиссёра дополняется явными правилами участия.

`demo` подтверждает интеграцию и воспроизводимость механики, но не качество LLM.
Результат `local_smoke_nf4.json` подтверждает только отдельный настоящий ответ локальной модели.
Для завершения экспериментальной части нужны полный облачный или гибридный прогон, чтение пьесы,
настоящая озвучка и личные выводы автора в `REPORT.md`.

Источники: [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507),
[Yandex AI Studio](https://aistudio.yandex.ru/en/docs/ai-studio/operations/generation/create-prompt),
[ElevenLabs TTS](https://elevenlabs.io/docs/api-reference/text-to-speech/convert).
""")

    for index, cell in enumerate(cells):
        cell.id = f"theatre-{index:03}"
    return nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python (Agent Theatre .venv)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    target = ROOT / "LLM_Theatre.ipynb"
    expected = build()
    if args.check:
        existing = nbf.read(target, as_version=4)
        if [(c.cell_type, c.source) for c in existing.cells] != [
            (c.cell_type, c.source) for c in expected.cells
        ]:
            print("Notebook source drift: run python scripts/build_notebook.py", file=sys.stderr)
            raise SystemExit(1)
        print("Notebook mirrors current source and corpus.")
    else:
        nbf.write(expected, target)
        print(f"Built {len(expected.cells)} cells: {target}")


if __name__ == "__main__":
    main()
