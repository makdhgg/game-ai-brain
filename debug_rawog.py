# debug_raw.py
import json
import time
import ollama

from mainog import WorldLore, CrimeCase
from promtog import (
    STAGE1_PROMPT_TEMPLATE,
    SUSPECT_PROMPT_TEMPLATE,
    CONNECTIONS_PROMPT_TEMPLATE,
    VALIDATOR_PROMPT_TEMPLATE,
)


# ================= НАСТРОЙКИ =================
MODEL = "qwen3.5:9b"
NUM_GPU = 20
NUM_CTX = 16384
NUM_PREDICT = 2048
USE_JSON_FORMAT = True
STREAM = True
THINK = False
# =============================================


def g(obj, key, default=None):
    """Доступ к полю и у dict, и у pydantic-объекта (ollama 0.6)."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def show_local_models():
    print("=" * 70)
    print("📦 ЛОКАЛЬНЫЕ МОДЕЛИ OLLAMA")
    print("=" * 70)
    try:
        resp = ollama.list()
        models = g(resp, "models", []) or []
        for m in models:
            name = g(m, "model") or g(m, "name")
            size = g(m, "size", 0) or 0
            print(f"  • {name}  ({size / 1e9:.1f} GB)")
    except Exception as e:
        print(f"  ⚠️ Не удалось получить список моделей: {e}")
    print()


def raw_call(prompt: str, label: str, temperature: float, num_predict: int = None, num_ctx: int = None) -> str:
    if num_predict is None:
        num_predict = NUM_PREDICT
    if num_ctx is None:
        num_ctx = NUM_CTX

    print("=" * 70)
    print(f"🧪 {label} | модель: {MODEL}")
    print(f"   Промпт: {len(prompt)} символов | num_ctx={num_ctx} | num_predict={num_predict}")
    print("=" * 70)

    options = {
        "num_gpu": NUM_GPU,
        "num_ctx": num_ctx,
        "num_predict": num_predict,
        "temperature": temperature,
    }
    kwargs = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "options": options,
        "stream": STREAM,
    }
    if USE_JSON_FORMAT:
        kwargs["format"] = "json"
    if THINK is not None:
        kwargs["think"] = THINK

    text_parts, thinking_parts = [], []
    meta = {}
    start = time.time()

    print("\n--- НАЧАЛО СЫРОГО ВЫВОДА МОДЕЛИ ---\n")

    try:
        result = ollama.chat(**kwargs)
        if STREAM:
            for chunk in result:
                msg = g(chunk, "message")
                piece = g(msg, "content") or ""
                if piece:
                    print(piece, end="", flush=True)
                    text_parts.append(piece)
                th = g(msg, "thinking") or ""
                if th:
                    thinking_parts.append(th)
                if g(chunk, "done"):
                    meta = chunk
        else:
            msg = g(result, "message")
            piece = g(msg, "content") or ""
            print(piece)
            text_parts.append(piece)
            th = g(msg, "thinking") or ""
            if th:
                thinking_parts.append(th)
            meta = result

    except (TypeError, ollama.ResponseError) as e:
        print(f"\n⚠️ Ошибка вызова: {e}\n↻ Повтор без think и без format...\n")
        kwargs.pop("think", None)
        kwargs.pop("format", None)
        kwargs["stream"] = False
        result = ollama.chat(**kwargs)
        msg = g(result, "message")
        piece = g(msg, "content") or ""
        print(piece)
        text_parts.append(piece)
        meta = result

    elapsed = time.time() - start
    raw_text = "".join(text_parts)
    print("\n\n--- КОНЕЦ СЫРОГО ВЫВОДА ---\n")

    print("📊 ДИАГНОСТИКА:")
    print(f"   Время: {elapsed:.1f} сек")
    print(f"   Символов в ответе: {len(raw_text)}")
    done_reason = g(meta, "done_reason")
    prompt_tokens = g(meta, "prompt_eval_count")
    out_tokens = g(meta, "eval_count")
    eval_dur = g(meta, "eval_duration") or 0
    print(f"   done_reason: {done_reason}")
    print(f"   Токенов промпта: {prompt_tokens}")
    print(f"   Токенов ответа: {out_tokens}")
    if eval_dur and out_tokens:
        print(f"   Скорость: {out_tokens / (eval_dur / 1e9):.1f} tok/s")

    if done_reason == "length":
        print("   🔴 ОТВЕТ ОБРЕЗАН ПО ЛИМИТУ num_predict → JSON будет невалидным!")
    if prompt_tokens and out_tokens and (prompt_tokens + out_tokens) >= num_ctx:
        print("   🔴 Упёрлись в num_ctx → контекст мог быть обрезан!")
    if thinking_parts:
        print(f"   🟡 Модель писала 'thinking' ({sum(len(t) for t in thinking_parts)} символов)")

    try:
        data = json.loads(raw_text)
        print("   ✅ JSON валиден")
        if isinstance(data, dict):
            print(f"   Ключи верхнего уровня: {list(data.keys())}")
            if "full_name" in data:
                print(f"   Имя подозреваемого: {data['full_name']}")
                if "is_guilty" in data:
                    print(f"   is_guilty: {data['is_guilty']}")
            if "suspect_connections" in data and isinstance(data["suspect_connections"], list):
                print(f"   Кол-во связей: {len(data['suspect_connections'])}")
            if "is_consistent" in data:
                print(f"   is_consistent: {data['is_consistent']}")
                issues = data.get('issues', [])
                print(f"   Кол-во проблем: {len(issues)}")
                corrected = data.get('corrected_suspects', [])
                print(f"   Кол-во исправленных подозреваемых: {len(corrected)}")
            for nested in ("location", "timeline", "victim_info"):
                if nested in data and isinstance(data[nested], dict):
                    print(f"   Ключи {nested}: {list(data[nested].keys())}")
    except json.JSONDecodeError as e:
        print(f"   ❌ JSON НЕВАЛИДЕН: {e}")
        print(f"   Хвост ответа (последние 200 символов): ...{raw_text[-200:]}")

    fname = f"raw_{label}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(raw_text)
    print(f"   💾 Сырой вывод сохранён в {fname}\n")
    return raw_text


if __name__ == "__main__":
    show_local_models()

    world = WorldLore(
        universe_name="Нуарный Детройт 1940-х",
        era="1947 год, послевоенная Америка",
        atmosphere="мрачный нуар, дождливые улицы, джаз из баров, сигаретный дым",
        tech_level="технологии 1940-х: печатные машинки, дисковые телефоны, хромовые автомобили",
        social_norms="влияние мафии, коррупция в полиции, жёсткие социальные роли",
        custom_details="Мафия контролирует профсоюзы. Частные детективы — последние честные люди.",
    )

    # ЭТАП 1: Генерация преступления
    prompt1 = STAGE1_PROMPT_TEMPLATE.format(world_lore=world.to_prompt_text())
    raw1 = raw_call(prompt1, "stage1_crime", temperature=0.9)

    # ЭТАП 2: Генерация одного подозреваемого с якорями фактов
    try:
        crime_data = json.loads(raw1)
        crime = CrimeCase.from_dict(crime_data)

        evidence_list = "\n".join([f"   - {ev}" for ev in crime.initial_evidence])

        prompt2 = SUSPECT_PROMPT_TEMPLATE.format(
            world_lore=world.to_prompt_text(),
            estimated_crime_time=crime.timeline.estimated_crime_time,
            discovery_time=crime.timeline.discovery_time,
            location_type=crime.location.type,
            location_description=crime.location.description,
            victim_description=crime.victim_info.description,
            victim_condition=crime.victim_info.condition,
            evidence_list=evidence_list,
            tech_level=world.tech_level,
            weather=crime.location.weather,
            crime_case=raw1,
            suspect_number=1,
            total_suspects=4,
            guilty_status="ВИНОВНЫЙ",
            previous_suspects="Пока нет. Это первый подозреваемый.",
            is_guilty_bool="true"
        )
        raw2 = raw_call(prompt2, "suspect_1_guilty", temperature=0.85, num_predict=3072)

        # ЭТАП 3 (опционально): проверка промпта валидатора
        # Раскомментируй, если хочешь проверить валидатор на сырых данных
        #
        # try:
        #     suspect_data = json.loads(raw2)
        #     full_case = {
        #         "crime": crime_data,
        #         "suspects": {
        #             "suspects": [suspect_data],
        #             "suspect_connections": []
        #         }
        #     }
        #     prompt3 = VALIDATOR_PROMPT_TEMPLATE.format(
        #         world_lore=world.to_prompt_text(),
        #         full_case=json.dumps(full_case, ensure_ascii=False, indent=2)
        #     )
        #     raw3 = raw_call(prompt3, "validator_test", temperature=0.2, num_predict=8192, num_ctx=32768)
        # except json.JSONDecodeError:
        #     print("⏭ Валидатор пропущен: подозреваемый вернул невалидный JSON.")

    except json.JSONDecodeError:
        print("⏭ Этап 2 пропущен: этап 1 вернул невалидный JSON.")