#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vlm.py — «судья» на визуально-языковой модели (VLM) для Autolabel Studio.

Смотрит на вырезку и отвечает, какой это класс из списка (или «ничего из этого»).
Два способа:
  ollama — локально через Ollama (https://ollama.com), бесплатно, без интернета после скачивания
  openai — любой OpenAI-совместимый API: российские агрегаторы (ProxyAPI, VseGPT, AITunnel, Polza,
           provod), OpenRouter, OpenAI и т.п.

Проверка из консоли:
  python vlm.py test crop.jpg --classes classes.yaml --backend ollama --model qwen3-vl:4b
  python vlm.py test crop.jpg --classes classes.yaml --backend openai --url https://api.proxyapi.ru/v1 \
        --model openai/gpt-4o-mini --key sk-...
  python vlm.py models --backend openai --url https://api.vsegpt.ru/v1 --key ...   список моделей
  python vlm.py pull qwen3-vl:4b                                                   скачать в Ollama
"""
import base64
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# ----------------------------------------------------------------------------- каталоги

# Локальные модели Ollama с картинками: (тег, скачать, видеопамять ~, описание)
# Размеры — со страниц ollama.com/library (октябрь 2026), видеопамять — примерная оценка.
LOCAL_MODELS = [
    ("moondream:1.8b",          "1.7 ГБ", "2 ГБ",   "крошечная, быстрая; классы различает слабо"),
    ("qwen3-vl:2b",             "1.9 ГБ", "3 ГБ",   "самая лёгкая из хороших; для CPU и слабых карт"),
    ("granite3.2-vision:2b",    "2.4 ГБ", "3 ГБ",   "IBM, лёгкая; больше для документов"),
    ("qwen2.5vl:3b",            "3.2 ГБ", "5 ГБ",   "проверенная, хорошо читает знаки и надписи"),
    ("qwen3-vl:4b",             "3.3 ГБ", "5 ГБ",   "★ рекомендуется для 6 ГБ видеопамяти"),
    ("gemma3:4b",               "3.4 ГБ", "5 ГБ",   "Google, хорошо понимает сцены"),
    ("llava:7b",                "4.7 ГБ", "6 ГБ",   "старая классика, слабее Qwen"),
    ("minicpm-v:8b",            "5.5 ГБ", "7 ГБ",   "сильная на мелких деталях"),
    ("qwen2.5vl:7b",            "6.0 ГБ", "8 ГБ",   "точнее 3b, нужна карта от 8 ГБ"),
    ("qwen3-vl:8b",             "6.2 ГБ", "8 ГБ",   "★ лучшая для карт 8–12 ГБ"),
    ("llama3.2-vision:11b",     "7.8 ГБ", "10 ГБ",  "Meta, неплохо, но медленнее Qwen"),
    ("llava:13b",               "8.0 ГБ", "10 ГБ",  "старая, крупнее"),
    ("gemma3:12b",              "8.2 ГБ", "12 ГБ",  "точная, нужна мощная карта"),
    ("mistral-small3.2:24b",    "15 ГБ",  "18 ГБ",  "Mistral, сильная, для 24 ГБ карт"),
    ("gemma3:27b",              "17 ГБ",  "20 ГБ",  "очень точная, для 24 ГБ карт"),
    ("qwen3-vl:30b",            "20 ГБ",  "24 ГБ",  "MoE, быстрая для своего размера"),
    ("qwen3-vl:32b",            "21 ГБ",  "24 ГБ",  "одна из лучших локальных"),
    ("qwen2.5vl:32b",           "21 ГБ",  "24 ГБ",  "очень точная"),
]

# Модели API с картинками и ПРИМЕРНЫЕ цены (USD за 1 млн токенов, вход / выход) — без наценки
# агрегатора. Имена — как в OpenRouter; у ProxyAPI/VseGPT и др. имя может отличаться.
# img_tok — сколько токенов примерно «стоит» одна вырезка 256–768 px у этой модели.
# Источники: OpenRouter /api/v1/models и обзоры цен (октябрь 2026). Обновить: кнопка «Цены с OpenRouter».
API_PRICES = [
    # id,                                  вход,  выход, img_tok, примечание
    ("openai/gpt-5-nano",                  0.05,  0.40,  400,  "очень дёшево, для простых случаев"),
    ("nvidia/nemotron-3.5-super-vl:free",  0.0,   0.0,   400,  "бесплатно (лимиты), качество среднее"),
    ("google/gemini-2.5-flash-lite",       0.10,  0.40,  260,  "★ дёшево и неплохо"),
    ("openai/gpt-4.1-nano",                0.10,  0.40,  400,  "дёшево"),
    ("openai/gpt-6-luna",                  0.10,  0.50,  400,  "новая дешёвая OpenAI"),
    ("anthropic/claude-haiku-5.5",         0.10,  0.50,  400,  "новая дешёвая Claude"),
    ("xiaomi/mimo-v2.6-flash",             0.14,  0.28,  400,  "дёшево"),
    ("nvidia/nemotron-3.5-super-vl",       0.15,  0.50,  400,  ""),
    ("minimax/minimax-m3.1-flash",         0.15,  0.60,  400,  ""),
    ("deepseek/deepseek-v4.1-flash-1017",  0.15,  0.60,  400,  ""),
    ("openai/gpt-4o-mini",                 0.15,  0.60,  8500, "картинка считается как ~8500 токенов — не так дёшево"),
    ("openai/gpt-5-mini",                  0.25,  2.00,  400,  "★ хороший баланс"),
    ("google/gemini-2.5-flash",            0.30,  2.50,  260,  "★ отлично видит мелкие детали"),
    ("moondream/moondream3-preview",       0.30,  2.50,  400,  "специалист по зрению"),
    ("openai/gpt-4.1-mini",                0.40,  1.60,  400,  ""),
    ("anthropic/claude-haiku-4.5",         1.00,  5.00,  350,  ""),
    ("google/gemini-2.5-pro",              1.25,  10.0,  260,  "дорого, очень точно"),
    ("mistralai/mistral-large-4-0",        1.36,  4.18,  400,  ""),
    ("openai/gpt-5.2",                     1.75,  14.0,  400,  ""),
    ("openai/gpt-4.1",                     2.00,  8.00,  400,  ""),
    ("openai/gpt-6.1-sol",                 2.00,  10.0,  400,  "флагман OpenAI"),
    ("anthropic/claude-sonnet-5.5",        2.00,  10.0,  400,  "флагман Claude"),
    ("x-ai/grok-4.7",                      2.00,  6.00,  400,  ""),
    ("openai/gpt-4o",                      2.50,  10.0,  255,  ""),
    ("anthropic/claude-sonnet-4.5",        3.00,  15.0,  350,  ""),
    ("anthropic/claude-opus-5.5",          4.00,  20.0,  400,  "очень дорого, избыточно для знаков"),
    ("qwen/qwen3.8-max-prime",             4.00,  12.0,  400,  ""),
    ("typesafe/jev-router",                None,  None,  400,  "роутер: сам выбирает модель, цена плавает"),
]
PROMPT_TOK, ANSWER_TOK = 300, 20   # текст запроса и короткий JSON-ответ


def cost_per_1000(price_in, price_out, img_tok=400):
    """Примерная стоимость проверки 1000 вырезок, USD (без наценки агрегатора)."""
    if price_in is None or price_out is None:
        return None
    return 1000 * ((PROMPT_TOK + img_tok) * price_in + ANSWER_TOK * price_out) / 1e6


def openrouter_prices(proxy=None):
    """Живой список моделей с картинками и ценами с OpenRouter (без ключа). -> [(id, in, out)] $/1M"""
    r = _session(proxy).get("https://openrouter.ai/api/v1/models", timeout=30)
    r.raise_for_status()
    out = []
    for m in r.json().get("data", []):
        arch = m.get("architecture") or {}
        mods = arch.get("input_modalities") or []
        if "image" not in mods and "image" not in str(arch.get("modality", "")).split("->")[0]:
            continue
        pr = m.get("pricing") or {}
        try:
            pi, po = float(pr.get("prompt", -1)) * 1e6, float(pr.get("completion", -1)) * 1e6
        except (TypeError, ValueError):
            continue
        if pi < 0 or po < 0:
            pi = po = None
        out.append((m.get("id"), pi, po))
    return sorted(out, key=lambda t: (t[1] is None, t[1] or 0, t[0]))


# Российские агрегаторы и другие OpenAI-совместимые API (оплата в рублях — у первых пяти)
API_PROVIDERS = [
    # имя,                 base_url,                         пример модели с картинками,      сайт
    ("ProxyAPI",           "https://api.proxyapi.ru/v1",     "openai/gpt-5-mini",             "https://proxyapi.ru"),
    ("VseGPT",             "https://api.vsegpt.ru/v1",       "openai/gpt-4o-mini",            "https://vsegpt.ru"),
    ("AITunnel",           "https://api.aitunnel.ru/v1",     "gpt-4o-mini",                   "https://aitunnel.ru"),
    ("AITunnel (зеркало)", "https://ru-api.aitunnel.ru/v1",  "gpt-4o-mini",                   "https://aitunnel.ru"),
    ("Polza.ai",           "https://polza.ai/api/v1",        "openai/gpt-4o-mini",            "https://polza.ai"),
    ("RouterAI",           "https://routerai.ru/api/v1",     "google/gemini-2.5-flash-lite",  "https://routerai.ru"),
    ("provod.ai",          "https://api.provod.ai/v1",       "gpt-4o-mini",                   "https://provod.ai"),
    ("OpenRouter",         "https://openrouter.ai/api/v1",   "google/gemini-2.5-flash-lite",  "https://openrouter.ai"),
    ("OpenAI",             "https://api.openai.com/v1",      "gpt-4o-mini",                   "https://platform.openai.com"),
    ("Ollama как API",     "http://localhost:11434/v1",      "qwen3-vl:4b",                   "https://ollama.com"),
    ("Своё",               "",                               "",                              ""),
]

# Модели с картинками, которые обычно есть у агрегаторов (точное имя смотрите кнопкой «Список моделей»)
API_MODEL_HINTS = [m[0] for m in API_PRICES]

OLLAMA_URL = "http://localhost:11434"


# ----------------------------------------------------------------------------- утилиты

def _session(proxy=None):
    import requests
    s = requests.Session()
    s.trust_env = False  # без системных прокси, если не задан явно
    if proxy:
        s.proxies = {"http": proxy, "https": proxy}
    return s


def encode_crop(bgr, min_side=384, max_side=768, quality=92):
    """BGR-вырезка -> base64 JPEG (мелкие увеличиваем — модели плохо видят 30-пиксельные знаки)."""
    import cv2
    h, w = bgr.shape[:2]
    k = 1.0
    if max(h, w) < min_side:
        k = min_side / max(h, w, 1)
    elif max(h, w) > max_side:
        k = max_side / max(h, w)
    if k != 1.0:
        bgr = cv2.resize(bgr, (max(1, int(w * k)), max(1, int(h * k))),
                         interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def build_prompt(classes, context=""):
    lines = []
    for i, c in enumerate(classes, 1):
        d = c.get("vlm") or (c.get("desc") or [""])[0] or (c.get("prompts") or [c["name"]])[0]
        lines.append(f"{i}. {c['name']} — {d}")
    ctx = f"Context: {context}\n" if context else ""
    return (
        "You are checking labels for an object detection dataset. The image is a crop around ONE candidate "
        "object cut from a video frame (it may be small, blurry, low-resolution or partly outside the crop).\n"
        "Objects can be miniature, toy or printed models (e.g. small model road signs on a test track) — "
        "judge by shape, colors and symbol, treat models exactly like the real thing.\n"
        f"{ctx}"
        "Which of these classes is the main object in the center of the crop?\n"
        + "\n".join(lines) +
        "\n0. none — clearly something else: background, a box/sticker/poster, a sign NOT in the list\n"
        "If the object matches a class description reasonably well, choose that class rather than none.\n"
        "Answer ONLY with JSON: {\"seen\": \"<what is in the center, max 8 words>\", "
        "\"class\": <number 0-%d>, \"name\": \"<class name or none>\", \"confidence\": <0..1>}" % len(classes)
    )


def parse_answer(text, n_cls, names=None):
    """-> (индекс класса 0..n-1 или -1 = «ничего», уверенность) или None"""
    if text is None:
        return None
    t = str(text).strip()
    m = re.search(r"\{.*?\}", t, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            q = max(0.0, min(1.0, float(d.get("confidence", d.get("conf", 0.8)) or 0.8)))
            nm = str(d.get("name", "")).strip().lower()
            if nm and names:  # имя надёжнее номера (модели путают нумерацию с 0 и с 1)
                for i, n in enumerate(names):
                    if nm == n.lower():
                        return i, q
                if nm in ("none", "nothing", "other", "unclear", "нет", "ничего"):
                    return -1, q
            k = int(float(d.get("class", d.get("id", -1))))
            if 0 <= k <= n_cls:
                return k - 1, q
        except Exception:
            pass
    if names:  # ответили именем
        low = t.lower()
        for i, n in enumerate(names):
            if n.lower() in low:
                return i, 0.7
        if re.search(r"\b(none|нет|ничего)\b", low):
            return -1, 0.7
    m = re.search(r"\b(\d{1,2})\b", t)
    if m and 0 <= int(m.group(1)) <= n_cls:
        return int(m.group(1)) - 1, 0.6
    return None


# ----------------------------------------------------------------------------- клиент

class VLMJudge:
    def __init__(self, backend, model, classes, url=None, key=None, proxy=None, timeout=90,
                 workers=4, context=""):
        self.backend = backend
        self.model = model
        self.classes = classes
        self.names = [c["name"] for c in classes]
        self.url = (url or (OLLAMA_URL if backend == "ollama" else "")).rstrip("/")
        self.key = key or os.environ.get("AUTOLABEL_VLM_KEY", "")
        self.s = _session(proxy)
        self.timeout = timeout
        self.workers = max(1, workers if backend != "ollama" else 1)
        self.prompt = build_prompt(classes, context)
        self.fails = 0
        self.calls = 0
        self.last_error = ""

    # ---- один запрос
    def _ask_ollama(self, b64):
        n = len(self.classes)
        schema = {"type": "object", "properties": {"class": {"type": "integer", "minimum": 0, "maximum": n},
                                                   "name": {"type": "string"},
                                                   "confidence": {"type": "number"}},
                  "required": ["class", "confidence"]}
        r = self.s.post(f"{self.url}/api/chat", timeout=self.timeout, json={
            "model": self.model, "stream": False, "format": schema, "keep_alive": "30m",
            "options": {"temperature": 0},
            "messages": [{"role": "user", "content": self.prompt, "images": [b64]}]})
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "")

    # Варианты параметров запроса. «Думающие» модели (gpt-5, o3, gpt-6, *-thinking…) тратят лимит токенов
    # на рассуждения и возвращают ПУСТОЙ ответ при max_tokens=60, а temperature=0 им вообще запрещён.
    VARIANTS = {
        "std": lambda: {"temperature": 0, "max_tokens": 300},
        "reasoning": lambda: {"max_completion_tokens": 4000, "reasoning_effort": "minimal"},
        "reasoning_low": lambda: {"max_completion_tokens": 4000, "reasoning_effort": "low"},
        "plain": lambda: {"max_tokens": 4000},
    }
    REASONING_RE = re.compile(r"(^|/)(o\d|gpt-5|gpt-6|.*think|.*reason|deepseek-r)", re.I)

    def _variants(self):
        if getattr(self, "variant", None):
            return [self.variant]
        first = ["reasoning", "reasoning_low", "plain", "std"] if self.REASONING_RE.search(self.model) else \
            ["std", "plain", "reasoning"]
        return first

    @staticmethod
    def _content(js):
        """Текст ответа: строка, список частей или пусто (+ причина)."""
        ch = (js.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        c = msg.get("content")
        if isinstance(c, list):
            c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
        return (c or "").strip(), ch.get("finish_reason")

    def _ask_openai(self, b64):
        if not self.url:
            raise RuntimeError("не задан адрес API (base_url)")
        h = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = f"Bearer {self.key}"
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": self.prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}]
        last = ""
        for v in self._variants():
            body = {"model": self.model, "messages": msgs, **self.VARIANTS[v]()}
            r = self.s.post(f"{self.url}/chat/completions", headers=h, timeout=self.timeout, json=body)
            if r.status_code in (401, 402, 403, 404, 429) or r.status_code >= 500:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            if r.status_code >= 400:  # 400 — скорее всего не тот набор параметров, пробуем следующий
                last = f"HTTP 400 ({v}): {r.text[:160]}"
                continue
            txt, fin = self._content(r.json())
            if txt:
                self.variant = v  # запоминаем рабочий вариант для этой модели
                return txt
            last = f"пустой ответ ({v}, finish_reason={fin})"
        raise RuntimeError(last or "пустой ответ")

    def ask(self, bgr):
        """-> (класс или -1, уверенность, сырой ответ) либо (None, 0, ошибка)"""
        b64 = encode_crop(bgr)
        err = ""
        for attempt in range(3):
            try:
                self.calls += 1
                txt = self._ask_ollama(b64) if self.backend == "ollama" else self._ask_openai(b64)
                res = parse_answer(txt, len(self.classes), self.names)
                if res is not None:
                    self.fails = 0
                    return res[0], res[1], txt
                err = f"непонятный ответ: {str(txt)[:120]}"
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:200]}"
                if "HTTP 401" in err or "HTTP 402" in err or "HTTP 404" in err:
                    break  # ключ/баланс/модель — повтор не поможет
                time.sleep(2 * (attempt + 1))
        self.fails += 1
        self.last_error = err
        return None, 0.0, err

    def ask_many(self, crops):
        if self.workers == 1 or len(crops) < 2:
            return [self.ask(c) for c in crops]
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(self.ask, crops))

    def describe(self):
        return f"{'локально (Ollama)' if self.backend == 'ollama' else self.url} · {self.model}"


# ----------------------------------------------------------------------------- Ollama / API сервис

def ollama_status(url=OLLAMA_URL):
    """-> (работает ли, [установленные модели], ошибка)"""
    try:
        r = _session().get(f"{url.rstrip('/')}/api/tags", timeout=4)
        r.raise_for_status()
        return True, [m["name"] for m in r.json().get("models", [])], ""
    except Exception as e:
        return False, [], f"{type(e).__name__}"


def ollama_pull(model, url=OLLAMA_URL, progress=None):
    """Скачивает модель в Ollama. progress(text, percent|None)."""
    with _session().post(f"{url.rstrip('/')}/api/pull", json={"model": model, "stream": True},
                         stream=True, timeout=(10, 3600)) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            d = json.loads(line)
            if d.get("error"):
                raise RuntimeError(d["error"])
            st = d.get("status", "")
            pct = None
            if d.get("total"):
                pct = 100.0 * d.get("completed", 0) / d["total"]
            if progress:
                progress(st, pct)
            if st == "success":
                return True
    return True


def api_models(url, key=None, proxy=None):
    h = {"Authorization": f"Bearer {key}"} if key else {}
    r = _session(proxy).get(f"{url.rstrip('/')}/models", headers=h, timeout=20)
    r.raise_for_status()
    d = r.json()
    items = d.get("data", d if isinstance(d, list) else [])
    return sorted({(m.get("id") if isinstance(m, dict) else str(m)) for m in items if m})


# ----------------------------------------------------------------------------- CLI

def _main():
    import argparse
    ap = argparse.ArgumentParser(description="VLM-судья для Autolabel Studio",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("test", "models"):
        p = sub.add_parser(name)
        p.add_argument("--backend", choices=["ollama", "openai"], default="ollama")
        p.add_argument("--url")
        p.add_argument("--key")
        p.add_argument("--model", default="qwen3-vl:4b")
        p.add_argument("--proxy")
        if name == "test":
            p.add_argument("image")
            p.add_argument("--classes", required=True)
    p = sub.add_parser("pull")
    p.add_argument("model")
    sub.add_parser("list", help="каталог локальных моделей и агрегаторов")
    a = ap.parse_args()
    if a.cmd == "list":
        print("Локальные модели (Ollama):")
        for t, s, v, n in LOCAL_MODELS:
            print(f"  {t:<16} скачать {s:<7} видеопамять ~{v:<6} {n}")
        print("\nМодели API (примерные цены, $ за 1 млн токенов, без наценки):")
        for mid, pi, po, it, note in API_PRICES:
            c = cost_per_1000(pi, po, it)
            print(f"  {mid:<36} {'?' if pi is None else f'{pi:6.2f}'} / {'?' if po is None else f'{po:6.2f}'}"
                  f"   ≈ {'?' if c is None else f'${c:.3f}'} за 1000 вырезок  {note}")
        print("\nAPI (OpenAI-совместимые):")
        for n, u, m, site in API_PROVIDERS:
            if u:
                print(f"  {n:<20} {u:<34} напр. {m}")
        return
    if a.cmd == "pull":
        ollama_pull(a.model, progress=lambda s, p: print(f"\r{s} {'' if p is None else f'{p:5.1f}%'}   ",
                                                          end="", flush=True))
        print("\n[✓] готово")
        return
    if a.cmd == "models":
        url = a.url or (OLLAMA_URL + "/v1" if a.backend == "ollama" else "")
        for m in api_models(url, a.key, a.proxy):
            print(m)
        return
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import cv2
    import numpy as np
    import autolabel as AL
    cfg = AL.load_config(a.classes)
    img = cv2.imdecode(np.fromfile(a.image, np.uint8), cv2.IMREAD_COLOR)
    j = VLMJudge(a.backend, a.model, cfg["classes"], a.url, a.key, a.proxy)
    t0 = time.time()
    k, q, raw = j.ask(img)
    name = "ничего из списка" if k == -1 else ("ОШИБКА" if k is None else j.names[k])
    print(f"{name}  (уверенность {q:.2f}, {time.time() - t0:.1f} с)\nответ модели: {raw}")


if __name__ == "__main__":
    _main()
