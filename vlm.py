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

# Локальные модели Ollama с картинками (размер скачивания, сколько нужно видеопамяти, оценка)
LOCAL_MODELS = [
    # tag,               скачать, VRAM,  примечание
    ("qwen3-vl:2b",      "1.9 ГБ", "3 ГБ",  "самая лёгкая, для слабых видеокарт/CPU"),
    ("qwen3-vl:4b",      "3.3 ГБ", "5 ГБ",  "рекомендуется для 6 ГБ видеопамяти"),
    ("qwen3-vl:8b",      "6.2 ГБ", "8 ГБ",  "точнее, нужна карта от 8 ГБ"),
    ("qwen2.5vl:3b",     "3.2 ГБ", "5 ГБ",  "проверенная, хорошо читает знаки"),
    ("qwen2.5vl:7b",     "6.0 ГБ", "8 ГБ",  "точнее, нужна карта от 8 ГБ"),
    ("gemma3:4b",        "3.4 ГБ", "5 ГБ",  "Google, хорошо понимает сцены"),
    ("gemma3:12b",       "8.2 ГБ", "12 ГБ", "точная, нужна мощная карта"),
    ("qwen3-vl:30b",     "20 ГБ",  "24 ГБ", "очень точная, для серверных карт"),
]

# Российские агрегаторы и другие OpenAI-совместимые API (оплата в рублях — у первых пяти)
API_PROVIDERS = [
    # имя,                 base_url,                         пример модели с картинками,      сайт
    ("ProxyAPI",           "https://api.proxyapi.ru/v1",     "openai/gpt-4o-mini",            "https://proxyapi.ru"),
    ("VseGPT",             "https://api.vsegpt.ru/v1",       "openai/gpt-4o-mini",            "https://vsegpt.ru"),
    ("AITunnel",           "https://api.aitunnel.ru/v1",     "gpt-4o-mini",                   "https://aitunnel.ru"),
    ("AITunnel (зеркало)", "https://ru-api.aitunnel.ru/v1",  "gpt-4o-mini",                   "https://aitunnel.ru"),
    ("Polza.ai",           "https://polza.ai/api/v1",        "openai/gpt-4o-mini",            "https://polza.ai"),
    ("provod.ai",          "https://api.provod.ai/v1",       "gpt-4o-mini",                   "https://provod.ai"),
    ("OpenRouter",         "https://openrouter.ai/api/v1",   "openai/gpt-4o-mini",            "https://openrouter.ai"),
    ("OpenAI",             "https://api.openai.com/v1",      "gpt-4o-mini",                   "https://platform.openai.com"),
    ("Ollama как API",     "http://localhost:11434/v1",      "qwen3-vl:4b",                   "https://ollama.com"),
    ("Своё",               "",                               "",                              ""),
]

# Модели с картинками, которые обычно есть у агрегаторов (точное имя смотрите кнопкой «Список моделей»)
API_MODEL_HINTS = [
    "openai/gpt-4o-mini", "openai/gpt-4.1-mini", "openai/gpt-4o", "anthropic/claude-sonnet-4-5",
    "anthropic/claude-haiku-4-5", "google/gemini-2.5-flash", "google/gemini-2.5-pro",
    "qwen/qwen2.5-vl-72b-instruct", "gpt-4o-mini", "gpt-4.1-mini",
]

OLLAMA_URL = "http://localhost:11434"


# ----------------------------------------------------------------------------- утилиты

def _session(proxy=None):
    import requests
    s = requests.Session()
    s.trust_env = False  # без системных прокси, если не задан явно
    if proxy:
        s.proxies = {"http": proxy, "https": proxy}
    return s


def encode_crop(bgr, min_side=256, max_side=768, quality=90):
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
        "object cut from a video frame (it may be small, blurry or partly outside the crop).\n"
        f"{ctx}"
        "Which of these classes is the main object in the center of the crop?\n"
        + "\n".join(lines) +
        "\n0. none — something else, background, a different sign, a box/sticker/poster, or unclear\n"
        "Answer ONLY with JSON: {\"class\": <number 0-%d>, \"confidence\": <0..1>}" % len(classes)
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
            k = int(float(d.get("class", d.get("id", -1))))
            q = float(d.get("confidence", d.get("conf", 0.8)))
            if 0 <= k <= n_cls:
                return k - 1, max(0.0, min(1.0, q))
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
                                                   "confidence": {"type": "number"}},
                  "required": ["class", "confidence"]}
        r = self.s.post(f"{self.url}/api/chat", timeout=self.timeout, json={
            "model": self.model, "stream": False, "format": schema, "keep_alive": "30m",
            "options": {"temperature": 0},
            "messages": [{"role": "user", "content": self.prompt, "images": [b64]}]})
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "")

    def _ask_openai(self, b64):
        if not self.url:
            raise RuntimeError("не задан адрес API (base_url)")
        h = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = f"Bearer {self.key}"
        r = self.s.post(f"{self.url}/chat/completions", headers=h, timeout=self.timeout, json={
            "model": self.model, "temperature": 0, "max_tokens": 60,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": self.prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}]})
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        return r.json()["choices"][0]["message"]["content"]

    def ask(self, bgr):
        """-> (класс или -1, уверенность, сырой ответ) либо (None, 0, ошибка)"""
        b64 = encode_crop(bgr)
        err = ""
        for attempt in range(2):
            try:
                self.calls += 1
                txt = self._ask_ollama(b64) if self.backend == "ollama" else self._ask_openai(b64)
                res = parse_answer(txt, len(self.classes), self.names)
                if res is not None:
                    self.fails = 0
                    return res[0], res[1], txt
                err = f"непонятный ответ: {str(txt)[:80]}"
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:160]}"
                time.sleep(1 + attempt)
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
