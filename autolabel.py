#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
autolabel.py — универсальная автоматическая разметка датасета для Darknet (YOLOv4-tiny)
по текстовому описанию классов: из видео, картинок или папок. Любые объекты, не только знаки.

Каскад распознавания:
  1. Поиск:      Grounding DINO (+ YOLO-World в пресете best), опционально нарезка кадра
                 на тайлы для мелких объектов; рамки детекторов сливаются (WBF).
  2. Класс:      голос детектора + CLIP по текстовому описанию + CLIP по образцам
                 (папка refs/<класс>/*.jpg) → одна итоговая вероятность; «мусор» отбрасывается.
  3. Чистка:     NMS, вложенные рамки («знак + столб», «деталь внутри объекта»), форма рамки.
  4. Видео:      (--track) трекинг рамок и голосование за класс по всему треку.
  5. Сомнения:   кадры с неуверенными рамками не попадают в датасет (или --uncertain drop/keep).
  6. Проверка:   вырезки всех рамок лежат в review/<класс>/ — удалите ошибочные или
                 перенесите в нужную папку и выполните команду review.

Сеть: скрипт сам находит рабочий прокси (переменные среды, настройки Windows, PAC-файл),
выбирает huggingface.co или зеркало hf-mirror.com, качает с докачкой в ./models,
доустанавливает недостающие пакеты. Потом работает офлайн.

Команды:
  python autolabel.py init                              пример classes.yaml + папки refs/
  python autolabel.py net                               проверить сеть и прокси
  python autolabel.py download [--preset best]          только скачать модели
  python autolabel.py run --source video.mp4 --classes classes.yaml [--preset best] [--preview]
  python autolabel.py review --dataset dataset          применить ручную проверку вырезок

Пресеты:  fast      — DINO tiny + CLIP base (быстро)
          balanced  — DINO tiny + CLIP large (по умолчанию)
          best      — DINO base + YOLO-World x + CLIP large + тайлы 2x2 (точно, медленно)

Установка: torch поставьте под свою видеокарту (https://pytorch.org), остальное скрипт
доустановит сам:  pip install torch
"""
import argparse
import importlib
import importlib.util
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".ts"}

PRESETS = {
    "fast": dict(gdino_model="IDEA-Research/grounding-dino-tiny",
                 clip_model="openai/clip-vit-base-patch32", tiles=0, yoloworld="none"),
    "balanced": dict(gdino_model="IDEA-Research/grounding-dino-tiny",
                     clip_model="openai/clip-vit-large-patch14", tiles=0, yoloworld="none"),
    "best": dict(gdino_model="IDEA-Research/grounding-dino-base",
                 clip_model="openai/clip-vit-large-patch14", tiles=2, yoloworld="yolov8x-worldv2.pt"),
}

HF_ENDPOINTS = ["https://huggingface.co", "https://hf-mirror.com"]
PROBE_REPO = "openai/clip-vit-base-patch32"
YW_TAGS = ("v8.3.0", "v8.2.0", "v8.1.0")

# Запасные списки файлов, если API сайта недоступно
KNOWN_FILES = {
    "IDEA-Research/grounding-dino-tiny": [
        "config.json", "model.safetensors", "preprocessor_config.json", "special_tokens_map.json",
        "tokenizer.json", "tokenizer_config.json", "vocab.txt"],
    "IDEA-Research/grounding-dino-base": [
        "config.json", "model.safetensors", "preprocessor_config.json", "special_tokens_map.json",
        "tokenizer.json", "tokenizer_config.json", "vocab.txt"],
}
for _clip in ("openai/clip-vit-base-patch32", "openai/clip-vit-base-patch16", "openai/clip-vit-large-patch14"):
    KNOWN_FILES[_clip] = ["config.json", "merges.txt", "model.safetensors", "preprocessor_config.json",
                          "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"]

HF_SMALL_FILES = {"config.json", "preprocessor_config.json", "processor_config.json", "tokenizer.json",
                  "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "vocab.json",
                  "merges.txt", "added_tokens.json", "generation_config.json", "spiece.model",
                  "sentencepiece.bpe.model", "model.safetensors.index.json"}

BASE_PACKAGES = {"requests": "requests", "transformers": "transformers",
                 "cv2": "opencv-python", "PIL": "pillow", "yaml": "pyyaml", "numpy": "numpy"}
PROXY_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
# --no-proxy или AUTOLABEL_NO_PROXY=1: только прямое подключение, прокси не используется нигде
NO_PROXY_MODE = os.environ.get("AUTOLABEL_NO_PROXY") == "1"


def disable_proxy():
    global NO_PROXY_MODE
    NO_PROXY_MODE = True
    os.environ["AUTOLABEL_NO_PROXY"] = "1"
    for k in PROXY_KEYS:
        os.environ.pop(k, None)
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "*"  # и системный прокси Windows тоже мимо
    os.environ["PIP_PROXY"] = ""


def log(msg):
    print(msg, flush=True)


def imread_u(path):
    """cv2.imread, который понимает русские буквы в пути (Windows)."""
    import cv2
    import numpy as np
    try:
        return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_u(path, img, quality=95):
    import cv2
    ext = Path(path).suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, quality] if ext.lower() in (".jpg", ".jpeg") else [])
    if ok:
        buf.tofile(str(path))
    return ok


def replace_retry(src, dst, tries=10):
    """os.replace, переживающий кратковременную блокировку файла в Windows."""
    for _ in range(tries):
        try:
            os.replace(src, dst)
            return True
        except PermissionError:
            time.sleep(0.03)
    return False


def open_video(p):
    """cv2.VideoCapture, который не ломается на русских буквах в пути (Windows): при неудаче
    пробует короткое имя 8.3, потом временную копию с латинским именем."""
    import cv2
    cap = cv2.VideoCapture(str(p))
    if cap.isOpened():
        return cap, None
    cap.release()
    if sys.platform == "win32":
        try:
            import ctypes
            buf = ctypes.create_unicode_buffer(1024)
            if ctypes.windll.kernel32.GetShortPathNameW(str(p), buf, 1024):
                cap = cv2.VideoCapture(buf.value)
                if cap.isOpened():
                    return cap, None
                cap.release()
        except Exception:
            pass
    if not str(p).isascii():
        import tempfile
        tmp = Path(tempfile.gettempdir()) / f"autolabel_{abs(hash(str(p)))}{p.suffix}"
        try:
            log(f"[i] {p.name}: в пути русские буквы — копирую во временную папку...")
            shutil.copy2(p, tmp)
            cap = cv2.VideoCapture(str(tmp))
            if cap.isOpened():
                return cap, tmp
            cap.release()
            tmp.unlink(missing_ok=True)
        except Exception as e:
            log(f"[!] не удалось скопировать: {e}")
    log(f"[!] не открылось видео {p} — проверьте, что файл целый и OpenCV знает этот формат")
    return None, None


def natural_key(p):
    """video2 раньше video10."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(p))]


def safe_name(s):
    return re.sub(r"[^\w\-]+", "_", str(s)).strip("_")[:60] or "x"


# ============================================================================ сеть и прокси

def norm_proxy(p):
    p = (p or "").strip()
    if not p:
        return None
    return p if "://" in p else "http://" + p


def pac_proxies(url):
    try:
        import requests
        s = requests.Session()
        s.trust_env = False
        if url.lower().startswith("file:"):
            text = Path(re.sub(r"^file:/*", "", url)).read_text(errors="ignore")
        else:
            text = s.get(url, timeout=8).text
        return list(dict.fromkeys(re.findall(r"PROXY\s+([\w.\-]+:\d+)", text)))
    except Exception:
        return []


def proxy_candidates(user_proxy=None, with_pac=True):
    """Все возможные пути в интернет: явный прокси, переменные среды, Windows, PAC, напрямую."""
    if NO_PROXY_MODE:
        return [("без прокси", None)]
    c = []
    if user_proxy:
        c.append(("--proxy", norm_proxy(user_proxy)))
    for k in PROXY_KEYS:
        if os.environ.get(k):
            c.append((f"переменная {k}", norm_proxy(os.environ[k])))
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                r"Software\Microsoft\Windows\CurrentVersion\Internet Settings") as key:
                def get(n):
                    try:
                        return winreg.QueryValueEx(key, n)[0]
                    except OSError:
                        return None
                srv, enabled, pac = get("ProxyServer"), get("ProxyEnable"), get("AutoConfigURL")
            if srv:
                parts = dict(x.split("=", 1) for x in srv.split(";") if "=" in x)
                one = parts.get("https") or parts.get("http") or (srv if "=" not in srv else None)
                if one:
                    c.append(("настройки Windows" + ("" if enabled else " (выключен)"), norm_proxy(one)))
            if pac and with_pac:
                for hp in pac_proxies(pac):
                    c.append(("PAC-файл", norm_proxy(hp)))
        except Exception:
            pass
    c.append(("без прокси", None))
    seen, out = set(), []
    for name, p in c:
        if p not in seen:
            seen.add(p)
            out.append((name, p))
    return out


def set_env_proxy(proxy):
    """Задаёт прокси для всего процесса (requests, urllib, pip, ultralytics...)."""
    for k in PROXY_KEYS + ("NO_PROXY", "no_proxy"):
        os.environ.pop(k, None)
    if proxy:
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            os.environ[k] = proxy
        os.environ["NO_PROXY"] = os.environ["no_proxy"] = "localhost,127.0.0.1"
    else:  # не дать Windows подсунуть системный прокси
        os.environ["NO_PROXY"] = os.environ["no_proxy"] = "*"


def inject_truststore():
    """Сертификаты из хранилища Windows/macOS — нужно, если прокси подменяет HTTPS.
    Пакет truststore необязателен (нужен Python 3.10+): нет — работаем без него."""
    if importlib.util.find_spec("truststore") is None and sys.version_info >= (3, 10):
        for name, p in proxy_candidates(None, with_pac=False):
            cmd = [sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", "truststore"]
            if p:
                cmd += ["--proxy", p]
            if subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
                importlib.invalidate_caches()
                break
    try:
        import truststore
        truststore.inject_into_ssl()
    except Exception:
        pass


def ca_bundle():
    """Свой файл сертификатов (корпоративный прокси), если задан в переменных среды."""
    for k in ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE"):
        v = os.environ.get(k)
        if v and Path(v).is_file():
            return v
    return True


class Net:
    def __init__(self, proxy, endpoint=None):
        import requests
        self.proxy, self.endpoint = proxy, endpoint
        self.s = requests.Session()
        self.s.trust_env = False
        self.s.verify = ca_bundle()
        self.s.headers["User-Agent"] = "autolabel/2.0"
        if proxy:
            self.s.proxies = {"http": proxy, "https": proxy}


def probe(url, proxy, timeout=10):
    import requests
    s = requests.Session()
    s.trust_env = False
    s.verify = ca_bundle()
    if proxy:
        s.proxies = {"http": proxy, "https": proxy}
    try:
        r = s.head(url, timeout=timeout, allow_redirects=False)
        return r.status_code < 400, f"HTTP {r.status_code}"
    except requests.exceptions.SSLError:
        return False, "ошибка SSL (прокси подменяет сертификат: pip install truststore)"
    except requests.exceptions.ProxyError:
        return False, "прокси не отвечает"
    except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout):
        return False, "таймаут"
    except requests.exceptions.ConnectionError:
        return False, "нет соединения"
    except Exception as e:
        return False, type(e).__name__


def find_route(targets, cands, verbose=True):
    """targets: [(подпись, endpoint, url_для_проверки)] -> Net или None"""
    for label, endpoint, url in targets:
        for name, p in cands:
            ok, msg = probe(url, p)
            if verbose:
                log(f"    {'✓' if ok else '✗'} {label:<16} через {name:<26} {p or '':<32} {msg}")
            if ok:
                return Net(p, endpoint)
    return None


class NetManager:
    """Ищет сеть только когда что-то действительно нужно скачать."""

    def __init__(self, a):
        self.a, self._hf, self._gh, self._cands = a, False, False, None

    def cands(self):
        if self._cands is None:
            self._cands = proxy_candidates(self.a.proxy)
        return self._cands

    def hf(self):
        if self._hf is False:
            self._hf = None
            if not self.a.offline:
                eps = [self.a.endpoint.rstrip("/")] if self.a.endpoint else HF_ENDPOINTS
                log("[i] ищу доступ к HuggingFace...")
                self._hf = find_route(
                    [(ep.split("//")[1], ep, f"{ep}/{PROBE_REPO}/resolve/main/config.json") for ep in eps],
                    self.cands())
                if self._hf:
                    set_env_proxy(self._hf.proxy)
                    os.environ["HF_ENDPOINT"] = self._hf.endpoint
                    log(f"[✓] качаю с {self._hf.endpoint} через {self._hf.proxy or 'прямое подключение'}")
        return self._hf

    def gh(self):
        if self._gh is False:
            self._gh = None
            if not self.a.offline:
                log("[i] ищу доступ к GitHub...")
                hf = self._hf if self._hf else None
                cands = ([("тот же, что для HF", hf.proxy)] if hf else []) + self.cands()
                self._gh = find_route([("github.com", None, "https://github.com/ultralytics/assets/releases")],
                                      cands)
                if self._gh:
                    set_env_proxy(self._gh.proxy)
        return self._gh


# ============================================================================ пакеты и загрузка

def ensure_packages(mods, cands=None):
    missing = [pip for mod, pip in mods.items() if importlib.util.find_spec(mod) is None]
    if not missing:
        return
    log(f"[i] доустанавливаю пакеты: {' '.join(missing)}")
    for name, p in (cands or proxy_candidates(None, with_pac=False)):
        env = dict(os.environ)
        for k in PROXY_KEYS:
            env.pop(k, None)
        cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *missing]
        if p:
            cmd += ["--proxy", p]
        elif NO_PROXY_MODE:
            cmd += ["--proxy", ""]
            env["NO_PROXY"] = env["no_proxy"] = "*"
        log(f"    pip через {name} {p or ''}")
        if subprocess.call(cmd, env=env) == 0:
            importlib.invalidate_caches()
            return
    sys.exit(f"[!] Не удалось установить: {' '.join(missing)}. Установите вручную: pip install {' '.join(missing)}")


def download_file(net, url, dest, desc=None, attempts=6):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    desc = desc or dest.name
    for att in range(1, attempts + 1):
        pos = tmp.stat().st_size if tmp.exists() else 0
        done = pos
        try:
            hdr = {"Range": f"bytes={pos}-"} if pos else {}
            with net.s.get(url, headers=hdr, stream=True, timeout=(20, 120)) as r:
                if r.status_code == 404:
                    raise FileNotFoundError(url)
                if r.status_code != 416:  # 416 — уже скачан целиком
                    r.raise_for_status()
                    if pos and r.status_code != 206:
                        pos = done = 0
                    total = pos + int(r.headers.get("Content-Length") or 0)
                    t0 = 0.0
                    with open(tmp, "ab" if pos else "wb") as f:
                        for chunk in r.iter_content(1 << 20):
                            f.write(chunk)
                            done += len(chunk)
                            if time.time() - t0 > 0.5:
                                t0 = time.time()
                                pct = f"{100 * done / total:5.1f}%" if total else ""
                                print(f"\r    {desc}: {done / 2**20:9.1f} МБ {pct}   ", end="", flush=True)
            print(f"\r    {desc}: {done / 2**20:9.1f} МБ — готово          ", flush=True)
            break
        except FileNotFoundError:
            raise
        except Exception as e:
            hint = " (прокси подменяет сертификат: pip install truststore)" if "SSL" in type(e).__name__ else ""
            print(f"\n    [!] {desc}: {type(e).__name__}{hint} — повтор {att}/{attempts}", flush=True)
            time.sleep(min(30, 2 ** att))
    else:
        raise RuntimeError(f"не удалось скачать {url}")
    tmp.replace(dest)


def hf_select(files):
    top = [f for f in files if "/" not in f]
    sel = [f for f in top if f in HF_SMALL_FILES or re.fullmatch(r"model-\d+-of-\d+\.safetensors", f)]
    if "model.safetensors" in top:
        sel.append("model.safetensors")
    elif not any(f.endswith(".safetensors") for f in sel):
        sel += [f for f in top if f == "pytorch_model.bin" or re.fullmatch(r"pytorch_model-\d+-of-\d+\.bin", f)]
        sel += [f for f in top if f == "pytorch_model.bin.index.json"]
    return sorted(set(sel))


def hf_files(net, repo):
    try:
        r = net.s.get(f"{net.endpoint}/api/models/{repo}", timeout=20)
        r.raise_for_status()
        files = [x["rfilename"] for x in r.json().get("siblings", [])]
        if files:
            return hf_select(files)
    except Exception:
        pass
    if repo in KNOWN_FILES:
        return KNOWN_FILES[repo]
    sys.exit(f"[!] Не удалось получить список файлов модели {repo}")


def hf_local_dir(repo, models_dir):
    return Path(models_dir) / repo.replace("/", "__")


def hf_complete(p):
    """Модель уже скачана: есть метка .complete или (скачано вручную) config.json + веса."""
    if not p.is_dir() or list(p.glob("*.part")):
        return False
    if (p / ".complete").exists():
        return True
    has_w = any(p.glob("*.safetensors")) or any(p.glob("pytorch_model*.bin"))
    return (p / "config.json").exists() and has_w


def manual_help(repo, local):
    files = KNOWN_FILES.get(repo, ["config.json", "model.safetensors", "preprocessor_config.json",
                                   "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"])
    lines = "\n".join(f"      https://huggingface.co/{repo}/resolve/main/{f}" for f in files)
    return (f"[!] Нет доступа в интернет, а модели {repo} нет в {local}.\n"
            f"    Скачайте браузером (или через hf-mirror.com) файлы:\n{lines}\n"
            f"    и положите в папку {local.resolve()}, затем создайте в ней пустой файл .complete\n"
            f"    Проверить сеть: python autolabel.py net")


def ensure_hf_model(repo, models_dir, nm):
    p = Path(repo).expanduser()
    if p.is_dir():
        if not (p / "config.json").exists():
            sys.exit(f"[!] В папке {p.resolve()} нет config.json")
        return str(p.resolve())
    if p.suffix or "\\" in repo or repo.startswith(("./", ".\\", "models")):
        sys.exit(f"[!] Папка модели не найдена: {p.resolve()} (текущая папка {Path.cwd()})")
    local = hf_local_dir(repo, models_dir)
    if hf_complete(local):
        log(f"[i] {repo}: уже скачана ({local})")
        return str(local.resolve())
    net = nm.hf()
    if net is None:
        sys.exit(manual_help(repo, local))
    files = hf_files(net, repo)
    log(f"[↓] {repo} → {local}")
    for f in files:
        if not (local / f).exists():
            download_file(net, f"{net.endpoint}/{repo}/resolve/main/{f}", local / f, f)
    (local / ".complete").write_text("ok")
    return str(local.resolve())


def ensure_yoloworld(name, models_dir, nm):
    """Веса YOLO-World + текстовый энкодер CLIP, который ему нужен."""
    p = Path(name)
    if p.is_file():
        weights = str(p)
    else:
        local = Path(models_dir) / p.name
        if not local.is_file():
            net = nm.gh()
            if net is None:
                raise RuntimeError("нет доступа к github.com")
            log(f"[↓] {p.name} → {local}")
            for tag in YW_TAGS:
                try:
                    download_file(net, f"https://github.com/ultralytics/assets/releases/download/{tag}/{p.name}",
                                  local, p.name)
                    break
                except FileNotFoundError:
                    continue
            else:
                raise RuntimeError(f"{p.name} не найден в релизах ultralytics")
        weights = str(local)
    ensure_packages({"ultralytics": "ultralytics"}, nm.cands())
    if importlib.util.find_spec("clip") is None:
        net = nm.gh()
        cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
               "https://github.com/ultralytics/CLIP/archive/refs/heads/main.zip"]
        if net and net.proxy:
            cmd += ["--proxy", net.proxy]
        log("[i] устанавливаю CLIP для YOLO-World")
        if subprocess.call(cmd):
            raise RuntimeError("не удалось установить пакет clip")
        importlib.invalidate_caches()
    import clip
    clip.load("ViT-B/32", device="cpu")  # скачает веса при первом запуске
    return weights


# ============================================================================ конфиг классов

EXAMPLE_YAML = """\
# Классы для autolabel.py. Порядок задаёт ID класса в Darknet (0, 1, 2...).
#
# Поля класса (обязательно только name):
#   name       — имя в obj.names
#   prompt     — фраза (или список фраз) для детектора, по-английски, коротко и по делу
#   desc       — описание (или список) для CLIP: цвет, форма, отличительные детали
#   refs       — папка с образцами (по умолчанию refs/<name>/): 5–20 вырезок объекта
#   thr        — свой порог уверенности для класса (по умолчанию --accept)
#   max_aspect — макс. вытянутость рамки (длинная/короткая сторона), например 1.6 для знаков
#   refine     — color: подогнать рамку к цветной табличке знака (отрезать стойку и фон)
#
# Короткие формы тоже работают:
#   classes: [cat, dog, person]
#   classes: {cat: "a cat", dog: "a dog"}

# Что детектор ищет «в общем» (помогает найти всё, потом классификатор разберёт):
generic:
  - traffic sign
  - traffic light

# Если классификатор выбирает одно из этого — рамка выбрасывается.
# Ещё сильнее: положите вырезки ложных срабатываний в refs/_reject/
reject:
  - a photo of some other traffic sign
  - a photo of a car
  - a photo of a street lamp
  - a photo of a billboard or advertisement
  - a photo of a tree or building
  - a photo of the sky or road surface
  - a photo of a cardboard box
  - a photo of a sticker or label on a box
  - a photo of a wall, radiator or pipe
  - a photo of a floor or road markings

classes:
  - name: STOP                       # 2.5
    prompt: octagonal stop sign
    desc: a red octagonal STOP traffic sign
    max_aspect: 1.6
    refine: color          # подогнать рамку к табличке (без стойки)

  - name: parkovka                   # 6.4
    prompt: parking sign with big letter P
    desc: a blue square parking traffic sign with a white letter P
    max_aspect: 1.6
    refine: color          # подогнать рамку к табличке (без стойки)

  - name: ystypi dorogy              # 2.4
    prompt: upside down triangle yield sign
    desc: an inverted white triangle traffic sign with red border, give way sign
    max_aspect: 1.6
    refine: color          # подогнать рамку к табличке (без стойки)

  - name: dorozhnie raboti           # 1.25
    prompt: roadwork warning sign with worker
    desc: a white triangular warning traffic sign with red border showing a worker with a shovel
    max_aspect: 1.6
    refine: color          # подогнать рамку к табличке (без стойки)

  - name: peshehodniy perehod        # 5.19
    prompt: pedestrian crosswalk sign with walking person
    desc: a blue square pedestrian crossing traffic sign with a walking person on a white triangle
    max_aspect: 1.6
    refine: color          # подогнать рамку к табличке (без стойки)

  - name: iskusstvennaya nerovnost   # 1.17
    prompt: speed bump warning sign
    desc: a white triangular warning traffic sign with red border showing a speed bump
    max_aspect: 1.6
    refine: color          # подогнать рамку к табличке (без стойки)

  - name: viezd zapreshen            # 3.1
    prompt: round sign with red border and white center
    desc: a round traffic sign, white circle with a thick red border and nothing inside
    max_aspect: 1.6
    refine: color          # подогнать рамку к табличке (без стойки)

  - name: traffic                    # светофор
    prompt: traffic light
    desc: a photo of a traffic light with red, yellow and green signals
    max_aspect: 4
"""


def load_config(path):
    import yaml
    p = Path(path)
    if not p.exists():
        sys.exit(f"[!] Нет файла {p}. Создайте пример: python autolabel.py init")
    text = p.read_text(encoding="utf-8")
    cfg = json.loads(text) if p.suffix.lower() == ".json" else yaml.safe_load(text)
    if isinstance(cfg, list):
        cfg = {"classes": cfg}
    raw = cfg.get("classes")
    if isinstance(raw, dict):
        raw = [{"name": k, "prompt": v} if isinstance(v, str) else {"name": k, **(v or {})}
               for k, v in raw.items()]
    if not raw:
        sys.exit("[!] В конфиге нет классов")
    classes = []
    for c in raw:
        c = {"name": c} if isinstance(c, str) else dict(c)
        c["name"] = str(c["name"])
        pr = c.get("prompts") or c.get("prompt") or c["name"].replace("_", " ")
        c["prompts"] = [pr] if isinstance(pr, str) else [str(x) for x in pr]
        d = c.get("desc") or []
        c["desc"] = [d] if isinstance(d, str) else [str(x) for x in d]
        classes.append(c)
    names = [c["name"] for c in classes]
    if len(set(names)) != len(names):
        sys.exit("[!] Имена классов повторяются")
    return {"classes": classes, "generic": list(cfg.get("generic") or []),
            "reject": list(cfg.get("reject") or ["a photo of something else", "a photo of the background"])}


# ============================================================================ геометрия

def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    return iw * ih if iw > 0 and ih > 0 else 0.0


def iou(a, b):
    i = inter(a, b)
    return i / (area(a) + area(b) - i + 1e-9)


def elong(b):
    w, h = b[2] - b[0], b[3] - b[1]
    return max(w / max(h, 1e-6), h / max(w, 1e-6))


def nms(boxes, scores, thr):
    order = sorted(range(len(boxes)), key=lambda i: -scores[i])
    keep = []
    for i in order:
        if all(iou(boxes[i], boxes[k]) <= thr for k in keep):
            keep.append(i)
    return keep


def wbf(dets, n_models, thr):
    """Weighted box fusion. dets: [(box, score, model, pvec|None)] -> [(box, score, pvec|None)]"""
    import numpy as np
    clusters = []
    for b, s, m, pv in sorted(dets, key=lambda d: -d[1]):
        best, bi = thr, -1
        for i, c in enumerate(clusters):
            v = iou(b, c["box"])
            if v >= best:
                best, bi = v, i
        if bi < 0:
            clusters.append({"box": b.copy(), "bs": [b], "ss": [s], "ms": {m}, "pv": [pv] if pv is not None else []})
        else:
            c = clusters[bi]
            c["bs"].append(b); c["ss"].append(s); c["ms"].add(m)
            if pv is not None:
                c["pv"].append(pv)
            w = np.array(c["ss"])
            c["box"] = (np.array(c["bs"]) * w[:, None]).sum(0) / w.sum()
    out = []
    for c in clusters:
        ss = np.array(c["ss"])
        score = float(ss.max()) if n_models == 1 else float(ss.mean() * (0.5 + 0.5 * len(c["ms"]) / n_models))
        pv = np.mean(c["pv"], 0) if c["pv"] else None
        out.append((c["box"], score, pv))
    return out


def resolve_nested(boxes, keep, ok, lim, thr):
    """Вложенные рамки. Внешняя рамка вытянута сильнее внутренней (знак + столб) — убираем внешнюю;
    иначе внутренняя — деталь объекта (буква на знаке, лампа светофора) — убираем внутреннюю.
    ok[i] — рамка похожа на объект; lim[i] — допустимая вытянутость её класса."""
    if thr <= 0:
        return list(keep), 0
    alive = set(keep)
    for i in sorted(keep, key=lambda k: area(boxes[k])):
        for j in keep:
            if i == j or i not in alive or j not in alive:
                continue
            if area(boxes[j]) < 1.5 * area(boxes[i]) or inter(boxes[i], boxes[j]) / (area(boxes[i]) + 1e-9) < thr:
                continue
            if ok[i] and elong(boxes[j]) > 1.25 * elong(boxes[i]) and not (ok[j] and lim[j] > 2):
                alive.discard(j)
            elif ok[j]:
                alive.discard(i)
    res = [k for k in keep if k in alive]
    return res, len(keep) - len(res)


def vote_tracks(frames, iou_thr, min_len=2):
    """Связывает рамки соседних кадров в треки и голосует за класс по всему треку."""
    import numpy as np
    tracks, prev, nid = defaultdict(list), [], 0
    for fr in frames:
        cur, used = [], set()
        for d in fr["dets"]:
            b, best, bt = d["box"], 0.0, None
            for pb, tid in prev:
                if tid in used:
                    continue
                s = iou(b, pb)
                if s < iou_thr:  # быстро движущийся/растущий объект — по расстоянию центров
                    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                    px, py = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
                    size = max(b[2] - b[0], b[3] - b[1], pb[2] - pb[0], pb[3] - pb[1])
                    ratio = (area(b) + 1e-9) / (area(pb) + 1e-9)
                    if ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 < size and 0.33 < ratio < 3:
                        s = iou_thr * 0.99
                    else:
                        continue
                if s > best:
                    best, bt = s, tid
            if bt is None:
                bt, nid = nid, nid + 1
            used.add(bt)
            tracks[bt].append(d)
            cur.append((b, bt))
        prev = cur
    n = 0
    for ds in tracks.values():
        if len(ds) < min_len:
            continue
        lp = np.sum([np.log(d["p"] + 1e-6) for d in ds], 0)
        c = int(lp.argmax())
        conf = float(np.mean([d["p"][c] for d in ds]))
        for d in ds:
            d["c"], d["conf"] = c, conf
        n += 1
    return n


# ============================================================================ модели

def _as_tensor(out, *names):
    import torch
    if torch.is_tensor(out):
        return out
    for n in names:
        v = getattr(out, n, None)
        if v is not None:
            return v
    raise RuntimeError("неожиданный формат вывода модели")


class GroundingDinoDet:
    """Детектор по тексту. Класс выбирается по отличительным словам каждого промпта."""

    def __init__(self, path, device, phrases, n_cls, box_thr):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(path, local_files_only=True).to(device).eval()
        self.device, self.box_thr, self.n_cls = device, box_thr, n_cls
        ps = [(t.lower().strip(" ."), o) for t, o in phrases]
        self.text = ". ".join(t for t, _ in ps) + "."
        enc = self.proc.tokenizer(self.text, return_offsets_mapping=True)
        if len(enc["input_ids"]) > 250:
            sys.exit("[!] Слишком длинные промпты для Grounding DINO (макс. ~250 токенов) — сократите prompt/generic")
        offs, pos, toks = enc["offset_mapping"], 0, []
        for t, _ in ps:
            s, e = pos, pos + len(t)
            toks.append([(i, self.text[a:b]) for i, (a, b) in enumerate(offs) if b > a and a >= s and b <= e])
            pos = e + 2
        self.all_idx = sorted({i for tk in toks for i, _ in tk})
        self.cls_idx = []
        for c in range(n_cls):
            own = [x for tk, (_, o) in zip(toks, ps) if o == c for x in tk]
            other = {s for tk, (_, o) in zip(toks, ps) if o != c for _, s in tk}
            uniq = [i for i, s in own if s not in other]
            self.cls_idx.append(uniq or [i for i, _ in own])

    def __call__(self, pil):
        import numpy as np
        torch = self.torch
        inputs = self.proc(images=pil, text=self.text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        prob = out.logits.sigmoid()[0].float().cpu()
        boxes = out.pred_boxes[0].float().cpu().numpy()
        det = prob[:, self.all_idx].max(-1).values
        cls = torch.stack([prob[:, ix].mean(-1) if ix else torch.zeros(len(prob)) for ix in self.cls_idx], -1).numpy()
        W, H = pil.size
        res = []
        for q in torch.nonzero(det > self.box_thr).flatten().tolist():
            pv = cls[q]
            pv = pv / pv.sum() if pv.sum() > 0 else None
            cx, cy, bw, bh = boxes[q]
            res.append((np.array([(cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H]),
                        float(det[q]), pv))
        return res


class YoloWorldDet:
    def __init__(self, weights, device, phrases, n_cls, conf):
        from ultralytics import YOLOWorld
        self.model = YOLOWorld(weights)
        self.model.set_classes([t for t, _ in phrases])
        self.owner = [o for _, o in phrases]
        self.n_cls, self.conf = n_cls, conf
        self.dev = "0" if str(device).startswith("cuda") else "cpu"

    def __call__(self, pil):
        import numpy as np
        r = self.model.predict(pil, conf=self.conf, device=self.dev, verbose=False)[0]
        res = []
        for xyxy, s, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
            o = self.owner[int(c)]
            pv = None
            if o >= 0:
                pv = np.full(self.n_cls, 0.1 / max(self.n_cls - 1, 1))
                pv[o] = 0.9
            res.append((xyxy.astype(float), float(s), pv))
        return res


def refine_color(rgb, b):
    """Подгоняет рамку к цветной табличке знака (красный/синий/жёлтый), отрезая стойку и фон."""
    import cv2
    import numpy as np
    H, W = rgb.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in b)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    crop = rgb[y1:y2, x1:x2]
    if crop.size == 0 or min(crop.shape[:2]) < 8:
        return b
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = ((h < 8) | (h > 165)) & (s > 100) & (v > 60)
    blue = (h > 95) & (h < 130) & (s > 100) & (v > 50)
    yellow = (h > 18) & (h < 34) & (s > 120) & (v > 120)
    m = (red | blue | yellow).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return b
    k = 1 + int(st[1:, cv2.CC_STAT_AREA].argmax())
    x, y, w, hh, a = st[k]
    if a < max(15, 0.004 * crop.shape[0] * crop.shape[1]) or w < 5 or hh < 5:
        return b
    # кольцо знака может распасться на дуги — присоединяем соседние крупные куски
    for j in range(1, n):
        if j == k or st[j, cv2.CC_STAT_AREA] < 0.1 * a:
            continue
        jx, jy, jw, jh = st[j, :4]
        gap = 0.3 * max(w, hh)
        if jx < x + w + gap and jx + jw > x - gap and jy < y + hh + gap and jy + jh > y - gap:
            nx, ny = min(x, jx), min(y, jy)
            w, hh = max(x + w, jx + jw) - nx, max(y + hh, jy + jh) - ny
            x, y = nx, ny
    pad = 0.05 * max(w, hh)
    return np.array([max(0, x1 + x - pad), max(0, y1 + y - pad),
                     min(W, x1 + x + w + pad), min(H, y1 + y + hh + pad)], float)


def run_tiled(det, pil, tiles, overlap=0.25):
    """Кадр целиком + сетка тайлов tiles x tiles (для мелких объектов)."""
    import numpy as np
    out = list(det(pil))
    if tiles < 2:
        return out
    W, H = pil.size
    tw, th = W / (tiles - (tiles - 1) * overlap), H / (tiles - (tiles - 1) * overlap)
    for i in range(tiles):
        for j in range(tiles):
            x0, y0 = int(i * tw * (1 - overlap)), int(j * th * (1 - overlap))
            x1, y1 = min(W, int(x0 + tw)), min(H, int(y0 + th))
            for b, s, pv in det(pil.crop((x0, y0, x1, y1))):
                b = b + np.array([x0, y0, x0, y0])
                m = 3  # рамки, обрезанные внутренним краем тайла, не берём
                if (x0 > 0 and b[0] <= x0 + m) or (x1 < W and b[2] >= x1 - m) or \
                   (y0 > 0 and b[1] <= y0 + m) or (y1 < H and b[3] >= y1 - m):
                    continue
                out.append((b, s, pv))
    return out


class ClipClf:
    """CLIP: вероятность класса по тексту и по образцам."""

    def __init__(self, path, device, classes, reject, refs_root, ref_min):
        import numpy as np
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.np, self.torch, self.device = np, torch, device
        self.model = CLIPModel.from_pretrained(path, local_files_only=True).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(path, local_files_only=True)
        self.scale = float(self.model.logit_scale.exp())
        self.C, self.ref_min = len(classes), ref_min
        cls_emb = []
        for c in classes:
            texts = c["desc"] + [f"a photo of {p}" for p in c["prompts"]]
            e = self._text(texts).mean(0)
            cls_emb.append(e / np.linalg.norm(e))
        rej = self._text(reject) if reject else np.zeros((0, cls_emb[0].shape[0]), np.float32)
        self.T = np.concatenate([np.stack(cls_emb), rej]).astype(np.float32)
        self.refs = {}
        from PIL import Image

        def load_dir(d):
            imgs = []
            if d.is_dir():
                for f in sorted(d.iterdir()):
                    if f.suffix.lower() in IMG_EXT:
                        try:
                            imgs.append(Image.open(f).convert("RGB"))
                        except Exception:
                            pass
            return self.embed(imgs) if imgs else None

        for ci, c in enumerate(classes):
            e = load_dir(Path(c.get("refs") or Path(refs_root) / c["name"]))
            if e is not None:
                self.refs[ci] = e
        # refs/_reject/ — вырезки того, что НЕ объект (коробки, стены, ...): похожие рамки выбрасываются
        self.rej_refs = load_dir(Path(refs_root) / "_reject")
        if self.rej_refs is not None:
            log(f"[i] образцы «не объект» (refs/_reject): {len(self.rej_refs)}")
        if self.refs:
            log("[i] образцы: " + ", ".join(f"{classes[c]['name']}={len(v)}" for c, v in self.refs.items()))
            few = [classes[c]["name"] for c in range(self.C) if c not in self.refs]
            if few:
                log(f"[i] без образцов (только текст): {', '.join(few)}")

    def _text(self, texts):
        torch = self.torch
        with torch.no_grad():
            t = self.proc(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            e = _as_tensor(self.model.get_text_features(**t), "text_embeds", "pooler_output").float()
        e = e / e.norm(dim=-1, keepdim=True)
        return e.cpu().numpy()

    def embed(self, images, bs=32):
        torch, np = self.torch, self.np
        outs = []
        for k in range(0, len(images), bs):
            with torch.no_grad():
                t = self.proc(images=images[k:k + bs], return_tensors="pt").to(self.device)
                e = _as_tensor(self.model.get_image_features(**t), "image_embeds", "pooler_output").float()
            outs.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy())
        return np.concatenate(outs) if outs else np.zeros((0, self.T.shape[1]), np.float32)

    def text_probs(self, F):
        np = self.np
        z = self.scale * F @ self.T.T
        z = np.exp(z - z.max(1, keepdims=True))
        p = z / z.sum(1, keepdims=True)
        return np.concatenate([p[:, :self.C], p[:, self.C:].sum(1, keepdims=True)], 1)

    def rej_sim(self, F):
        """Макс. сходство с образцами «не объект» (refs/_reject) или None."""
        return None if self.rej_refs is None else (F @ self.rej_refs.T).max(1)

    def cls_ref_sim(self, F):
        """Макс. сходство с образцами любого класса или None."""
        if not self.refs:
            return None
        return self.np.max(self.np.stack([(F @ R.T).max(1) for R in self.refs.values()], 1), 1)

    def ref_probs(self, F, pt):
        """Голос образцов refs/<класс>. Классы без образцов не трогаем (раньше их выбрасывало),
        мусор из refs/_reject здесь не учитывается — для него отдельный жёсткий фильтр."""
        np = self.np
        if not self.refs:
            return None
        top = lambda R: np.sort(F @ R.T, 1)[:, -3:].mean(1)  # noqa
        has = sorted(self.refs)
        sims = np.stack([top(self.refs[c]) for c in has], 1)
        mx = sims.max(1)
        e = np.exp((sims - mx[:, None]) * 50)
        P = pt.copy()
        P[:, has] = e / e.sum(1, keepdims=True) * P[:, has].sum(1, keepdims=True)
        covered = np.isin(pt[:, :self.C].argmax(1), has)  # текст думает, что это класс с образцами
        prej = np.where(covered, 1 / (1 + np.exp(-(self.ref_min - mx) * 25)), pt[:, self.C])
        P[:, :self.C] *= (1 - prej)[:, None] / (P[:, :self.C].sum(1, keepdims=True) + 1e-9)
        P[:, self.C] = prej
        return P


# ============================================================================ Darknet

YOLOV4_TINY_CFG = """[net]
# Testing
#batch=1
#subdivisions=1
# Training
batch={batch}
subdivisions={subdivisions}
width={width}
height={height}
channels=3
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

learning_rate=0.00261
burn_in=1000
max_batches = {max_batches}
policy=steps
steps={steps}
scales=.1,.1

[convolutional]
batch_normalize=1
filters=32
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[route]
layers=-1
groups=2
group_id=1

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

[route]
layers = -1,-2

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[route]
layers = -6,-1

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[route]
layers=-1
groups=2
group_id=1

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[route]
layers = -1,-2

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[route]
layers = -6,-1

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[route]
layers=-1
groups=2
group_id=1

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[route]
layers = -1,-2

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[route]
layers = -6,-1

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

##################################

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters={filters}
activation=linear

[yolo]
mask = 3,4,5
anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
classes={classes}
num=6
jitter=.3
scale_x_y = 1.05
cls_normalizer=1.0
iou_normalizer=0.07
iou_loss=ciou
ignore_thresh = .7
truth_thresh = 1
random=0
resize=1.5
nms_kind=greedynms
beta_nms=0.6

[route]
layers = -4

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[upsample]
stride=2

[route]
layers = -1, 23

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters={filters}
activation=linear

[yolo]
mask = 0,1,2
anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
classes={classes}
num=6
jitter=.3
scale_x_y = 1.05
cls_normalizer=1.0
iou_normalizer=0.07
iou_loss=ciou
ignore_thresh = .7
truth_thresh = 1
random=0
resize=1.5
nms_kind=greedynms
beta_nms=0.6
"""


def make_cfg(n, a):
    mb = max(6000, n * 2000)
    return YOLOV4_TINY_CFG.format(classes=n, filters=(n + 5) * 3, max_batches=mb,
                                  steps=f"{int(mb * .8)},{int(mb * .9)}", width=a.width, height=a.height,
                                  batch=a.batch, subdivisions=a.subdivisions)


def write_meta(out, names, a):
    """obj.names, obj.data и cfg — пишутся сразу при старте."""
    (out / "obj.names").write_text("\n".join(names) + "\n", encoding="utf-8")
    (out / "obj.data").write_text(f"classes = {len(names)}\ntrain = data/train.txt\nvalid = data/valid.txt\n"
                                  f"names = data/obj.names\nbackup = backup/\n", encoding="utf-8")
    (out / "yolov4-tiny-custom.cfg").write_text(make_cfg(len(names), a), encoding="utf-8")


def write_darknet(out, names, files, a):
    random.Random(a.seed).shuffle(files)
    n_val = max(1, int(len(files) * a.val)) if len(files) > 1 else 0
    valid, train = files[:n_val], files[n_val:]
    prefix = a.prefix.rstrip("/")
    (out / "obj.names").write_text("\n".join(names) + "\n", encoding="utf-8")
    (out / "train.txt").write_text("".join(f"{prefix}/{f}\n" for f in train), encoding="utf-8")
    (out / "valid.txt").write_text("".join(f"{prefix}/{f}\n" for f in valid), encoding="utf-8")
    (out / "obj.data").write_text(f"classes = {len(names)}\ntrain = data/train.txt\nvalid = data/valid.txt\n"
                                  f"names = data/obj.names\nbackup = backup/\n", encoding="utf-8")
    (out / "yolov4-tiny-custom.cfg").write_text(make_cfg(len(names), a), encoding="utf-8")
    return len(train), len(valid)


def make_review(obj, rv, names):
    """Вырезки всех рамок по папкам классов для ручной проверки."""
    import cv2
    if rv.exists():
        shutil.rmtree(rv)
    rv.mkdir(parents=True)
    folders = {}
    for i, n in enumerate(names):
        f = f"{i:02d}_{safe_name(n)}"
        folders[f] = i
        (rv / f).mkdir()
    for f in ("_delete", "_drop_frame"):
        (rv / f).mkdir()
    (rv / "README.txt").write_text(
        "Проверьте вырезки в папках классов.\n"
        "  Не тот класс   -> перенесите файл в папку нужного класса\n"
        "  Не объект      -> удалите файл (или перенесите в _delete)\n"
        "  Плохой кадр    -> перенесите в _drop_frame (кадр целиком уйдёт из датасета)\n"
        "Затем: python autolabel.py review --dataset <папка датасета>\n", encoding="utf-8")
    index = []
    for txt in sorted(obj.glob("*.txt")):
        lines = [l for l in txt.read_text().splitlines() if l.strip()]
        if not lines:
            continue
        img = imread_u(obj / f"{txt.stem}.jpg")
        if img is None:
            continue
        H, W = img.shape[:2]
        for k, l in enumerate(lines):
            c, xc, yc, bw, bh = l.split()[:5]
            c, xc, yc, bw, bh = int(c), float(xc) * W, float(yc) * H, float(bw) * W, float(bh) * H
            pw, ph = bw * 0.15 + 4, bh * 0.15 + 4
            x1, y1 = int(max(0, xc - bw / 2 - pw)), int(max(0, yc - bh / 2 - ph))
            x2, y2 = int(min(W, xc + bw / 2 + pw)), int(min(H, yc + bh / 2 + ph))
            crop = img[y1:y2, x1:x2]
            if crop.size == 0 or c >= len(names):
                continue
            key = f"{txt.stem}__{k}"
            imwrite_u(rv / f"{c:02d}_{safe_name(names[c])}" / f"{key}.jpg", crop)
            index.append(key)
    (rv / "_classes.json").write_text(json.dumps(folders, ensure_ascii=False, indent=1), encoding="utf-8")
    (rv / "_index.json").write_text(json.dumps(index), encoding="utf-8")
    return len(index)


# ============================================================================ команды

def apply_preset(a):
    for k, v in PRESETS[a.preset].items():
        if getattr(a, k, None) is None:
            setattr(a, k, v)
    if a.yoloworld and str(a.yoloworld).lower() == "none":
        a.yoloworld = None
    if a.clip_model and str(a.clip_model).lower() == "none":
        a.clip_model = None


def prepare(a):
    """Пакеты, сеть, модели. Возвращает пути к моделям."""
    ensure_packages({k: v for k, v in BASE_PACKAGES.items()})
    inject_truststore()
    apply_preset(a)
    nm = NetManager(a)
    Path(a.models).mkdir(parents=True, exist_ok=True)
    paths = {"gdino": ensure_hf_model(a.gdino_model, a.models, nm)}
    paths["clip"] = ensure_hf_model(a.clip_model, a.models, nm) if a.clip_model else None
    paths["yw"] = None
    if a.yoloworld:
        # YOLO-World при запуске сам докачивает текстовую модель через urllib —
        # ему нужен рабочий прокси в переменных среды, даже если веса уже есть
        if not a.offline and nm.gh() is None:
            log("[!] github.com недоступен — YOLO-World может не запуститься")
        try:
            paths["yw"] = ensure_yoloworld(a.yoloworld, a.models, nm)
        except Exception as e:
            log(f"[!] YOLO-World отключён: {e}. Работаю только с Grounding DINO.")
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return paths


def make_judge(a, classes):
    """VLM-судья (vlm.py): локально через Ollama или через OpenAI-совместимый API."""
    if not a.vlm or a.vlm == "off":
        return None
    try:
        import vlm as V
    except ImportError:
        log("[!] нет vlm.py рядом со скриптом — VLM выключена")
        return None
    if a.vlm == "ollama":
        url = a.vlm_url or V.OLLAMA_URL
        ok, have, err = V.ollama_status(url)
        if not ok:
            log(f"[!] Ollama не запущена ({err}). Установите с https://ollama.com и запустите — VLM выключена")
            return None
        if not any(m == a.vlm_model or m.split(":")[0] + ":latest" == a.vlm_model or
                   (":" not in a.vlm_model and m.startswith(a.vlm_model + ":")) for m in have):
            if a.offline:
                log(f"[!] модели {a.vlm_model} нет в Ollama, а скачивать запрещено (--offline) — VLM выключена")
                return None
            log(f"[↓] скачиваю {a.vlm_model} в Ollama...")
            last = [0.0]

            def prog(st, pct):
                if time.time() - last[0] > 2:
                    last[0] = time.time()
                    print(f"\r    {st} {'' if pct is None else f'{pct:5.1f}%'}      ", end="", flush=True)
            try:
                V.ollama_pull(a.vlm_model, url, prog)
                print()
            except Exception as e:
                log(f"\n[!] не удалось скачать {a.vlm_model}: {e} — VLM выключена")
                return None
        j = V.VLMJudge("ollama", a.vlm_model, classes, url, timeout=180, context=a.vlm_context)
    else:
        if not a.vlm_url:
            log("[!] для --vlm openai нужен --vlm-url (адрес API) — VLM выключена")
            return None
        j = V.VLMJudge("openai", a.vlm_model, classes, a.vlm_url, a.vlm_key, a.proxy,
                       workers=a.vlm_workers, context=a.vlm_context)
    log(f"[i] VLM-судья: {j.describe()}; проверяю {'все рамки' if a.vlm_mode == 'all' else 'сомнительные рамки'}")
    return j


def cmd_init(a):
    p = Path(a.out)
    if p.exists() and not a.force:
        sys.exit(f"[!] {p} уже есть (--force чтобы перезаписать)")
    p.write_text(EXAMPLE_YAML, encoding="utf-8")
    cfg = load_config(p)
    for c in cfg["classes"]:
        (Path(a.refs) / c["name"]).mkdir(parents=True, exist_ok=True)
    log(f"[✓] {p} создан; папки образцов: {a.refs}/<класс>/ (положите туда 5–20 вырезок, необязательно)")


def cmd_net(a):
    ensure_packages({"requests": "requests"})
    inject_truststore()
    cands = proxy_candidates(a.proxy)
    log("Найденные пути в интернет:")
    for name, p in cands:
        log(f"    {name:<28} {p or ''}")
    log("\nПроверка:")
    for ep in ([a.endpoint] if a.endpoint else HF_ENDPOINTS):
        find_route([(ep.split("//")[1], ep, f"{ep}/{PROBE_REPO}/resolve/main/config.json")], cands)
    find_route([("github.com", None, "https://github.com/ultralytics/assets/releases")], cands)


def cmd_download(a):
    paths = prepare(a)
    log("\n[✓] Модели готовы:")
    for k, v in paths.items():
        if v:
            log(f"    {k:<6} {v}")
    log(f"Папку {Path(a.models).resolve()} можно перенести на компьютер без интернета.")


def cmd_review(a):
    ds = Path(a.dataset)
    obj, rv = ds / "obj", ds / "review"
    if not (rv / "_classes.json").exists():
        sys.exit(f"[!] Нет {rv / '_classes.json'} — сначала выполните run")
    folders = json.loads((rv / "_classes.json").read_text(encoding="utf-8"))
    index = set(json.loads((rv / "_index.json").read_text()))
    names = (ds / "obj.names").read_text(encoding="utf-8").splitlines()
    loc = {}
    for d in rv.iterdir():
        if d.is_dir():
            for f in d.glob("*.jpg"):
                loc[f.stem] = d.name
    st = Counter()
    for txt in sorted(obj.glob("*.txt")):
        lines = [l for l in txt.read_text().splitlines() if l.strip()]
        new, drop = [], False
        for k, l in enumerate(lines):
            key = f"{txt.stem}__{k}"
            folder = loc.get(key)
            if folder is None:
                if key in index:
                    st["удалено рамок"] += 1
                    continue
                new.append(l)
                continue
            if folder == "_delete":
                st["удалено рамок"] += 1
                continue
            if folder == "_drop_frame":
                drop = True
                break
            ci = folders.get(folder)
            if ci is None:
                new.append(l)
                continue
            parts = l.split()
            if int(parts[0]) != ci:
                parts[0] = str(ci)
                st["исправлен класс"] += 1
            new.append(" ".join(parts))
        if drop:
            txt.unlink()
            (obj / f"{txt.stem}.jpg").unlink(missing_ok=True)
            st["удалено кадров"] += 1
            continue
        txt.write_text("".join(x + "\n" for x in new))
    files = sorted(f.name for f in obj.glob("*.jpg"))
    rep = json.loads((ds / "report.json").read_text(encoding="utf-8")) if (ds / "report.json").exists() else {}
    args = argparse.Namespace(**{**dict(val=0.15, prefix="data/obj", seed=0, width=416, height=416, batch=64,
                                        subdivisions=16), **{k: v for k, v in rep.get("args", {}).items()
                                                             if k in ("val", "prefix", "seed", "width", "height",
                                                                      "batch", "subdivisions")}})
    n_tr, n_va = write_darknet(ds, names, files, args)
    n = make_review(obj, rv, names)
    log(f"[✓] {dict(st) or 'изменений нет'}; кадров {len(files)} (train {n_tr}, valid {n_va}); вырезок {n}")


def cmd_run(a):
    if importlib.util.find_spec("torch") is None:
        sys.exit("[!] Не установлен PyTorch. Установите под свою систему: https://pytorch.org/get-started/locally/\n"
                 "    (без видеокарты: pip install torch torchvision)")
    paths = prepare(a)
    import cv2
    import numpy as np
    import torch
    from PIL import Image

    cfg = load_config(a.classes)
    classes = cfg["classes"]
    names = [c["name"] for c in classes]
    C = len(names)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    lim = [float(c.get("max_aspect") or a.max_aspect or 1e9) for c in classes]
    log(f"[i] устройство {device}; классов {C}; пресет {a.preset}; тайлы {a.tiles or 'нет'}")

    phrases = [(g, -1) for g in cfg["generic"]] + [(p, ci) for ci, c in enumerate(classes) for p in c["prompts"]]
    dets = [GroundingDinoDet(paths["gdino"], device, phrases, C, a.det_thr)]
    if paths["yw"]:
        try:
            dets.append(YoloWorldDet(paths["yw"], device, phrases, C, a.yw_conf))
            log("[i] детекторы: Grounding DINO + YOLO-World")
        except Exception as e:
            import traceback
            tb = traceback.extract_tb(e.__traceback__)
            where = " <- ".join(f"{Path(f.filename).name}:{f.lineno}" for f in tb[-3:])
            log(f"[!] YOLO-World не запустился ({e}) [{where}], работаю без него.\n"
                f"    Проверьте сеть: python autolabel.py net  (или укажите --proxy адрес:порт)")
    clf = ClipClf(paths["clip"], device, classes, cfg["reject"], a.refs, a.ref_min) if paths["clip"] else None
    judge = make_judge(a, classes)

    out = Path(a.out)
    obj = out / "obj"
    if obj.exists() and any(obj.iterdir()) and not a.append:
        sys.exit(f"[!] {obj} не пуста. Укажите другую --out или добавьте --append")
    obj.mkdir(parents=True, exist_ok=True)
    prev_dir = out / "preview"
    if a.preview:
        prev_dir.mkdir(exist_ok=True)
    write_meta(out, names, a)
    if not a.append:
        (out / "labels_meta.jsonl").unlink(missing_ok=True)
    prefix = a.prefix.rstrip("/")
    for lst in ("train.txt", "valid.txt"):
        if not a.append or not (out / lst).exists():
            (out / lst).write_text("", encoding="utf-8")
    split_rng = random.Random(a.seed + 1)
    counts = {"train": len((out / "train.txt").read_text().split()), "valid": len((out / "valid.txt").read_text().split())}

    def add_to_lists(name):
        """Кадр сразу попадает в train.txt или valid.txt — датасет готов в любой момент."""
        part = "valid" if (counts["valid"] == 0 and counts["train"] >= 5) or split_rng.random() < a.val else "train"
        with open(out / f"{part}.txt", "a", encoding="utf-8") as f:
            f.write(f"{prefix}/{name}\n")
        counts[part] += 1
    colors = [tuple(int(x) for x in np.random.RandomState(i + 7).randint(60, 255, 3)) for i in range(C)]
    rng = random.Random(a.seed)
    stats, files = Counter(), sorted(f.name for f in obj.glob("*.jpg"))
    idx = len(files)

    def class_probs(pil, fused):
        N = len(fused)
        Pd = np.ones((N, C + 1))
        for k, (_, _, pv) in enumerate(fused):
            if pv is not None:
                Pd[k, :C], Pd[k, C] = pv * 0.9, 0.1
        Pd /= Pd.sum(1, keepdims=True)
        L, W = a.w_det * np.log(Pd + 1e-6), a.w_det
        if clf is not None:
            Wd, Hd = pil.size
            crops = []
            for b, _, _ in fused:
                pw, ph = 0.1 * (b[2] - b[0]), 0.1 * (b[3] - b[1])
                crops.append(pil.crop((max(0, b[0] - pw), max(0, b[1] - ph), min(Wd, b[2] + pw), min(Hd, b[3] + ph))))
            F = clf.embed(crops)
            pt = clf.text_probs(F)
            L, W = L + a.w_text * np.log(pt + 1e-6), W + a.w_text
            pr = clf.ref_probs(F, pt)
            if pr is not None:
                L, W = L + a.w_ref * np.log(pr + 1e-6), W + a.w_ref
        L /= W
        P = np.exp(L - L.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        rs = np.zeros(N)
        if clf is not None and getattr(clf, "rej_refs", None) is not None:
            # жёсткий фильтр: рамка похожа на образец из refs/_reject сильнее, чем на образцы классов
            rs = clf.rej_sim(F)
            cs = clf.cls_ref_sim(F)
            hard = rs >= a.reject_sim
            if cs is not None:  # есть образцы классов — выбрасываем, только если на мусор похоже заметно сильнее
                hard &= rs > cs + a.reject_margin
            P[hard] = 0.0
            P[hard, C] = 1.0
        return P, rs

    def detect(pil, w, h):
        raw = []
        for mi, d in enumerate(dets):
            for b, s, pv in run_tiled(d, pil, a.tiles):
                b = np.clip(b, 0, [w, h, w, h])
                bw, bh = b[2] - b[0], b[3] - b[1]
                if bw < a.min_size or bh < a.min_size or bw * bh > a.max_area * w * h:
                    continue
                raw.append((b, s, mi, pv))
        fused = [f for f in wbf(raw, len(dets), a.fuse_iou) if f[1] >= a.det_thr]
        if not fused:
            return []
        P, RS = class_probs(pil, fused)
        B = [f[0] for f in fused]
        cbest = P[:, :C].argmax(1)
        if any(c.get("refine") == "color" for c in classes):
            rgb, changed = np.asarray(pil), []
            for i, c in enumerate(cbest):
                if classes[c].get("refine") == "color" and P[i].argmax() != C:
                    nb = refine_color(rgb, B[i])
                    if nb is not B[i]:
                        B[i] = nb
                        changed.append(i)
            if changed:  # переклассифицировать уже по табличке, без стойки и фона
                stats["рамок подогнано по цвету"] += len(changed)
                P2, RS2 = class_probs(pil, [(B[i], fused[i][1], fused[i][2]) for i in changed])
                P[changed] = P2
                RS[changed] = np.maximum(RS[changed], RS2)
                cbest = P[:, :C].argmax(1)
        keep = nms(B, list(np.array([f[1] for f in fused]) * P[:, :C].max(1)), a.iou)
        ok = [bool(P[i].argmax() != C and P[i].max() >= a.low) for i in range(len(B))]
        keep, n = resolve_nested(B, keep, ok, [lim[c] for c in cbest], a.contain)
        stats["рамок убрано: вложенные"] += n
        vlm_mark = {}
        if judge is not None and keep:
            vlm_mark = vlm_check(pil, B, P, RS, keep)
        return [{"box": B[i], "det": fused[i][1], "p": P[i], "rej": float(RS[i]), "vlm": vlm_mark.get(i)}
                for i in keep]

    def vlm_check(pil, B, P, RS, keep):
        """Сомнительные (или все) рамки — на проверку VLM; её ответ решает класс."""
        nonlocal judge
        send = []
        for i in keep:
            hard = RS[i] >= a.reject_sim and P[i, C] >= 0.999
            top = float(P[i, :C].max())
            c = int(P[i, :C].argmax())
            thr = float(classes[c].get("thr", a.accept))
            sure = P[i].argmax() != C and P[i, c] >= thr
            if hard or top < a.vlm_low:
                continue
            if a.vlm_mode == "all" or not sure:
                send.append(i)
        if not send:
            return {}
        rgb = np.asarray(pil)
        Hh, Ww = rgb.shape[:2]
        crops = []
        for i in send:
            x1, y1, x2, y2 = B[i]
            pw, ph = 0.25 * (x2 - x1) + 4, 0.25 * (y2 - y1) + 4
            cr = rgb[int(max(0, y1 - ph)):int(min(Hh, y2 + ph)), int(max(0, x1 - pw)):int(min(Ww, x2 + pw))]
            crops.append(cv2.cvtColor(np.ascontiguousarray(cr), cv2.COLOR_RGB2BGR))
        res = judge.ask_many(crops)
        marks = {}
        for i, (k, q, raw) in zip(send, res):
            if k is None:
                stats["VLM: ошибок"] += 1
                continue
            stats["VLM: проверено рамок"] += 1
            old = int(P[i].argmax())
            if k == -1:
                P[i] = 0.0
                P[i, C] = 1.0
                stats["VLM: выбросил"] += int(old != C)
                marks[i] = "none"
            else:
                thr = float(classes[k].get("thr", a.accept))
                v = min(0.99, max(q, thr + 0.02))
                P[i] = 0.0
                P[i, k] = v
                P[i, C] = 1.0 - v
                stats["VLM: подтвердил" if old == k else "VLM: сменил класс"] += 1
                marks[i] = names[k]
        if judge.fails >= 6:
            log(f"[!] VLM не отвечает 6 раз подряд ({judge.last_error}) — дальше размечаю без неё")
            judge = None
        return marks

    def finalize(frames):
        if not frames:
            return
        if a.track and len(frames) > 1:
            stats["треков с голосованием"] += vote_tracks(frames, a.track_iou)
        for fr in frames:
            jpg = obj / f"{fr['fname']}.jpg"
            acc, unc = [], 0
            for d in fr["dets"]:
                c = d.get("c", int(d["p"].argmax()))
                conf = d.get("conf", float(d["p"][c]))
                if c == C:
                    stats["рамок убрано: не объект"] += 1
                    continue
                if elong(d["box"]) > lim[c]:
                    stats["рамок убрано: форма"] += 1
                    continue
                thr = float(classes[c].get("thr", a.accept))
                if conf >= thr or (conf >= a.low and a.uncertain == "keep"):
                    acc.append((d["box"], c, conf))
                elif conf >= a.low:
                    unc += 1
                else:
                    stats["рамок убрано: низкая уверенность"] += 1
            if unc and a.uncertain == "skip":
                jpg.unlink(missing_ok=True)
                stats["кадров пропущено: сомнительные"] += 1
                continue
            stats["рамок убрано: сомнительные"] += unc
            if not acc and not fr["empty_ok"] and rng.random() >= a.keep_empty:
                jpg.unlink(missing_ok=True)
                stats["кадров без объектов"] += 1
                continue
            w, h = fr["w"], fr["h"]
            with open(obj / f"{fr['fname']}.txt", "w") as f:
                for b, c, _ in acc:
                    x1, y1, x2, y2 = (float(v) for v in b)
                    f.write(f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                            f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")
                    stats[names[c]] += 1
            with open(out / "labels_meta.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"f": fr["fname"], "conf": [round(float(c), 3) for _, _, c in acc]}) + "\n")
            if not acc:
                stats["пустых кадров (негативы)"] += 1
            files.append(jpg.name)
            add_to_lists(jpg.name)
            if a.preview and acc:
                vis = imread_u(jpg)
                for b, c, conf in acc:
                    p1, p2 = (int(b[0]), int(b[1])), (int(b[2]), int(b[3]))
                    cv2.rectangle(vis, p1, p2, colors[c], 2)
                    cv2.putText(vis, f"{names[c]} {conf:.2f}", (p1[0], max(12, p1[1] - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[c], 2)
                imwrite_u(prev_dir / jpg.name, vis)

    src_info = {}
    n_videos = 0

    def iter_frames():
        nonlocal n_videos
        paths_, groups = [], []
        for gi, s in enumerate(a.source):
            p = Path(s)
            if p.is_file():
                paths_.append(p)
                groups.append(gi)
            elif p.is_dir():
                lst = sorted((x for x in p.rglob("*") if x.suffix.lower() in IMG_EXT | VID_EXT), key=natural_key)
                paths_ += lst
                groups += [gi] * len(lst)
            else:
                log(f"[!] не найден источник {p}")
        n_videos = sum(1 for x in paths_ if x.suffix.lower() in VID_EXT)
        skip_group = None
        for p, gi in zip(paths_, groups):
            if p.suffix.lower() in IMG_EXT:
                if skip_group == gi:
                    continue
                if skip_file.exists():  # пропуск папки с картинками
                    skip_file.unlink(missing_ok=True)
                    log(f"[i] пропускаю оставшиеся картинки источника {a.source[gi]}")
                    skip_group = gi
                    continue
                img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    yield "images", p.stem, img, False
            elif p.suffix.lower() in VID_EXT:
                cap, tmp_copy = open_video(p)
                if cap is None:
                    continue
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                fps = cap.get(cv2.CAP_PROP_FPS) or 0
                vw, vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                src_info.update(name=p.name, k=src_info.get("k", 0) + 1)
                step = a.every
                if a.per_video and n_frames:
                    step = max(a.every, n_frames // a.per_video)  # равномерно по всему видео
                log(f"[i] видео {src_info['k']}/{n_videos}: {p.name} — {n_frames} кадров, {fps:.0f} fps, {vw}x{vh}, "
                    f"беру каждый {step}-й" + (f" (до {a.per_video} кадров с видео)" if a.per_video else ""))
                i, fails, got = 0, 0, 0
                while True:
                    if skip_file.exists():
                        skip_file.unlink(missing_ok=True)
                        log(f"[i] видео {p.name} пропущено по команде (взято {got} кадров) — перехожу к следующему")
                        src_info["skipped"] = True
                        break
                    if not cap.grab():
                        # битый кусок/кодек: пробуем перескочить, а не бросать видео
                        if n_frames and i < n_frames - 2 and fails < 30:
                            fails += 1
                            i += 1
                            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                            continue
                        break
                    fails = 0
                    if i % step == 0:
                        ok, img = cap.retrieve()
                        if ok and img is not None:
                            got += 1
                            yield str(p), f"{p.stem}_f{i:06d}", img, True
                            if a.per_video and got >= a.per_video:
                                break
                    i += 1
                cap.release()
                if tmp_copy:
                    Path(tmp_copy).unlink(missing_ok=True)
                expect = min(n_frames // max(1, step), a.per_video or 10 ** 9)
                if src_info.pop("skipped", False):
                    pass
                elif n_frames and got < expect * 0.8:
                    log(f"[!] {p.name}: прочитано только {got} из ~{expect} нужных кадров — видео повреждено или "
                        f"кодек не поддерживается OpenCV (перекодируйте в H.264, например через ffmpeg/HandBrake)")
                else:
                    log(f"[i] видео {p.name} закончено: прочитано {i} кадров, отдано в разметку {got}")

    buf, cur_seq, prev_small, seen, t0, last, found_total = [], None, None, 0, time.time(), 0.0, 0
    state = {"last": 0.0}
    def progress(found, force=False):
        nonlocal_last = state["last"]
        if not force and time.time() - nonlocal_last <= 10 and seen != 1:
            return
        state["last"] = time.time()
        guess = []
        for d in sorted(found, key=lambda d: -float(d["p"].max())):
            c = int(d["p"].argmax())
            r = d.get("rej", 0)
            mark = ""
            if r > 0:
                mark = f" (сходство с _reject {r:.2f}{' → ВЫБРОШЕНО' if c == C and r >= a.reject_sim else ''})"
            if d.get("vlm"):
                mark += " ← VLM"
            guess.append(f"{'не объект' if c == C else names[c]} {float(d['p'][c]):.2f}{mark}")
        log(f"[кадр {seen}] {(state['last'] - t0) / seen:.1f} с/кадр | на кадре: {', '.join(guess) or 'ничего'}\n"
            f"           сохранено кадров: {len(files)} | " +
            ", ".join(f"{n}={stats[n]}" for n in names if stats[n]) +
            (f" (итоги по классам — после каждых {a.window} кадров)" if a.track and not files else ""))

    stop_file = out / "_STOP"
    stop_file.unlink(missing_ok=True)
    skip_file = out / "_SKIP"  # создать файл — пропустить текущее видео
    skip_file.unlink(missing_ok=True)
    total = 0
    for s_ in a.source:
        p_ = Path(s_)
        for f_ in ([p_] if p_.is_file() else sorted(x for x in p_.rglob("*")) if p_.is_dir() else []):
            if f_.suffix.lower() in IMG_EXT:
                total += 1
            elif f_.suffix.lower() in VID_EXT:
                cap_ = cv2.VideoCapture(str(f_))
                nf_ = int(cap_.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                total += min(nf_ // max(1, a.every) + 1, a.per_video or 10 ** 9)
                cap_.release()
    if a.max_frames:
        total = min(total, a.max_frames) if total else a.max_frames

    def live(bgr, found, name, state="running"):
        """Текущий кадр со ВСЕМИ рамками и их судьбой + состояние — для приложения."""
        if not a.live:
            return
        info = {"state": state, "seen": seen, "total": total, "saved": len(files),
                "spf": round((time.time() - t0) / max(seen, 1), 2), "frame": name,
                "video": src_info.get("name", ""), "video_k": src_info.get("k", 0), "videos": n_videos,
                "stats": {k: v for k, v in stats.items()}, "classes": names, "boxes": []}
        if bgr is not None:
            vis = bgr.copy()
            for d in found:
                c = int(d["p"].argmax())
                conf = float(d["p"][c])
                b = [int(v) for v in d["box"]]
                vm = " [VLM]" if d.get("vlm") else ""
                if c == C:
                    st, col, txt = "reject", (140, 140, 140), ((f"trash {d.get('rej', 0):.2f}"
                                                                if d.get("rej", 0) >= a.reject_sim else
                                                                f"not object {conf:.2f}") + vm)
                elif elong(d["box"]) > lim[c]:
                    st, col, txt = "shape", (140, 140, 140), f"bad shape ({names[c]})"
                elif conf >= float(classes[c].get("thr", a.accept)):
                    st, col, txt = "ok", colors[c], f"{names[c]} {conf:.2f}{vm}"
                elif conf >= a.low:
                    st, col, txt = "unsure", (0, 215, 255), f"? {names[c]} {conf:.2f}{vm}"
                else:
                    st, col, txt = "low", (140, 140, 140), f"weak {names[c]} {conf:.2f}"
                txt = txt.encode("ascii", "replace").decode()
                info["boxes"].append({"box": b, "status": st, "label": txt})
                th = 2 if st in ("ok", "unsure") else 1
                cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), col, th)
                (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                ty = b[1] - 4 if b[1] > tht + 6 else b[3] + tht + 4
                cv2.rectangle(vis, (b[0], ty - tht - 3), (b[0] + tw + 4, ty + 3), col, -1)
                cv2.putText(vis, txt, (b[0] + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            k = 960 / max(vis.shape[:2])
            if k < 1:
                vis = cv2.resize(vis, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            if imwrite_u(out / "_live.tmp.jpg", vis, 85):
                replace_retry(out / "_live.tmp.jpg", out / "_live.jpg")
        (out / "_live.tmp.json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
        replace_retry(out / "_live.tmp.json", out / "_live.json")

    log(f"[i] начинаю разметку, примерно кадров: {total or '?'} "
        f"(остановить: Ctrl+C — всё размеченное сохранится)")
    stopped = False
    try:
        for seq, name, bgr, is_video in iter_frames():
            if seq != cur_seq:
                finalize(buf)
                buf, cur_seq, prev_small = [], seq, None
            if a.max_frames and len(files) >= a.max_frames:
                break
            if stop_file.exists():
                log("[i] остановлено из приложения — сохраняю размеченное...")
                stopped = True
                break
            seen += 1
            if is_video and a.dedup > 0:
                small = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (64, 36)).astype(np.float32)
                if prev_small is not None and np.abs(small - prev_small).mean() < a.dedup:
                    stats["кадров пропущено: повторы"] += 1
                    continue
                prev_small = small
            h, w = bgr.shape[:2]
            if a.max_side and max(h, w) > a.max_side:
                k = a.max_side / max(h, w)
                bgr = cv2.resize(bgr, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
                h, w = bgr.shape[:2]
            pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            found = detect(pil, w, h)
            found_total += len(found)
            try:
                live(bgr, found, name)
            except Exception as e:
                log(f"[!] live: {e}")
            empty_ok = False
            if not found:
                if rng.random() >= a.keep_empty:
                    stats["кадров без объектов"] += 1
                    progress(found)
                    continue
                empty_ok = True
            fname = f"{idx:06d}_{safe_name(name)}"
            idx += 1
            imwrite_u(obj / f"{fname}.jpg", bgr)
            buf.append({"fname": fname, "w": w, "h": h, "dets": found, "empty_ok": empty_ok})
            if not is_video or not a.track or len(buf) >= a.window:
                finalize(buf)
                buf = []
            progress(found)
    except KeyboardInterrupt:
        log("\n[i] остановлено — сохраняю то, что успел разметить...")
        stopped = True
    finalize(buf)
    stop_file.unlink(missing_ok=True)
    live(None, [], "", "stopped" if stopped else "done")

    if not files:
        sys.exit("[!] Ничего не сохранено. Попробуйте --det-thr 0.15, --accept 0.4 или проверьте источник/промпты")
    n_tr, n_va = counts["train"], counts["valid"]
    n_rev = make_review(obj, out / "review", names) if a.review else 0

    log(f"\n[✓] кадров {len(files)} (train {n_tr}, valid {n_va}); просмотрено {seen}")
    for n in names:
        warn = "   ← мало, нужно больше данных" if stats[n] < a.warn_min else ""
        log(f"    {n:<28} {stats[n]}{warn}")
    for k, v in stats.items():
        if k not in names and v:
            log(f"    ({k}: {v})")
    (out / "report.json").write_text(json.dumps(
        {"args": {k: v for k, v in vars(a).items() if k != "func"}, "classes": names,
         "frames": len(files), "train": n_tr, "valid": n_va, "stats": dict(stats)},
        ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if a.review:
        log(f"\nПроверка: {out / 'review'} ({n_rev} вырезок) → исправьте и выполните "
            f"python autolabel.py review --dataset {out}")
    log(f"Обучение: скопируйте содержимое {out}/ в darknet/data/, cfg в darknet/cfg/ и запустите\n"
        f"  darknet detector train data/obj.data cfg/yolov4-tiny-custom.cfg yolov4-tiny.conv.29 -map")


# ============================================================================ CLI

def main():
    if sys.platform == "win32":
        for s in (sys.stdout, sys.stderr):
            try:
                s.reconfigure(encoding="utf-8")
            except Exception:
                pass

    net = argparse.ArgumentParser(add_help=False)
    g = net.add_argument_group("сеть и модели")
    g.add_argument("--models", default=str(Path(__file__).resolve().parent / "models"),
                   help="папка для моделей (по умолчанию models рядом со скриптом)")
    g.add_argument("--proxy", help="прокси вручную, например 10.255.201.171:8080")
    g.add_argument("--no-proxy", action="store_true", help="не использовать прокси вообще (прямое подключение)")
    g.add_argument("--endpoint", help="сайт моделей: https://huggingface.co или https://hf-mirror.com")
    g.add_argument("--offline", action="store_true", help="ничего не качать")
    g.add_argument("--preset", choices=list(PRESETS), default="balanced")
    g.add_argument("--gdino-model", help="id на HuggingFace или папка")
    g.add_argument("--clip-model", help="id на HuggingFace, папка или none")
    g.add_argument("--yoloworld", help="веса YOLO-World (yolov8x-worldv2.pt, ...) или none")

    ap = argparse.ArgumentParser(description="Универсальная авторазметка датасета для Darknet YOLOv4-tiny",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="создать пример classes.yaml")
    p.add_argument("--out", default="classes.yaml")
    p.add_argument("--refs", default="refs")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("net", parents=[net], help="проверить сеть и прокси")
    p.set_defaults(func=cmd_net)

    p = sub.add_parser("download", parents=[net], help="только скачать модели")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("review", help="применить ручную проверку вырезок")
    p.add_argument("--dataset", default="dataset")
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("run", parents=[net], help="разметить", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source", nargs="+", required=True, help="видео, картинки или папки")
    p.add_argument("--classes", required=True, help="YAML/JSON с классами (python autolabel.py init)")
    p.add_argument("--out", default="dataset")
    p.add_argument("--append", action="store_true", help="дописывать в существующий датасет")
    p.add_argument("--refs", default="refs", help="папка с образцами refs/<класс>/")
    p.add_argument("--device", help="cuda / cpu (по умолчанию авто)")
    q = p.add_argument_group("кадры")
    q.add_argument("--every", type=int, default=1, help="каждый N-й кадр видео (1 — все)")
    q.add_argument("--dedup", type=float, default=0, help="отсев почти одинаковых кадров, например 3 (0 — выкл.)")
    q.add_argument("--max-frames", type=int, default=0, help="сколько кадров сохранить всего (0 — все)")
    q.add_argument("--per-video", type=int, default=0,
                   help="сколько кадров взять с КАЖДОГО видео, равномерно по всей длине (0 — все по --every)")
    q.add_argument("--max-side", type=int, default=1280, help="уменьшать кадры крупнее")
    q.add_argument("--keep-empty", type=float, default=0.05, help="доля кадров без объектов (негативы)")
    q = p.add_argument_group("поиск")
    q.add_argument("--tiles", type=int, help="нарезка кадра N x N для мелких объектов (0 — нет)")
    q.add_argument("--det-thr", type=float, default=0.25, help="порог Grounding DINO")
    q.add_argument("--yw-conf", type=float, default=0.05, help="порог YOLO-World")
    q.add_argument("--fuse-iou", type=float, default=0.55, help="IoU слияния рамок разных детекторов")
    q.add_argument("--min-size", type=int, default=10, help="мин. сторона рамки, px")
    q.add_argument("--max-area", type=float, default=0.6, help="макс. доля площади кадра")
    q.add_argument("--iou", type=float, default=0.5, help="порог NMS")
    q.add_argument("--contain", type=float, default=0.8, help="порог вложенности рамок (0 — выкл.)")
    q.add_argument("--max-aspect", type=float, default=0, help="макс. вытянутость для всех классов (0 — нет)")
    q = p.add_argument_group("классификация")
    q.add_argument("--accept", type=float, default=0.55, help="уверенность, с которой рамка принимается")
    q.add_argument("--low", type=float, default=0.30, help="ниже — рамка отбрасывается, между — сомнительная")
    q.add_argument("--uncertain", choices=["skip", "drop", "keep"], default="skip",
                   help="сомнительные: skip — пропустить кадр, drop — убрать рамку, keep — оставить")
    q.add_argument("--w-det", type=float, default=0.5, help="вес голоса детектора")
    q.add_argument("--w-text", type=float, default=1.0, help="вес CLIP по тексту")
    q.add_argument("--w-ref", type=float, default=2.0, help="вес CLIP по образцам")
    q.add_argument("--ref-min", type=float, default=0.75, help="мин. сходство с образцами")
    q.add_argument("--reject-sim", type=float, default=0.90,
                   help="сходство с refs/_reject, при котором рамка выбрасывается (меньше — строже)")
    q.add_argument("--reject-margin", type=float, default=0.03,
                   help="на сколько сходство с _reject должно превышать сходство с образцами классов")
    v = p.add_argument_group("VLM-судья (визуальная языковая модель)")
    v.add_argument("--vlm", choices=["off", "ollama", "openai"], default="off",
                   help="ollama — локально; openai — любой OpenAI-совместимый API (агрегаторы)")
    v.add_argument("--vlm-model", default="qwen3-vl:4b", help="модель: qwen3-vl:4b, openai/gpt-4o-mini, ...")
    v.add_argument("--vlm-url", help="адрес: http://localhost:11434 или https://api.proxyapi.ru/v1 и т.п.")
    v.add_argument("--vlm-key", default=os.environ.get("AUTOLABEL_VLM_KEY", ""),
                   help="ключ API (лучше через переменную AUTOLABEL_VLM_KEY)")
    v.add_argument("--vlm-mode", choices=["uncertain", "all"], default="uncertain",
                   help="uncertain — только сомнительные рамки (быстро), all — все")
    v.add_argument("--vlm-low", type=float, default=0.15, help="рамки слабее этого VLM не показываем")
    v.add_argument("--vlm-workers", type=int, default=4, help="параллельных запросов к API")
    v.add_argument("--vlm-context", default="", help="подсказка о сцене, напр. 'Russian road signs on a toy track'")
    q.add_argument("--track", action="store_true",
                   help="голосование за класс по треку (точнее, но метки пишутся пачками по --window кадров)")
    q.add_argument("--track-iou", type=float, default=0.3)
    q.add_argument("--window", type=int, default=40, help="кадров в окне трекинга (метки пишутся по окнам)")
    q = p.add_argument_group("выход")
    q.add_argument("--preview", action="store_true", help="картинки с рамками в preview/")
    q.add_argument("--live", action="store_true", help="писать _live.jpg/_live.json для приложения")
    q.add_argument("--no-review", dest="review", action="store_false", help="не делать вырезки для проверки")
    q.add_argument("--val", type=float, default=0.15, help="доля valid")
    q.add_argument("--prefix", default="data/obj", help="путь к картинкам в train.txt относительно darknet")
    q.add_argument("--width", type=int, default=416)
    q.add_argument("--height", type=int, default=416)
    q.add_argument("--batch", type=int, default=64)
    q.add_argument("--subdivisions", type=int, default=16)
    q.add_argument("--warn-min", type=int, default=100, help="предупреждать, если объектов класса меньше")
    q.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_run)

    a = ap.parse_args()
    if getattr(a, "no_proxy", False) or NO_PROXY_MODE:
        disable_proxy()
        a.proxy = None
    if getattr(a, "cmd", None) == "run" and (a.width % 32 or a.height % 32):
        sys.exit("[!] --width/--height должны быть кратны 32")
    a.func(a)


if __name__ == "__main__":
    main()
