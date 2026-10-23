#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
install.py — установка Autolabel Studio «в один клик» (запускается из START.bat).

Что делает (прямое подключение, без прокси):
  1. создаёт окружение .venv рядом со скриптом
  2. ставит PyTorch под вашу видеокарту (NVIDIA → CUDA 12.6 или 11.8 по версии драйвера, иначе CPU)
  3. ставит остальные пакеты (transformers, opencv, PyQt5, ...)
  4. скачивает модели выбранного пресета в папку models
  5. создаёт ярлык «Autolabel Studio» на рабочем столе

Повторный запуск ничего не качает заново — только докачивает недостающее.
Параметры:  --preset fast|balanced|best   --cpu (не ставить CUDA)   --reinstall-torch   --dry-run
"""
import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV = HERE / ".venv"
WIN = sys.platform == "win32"
VPY = VENV / ("Scripts/python.exe" if WIN else "bin/python")
MARK = VENV / ".installed_ok"

TORCH_VER, VISION_VER = "2.7.1", "0.22.1"
PACKAGES = ["transformers>=4.45,<5", "opencv-python>=4.8", "pillow", "pyyaml", "numpy", "requests", "PyQt5"]
PROXY_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "PIP_PROXY")

DRY = False


def say(msg):
    print(msg, flush=True)


def clean_env():
    """Без прокси: убираем переменные и запрещаем системный прокси Windows."""
    env = dict(os.environ)
    for k in PROXY_KEYS:
        env.pop(k, None)
    env["NO_PROXY"] = env["no_proxy"] = "*"
    env["AUTOLABEL_NO_PROXY"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run(cmd, tries=1, check=True):
    say("   > " + " ".join(str(c) for c in cmd))
    if DRY:
        return 0
    for t in range(1, tries + 1):
        code = subprocess.call([str(c) for c in cmd], env=clean_env(), cwd=str(HERE))
        if code == 0:
            return 0
        if t < tries:
            say(f"   ! не получилось (код {code}), повтор {t + 1}/{tries} через 10 с...")
            time.sleep(10)
    if check:
        say(f"\n[ОШИБКА] Команда не выполнилась. Проверьте интернет и запустите START.bat ещё раз —\n"
            f"         уже скачанное повторно не качается.")
        sys.exit(1)
    return code


def pip(*args, tries=3):
    return run([VPY, "-m", "pip", "install", "--timeout", "60", "--retries", "10", *args], tries=tries)


def detect_gpu():
    """-> (имя, версия драйвера) или (None, None)"""
    exe = shutil.which("nvidia-smi")
    if not exe and WIN and Path(r"C:\Windows\System32\nvidia-smi.exe").exists():
        exe = r"C:\Windows\System32\nvidia-smi.exe"
    if not exe:
        return None, None
    try:
        out = subprocess.check_output([exe, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                      timeout=20).decode(errors="ignore").strip().splitlines()[0]
        name, drv = [x.strip() for x in out.split(",")[:2]]
        return name, drv
    except Exception:
        return None, None


def torch_variant(drv, force_cpu):
    """CUDA 12.x требует драйвер Windows ≥ 528.33 (Linux ≥ 525.60), CUDA 11.8 — ≥ 452.39 (Linux ≥ 450.80)."""
    if force_cpu or not drv:
        return "cpu"
    try:
        major = float(".".join(drv.split(".")[:2]))
    except ValueError:
        return "cpu"
    if major >= (528.33 if WIN else 525.60):
        return "cu126"
    if major >= (452.39 if WIN else 450.80):
        return "cu118"
    return "cpu"


def installed_torch():
    if DRY or not VPY.exists():
        return None
    try:
        out = subprocess.check_output([str(VPY), "-c", "import torch;print(torch.__version__, torch.version.cuda)"],
                                      env=clean_env(), stderr=subprocess.DEVNULL, timeout=120).decode().split()
        return out[0], (out[1] if len(out) > 1 else "None")
    except Exception:
        return None


def make_shortcut():
    if not WIN or DRY:
        return
    bat = HERE / "START.bat"
    ps = ("$d=[Environment]::GetFolderPath('Desktop');"
          "$s=(New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $d 'Autolabel Studio.lnk'));"
          f"$s.TargetPath='{bat}';$s.WorkingDirectory='{HERE}';$s.WindowStyle=7;$s.Save()")
    try:
        subprocess.call(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        say("   ярлык «Autolabel Studio» создан на рабочем столе")
    except Exception:
        pass


def main():
    global DRY
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="balanced", choices=["fast", "balanced", "best"])
    ap.add_argument("--cpu", action="store_true", help="ставить PyTorch без CUDA")
    ap.add_argument("--reinstall-torch", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="только показать, что будет сделано")
    a = ap.parse_args()
    DRY = a.dry_run

    say("=" * 64)
    say("  Установка Autolabel Studio")
    say("=" * 64)
    say(f"Папка: {HERE}")
    say(f"Python: {sys.version.split()[0]} ({platform.architecture()[0]})")
    if sys.version_info < (3, 9):
        say("[ОШИБКА] Нужен Python 3.9–3.12 (START.bat ставит 3.11 сам)")
        sys.exit(1)
    for f in ("autolabel.py", "autolabel_app.py"):
        if not (HERE / f).exists():
            say(f"[ОШИБКА] Рядом нет {f} — положите все файлы в одну папку")
            sys.exit(1)

    # 1. окружение
    say("\n[1/5] Окружение .venv")
    if VPY.exists() and not DRY:
        try:
            subprocess.check_call([str(VPY), "-c", "import sys"], stdout=subprocess.DEVNULL)
            say("   уже есть")
        except Exception:
            say("   повреждено — пересоздаю")
            shutil.rmtree(VENV, ignore_errors=True)
    if not VPY.exists():
        run([sys.executable, "-m", "venv", VENV])
    run([VPY, "-m", "pip", "install", "--upgrade", "pip", "--timeout", "60"], tries=3)

    # 2. PyTorch
    say("\n[2/5] PyTorch")
    name, drv = detect_gpu()
    variant = torch_variant(drv, a.cpu)
    if name:
        say(f"   видеокарта: {name}, драйвер {drv}")
    else:
        say("   видеокарта NVIDIA не найдена — будет работать на процессоре (медленно)")
    if name and variant == "cpu" and not a.cpu:
        say("   ! драйвер слишком старый для CUDA — обновите драйвер с nvidia.com и запустите снова")
    say(f"   сборка: {variant}")
    have = installed_torch()
    want_cuda = {"cu126": "12.6", "cu118": "11.8", "cpu": "None"}[variant]
    if have and have[0].startswith(TORCH_VER) and have[1] == want_cuda and not a.reinstall_torch:
        say(f"   уже установлен torch {have[0]} (CUDA {have[1]})")
    else:
        if have:
            run([VPY, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"], check=False)
        say("   скачиваю (для CUDA это ~2.5 ГБ, наберитесь терпения)...")
        pip(f"torch=={TORCH_VER}", f"torchvision=={VISION_VER}",
            "--index-url", f"https://download.pytorch.org/whl/{variant}")

    # 3. пакеты
    say("\n[3/5] Остальные пакеты")
    pip(*PACKAGES)

    # проверка
    if not DRY:
        code = ("import torch, transformers, cv2, PyQt5;"
                "ok=torch.cuda.is_available();"
                "print('   torch', torch.__version__, '| transformers', transformers.__version__, '| OpenCV', cv2.__version__);"
                "print('   видеокарта:', torch.cuda.get_device_name(0) if ok else 'нет (CPU)')")
        if run([VPY, "-c", code], check=False):
            say("[ОШИБКА] Пакеты установились с ошибкой — см. выше")
            sys.exit(1)

    # 4. модели
    say(f"\n[4/5] Модели (пресет {a.preset})")
    run([VPY, str(HERE / "autolabel.py"), "download", "--preset", a.preset, "--no-proxy"], tries=3)

    # 5. ярлык
    say("\n[5/5] Ярлык")
    make_shortcut()
    if not DRY:
        MARK.write_text(f"{time.ctime()} torch={variant} preset={a.preset}\n", encoding="utf-8")
    say("\n✔ Готово. Дальше просто запускайте START.bat или ярлык на рабочем столе.\n")


if __name__ == "__main__":
    main()
