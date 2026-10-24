#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video_to_frames.py — нарезка видео на кадры (картинки) для датасета.

Примеры:
  python video_to_frames.py video.mp4                       каждый кадр -> frames/video/
  python video_to_frames.py video.mp4 --every 10            каждый 10-й кадр
  python video_to_frames.py video.mp4 --fps 2               2 кадра в секунду видео
  python video_to_frames.py video.mp4 --count 300           300 кадров равномерно по всему видео
  python video_to_frames.py папка_с_видео --fps 1 --flat    все видео из папки в одну папку
  python video_to_frames.py video.mp4 --start 0:30 --end 2:15 --fps 5
  python video_to_frames.py video.mp4 --fps 3 --dedup 3 --blur 60 --max-side 1280

Можно просто перетащить видео (или папку) на CUT_VIDEO.bat.

Фильтры:
  --dedup N   пропускать почти одинаковые кадры (машина стоит). 2–5 — разумно, 0 — выкл.
  --blur N    пропускать смазанные кадры (резкость меньше N). 50–100 — разумно, 0 — выкл.
"""
import argparse
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".ts", ".3gp", ".flv"}


def natural_key(p):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(p))]


def parse_time(s):
    """'90', '1:30', '0:01:30', '1:30.5' -> секунды"""
    if s is None or s == "":
        return None
    parts = [float(x) for x in str(s).replace(",", ".").split(":")]
    sec = 0.0
    for x in parts:
        sec = sec * 60 + x
    return sec


def fmt_time(sec):
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def open_video(p):
    """VideoCapture, переживающий русские буквы в пути на Windows (через короткое имя 8.3)."""
    cap = cv2.VideoCapture(str(p))
    if cap.isOpened():
        return cap
    if sys.platform == "win32":
        try:
            import ctypes
            buf = ctypes.create_unicode_buffer(1024)
            if ctypes.windll.kernel32.GetShortPathNameW(str(p), buf, 1024):
                cap = cv2.VideoCapture(buf.value)
                if cap.isOpened():
                    return cap
        except Exception:
            pass
    return None


def imwrite_u(path, img, ext, quality):
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext == ".jpg" else [cv2.IMWRITE_PNG_COMPRESSION, 3]
    ok, buf = cv2.imencode(ext, img, params)
    if ok:
        buf.tofile(str(path))
    return ok


def collect(sources):
    out = []
    for s in sources:
        p = Path(s.strip('"'))
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out += sorted((x for x in p.rglob("*") if x.suffix.lower() in VID_EXT), key=natural_key)
        else:
            print(f"[!] не найдено: {p}")
    return out


def cut(p, a, out_root):
    cap = open_video(p)
    if cap is None:
        print(f"[!] не открылось: {p} (файл битый или кодек не поддерживается — перекодируйте в H.264)")
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if fps <= 0 or fps > 1000:
        fps = 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur = n / fps if n else 0
    t0 = parse_time(a.start) or 0.0
    t1 = parse_time(a.end)
    f0 = int(t0 * fps)
    f1 = min(n, int(t1 * fps)) if (t1 and n) else (int(t1 * fps) if t1 else n)
    span = max(0, (f1 or 0) - f0)

    # какие кадры брать
    targets = None
    if a.count:
        if span:
            targets = set(np.linspace(f0, max(f0, f1 - 1), num=min(a.count, span)).round().astype(int).tolist())
        rule = f"{a.count} кадров равномерно"
    elif a.fps:
        step = fps / a.fps
        rule = f"{a.fps:g} кадр/с (каждый ~{step:.1f}-й)"
    else:
        step = a.every
        rule = f"каждый {a.every}-й кадр"

    stem = re.sub(r"[^\w\-]+", "_", p.stem)
    out_dir = out_root if a.flat else out_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = ".png" if a.png else ".jpg"
    print(f"\n[i] {p.name}: {n} кадров, {fps:.1f} fps, {w}x{h}, {fmt_time(dur)}"
          + (f", отрезок {fmt_time(t0)}–{fmt_time(f1 / fps) if f1 else 'конец'}" if a.start or a.end else ""))
    print(f"    беру: {rule} -> {out_dir}")

    if f0 > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    i, saved, skip_dup, skip_blur, fails = f0, 0, 0, 0, 0
    next_t = float(f0)
    prev = None
    t_start, last_print = time.time(), 0.0
    while True:
        if f1 and i >= f1:
            break
        if not cap.grab():
            if n and i < n - 2 and fails < 30:  # битый кусок — перескочить
                fails += 1
                i += 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                continue
            break
        fails = 0
        take = (i in targets) if targets is not None else (i >= next_t - 1e-6)
        if take:
            if targets is None:
                next_t += step
            ok, img = cap.retrieve()
            if ok and img is not None:
                if a.max_side and max(img.shape[:2]) > a.max_side:
                    k = a.max_side / max(img.shape[:2])
                    img = cv2.resize(img, (round(img.shape[1] * k), round(img.shape[0] * k)), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                keep = True
                if a.blur > 0 and cv2.Laplacian(gray, cv2.CV_64F).var() < a.blur:
                    skip_blur += 1
                    keep = False
                if keep and a.dedup > 0:
                    small = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
                    if prev is not None and np.abs(small - prev).mean() < a.dedup:
                        skip_dup += 1
                        keep = False
                    else:
                        prev = small
                if keep:
                    name = f"{a.prefix}{stem}_{i:06d}{ext}"
                    if imwrite_u(out_dir / name, img, ext, a.quality):
                        saved += 1
                        if a.limit and saved >= a.limit:
                            break
        i += 1
        if time.time() - last_print > 1:
            last_print = time.time()
            pct = 100 * (i - f0) / span if span else 0
            print(f"\r    {pct:5.1f}%  кадр {i}/{f1 or n}  сохранено {saved}   ", end="", flush=True)
    cap.release()
    extra = []
    if skip_dup:
        extra.append(f"одинаковых пропущено {skip_dup}")
    if skip_blur:
        extra.append(f"смазанных пропущено {skip_blur}")
    print(f"\r    готово: сохранено {saved} кадров за {time.time() - t_start:.0f} с"
          + (f" ({', '.join(extra)})" if extra else "") + " " * 20)
    if n and i < f0 + span * 0.9 and not a.limit:
        print(f"    [!] прочитано только до кадра {i} из {f1 or n} — видео повреждено или кодек не поддерживается")
    return saved


def main():
    if sys.platform == "win32":
        for s in (sys.stdout, sys.stderr):
            try:
                s.reconfigure(encoding="utf-8")
            except Exception:
                pass
    ap = argparse.ArgumentParser(description="Нарезка видео на кадры",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("source", nargs="*", help="видео или папки с видео")
    ap.add_argument("--out", default="frames", help="куда сохранять (по умолчанию frames/)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--every", type=int, default=1, help="каждый N-й кадр (по умолчанию 1 — все)")
    g.add_argument("--fps", type=float, help="сколько кадров на секунду видео")
    g.add_argument("--count", type=int, help="сколько кадров всего с каждого видео, равномерно")
    ap.add_argument("--start", help="с какого времени, например 0:30")
    ap.add_argument("--end", help="до какого времени, например 2:15")
    ap.add_argument("--limit", type=int, default=0, help="не больше N сохранённых кадров с видео")
    ap.add_argument("--dedup", type=float, default=0, help="пропуск почти одинаковых кадров (2–5), 0 — выкл.")
    ap.add_argument("--blur", type=float, default=0, help="пропуск смазанных кадров (50–100), 0 — выкл.")
    ap.add_argument("--max-side", type=int, default=0, help="уменьшить, если сторона больше (0 — как есть)")
    ap.add_argument("--png", action="store_true", help="сохранять в PNG (без потерь, больше места)")
    ap.add_argument("--quality", type=int, default=95, help="качество JPG 1–100")
    ap.add_argument("--flat", action="store_true", help="все кадры в одну папку (без подпапок по видео)")
    ap.add_argument("--prefix", default="", help="приставка к именам файлов")
    a = ap.parse_args()

    srcs = a.source
    if not srcs:  # запустили без аргументов (двойной клик) — спросим
        try:
            s = input("Перетащите сюда видео или папку и нажмите Enter: ").strip().strip('"')
        except EOFError:
            s = ""
        if not s:
            ap.print_help()
            return
        srcs = [s]
        f = input("Сколько кадров в секунду брать? (Enter — 2): ").strip()
        a.fps = float(f.replace(",", ".")) if f else 2.0

    videos = collect(srcs)
    if not videos:
        sys.exit("[!] Видео не найдены")
    out = Path(a.out)
    total = 0
    for k, v in enumerate(videos, 1):
        print(f"\n=== {k}/{len(videos)} ===", end="")
        total += cut(v, a, out)
    print(f"\n[✓] Всего сохранено {total} кадров в {out.resolve()}")


if __name__ == "__main__":
    main()
