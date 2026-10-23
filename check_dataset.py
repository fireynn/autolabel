#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_dataset.py — проверка готового датасета Darknet (obj/*.jpg + obj/*.txt + obj.names).

Режимы:
  python check_dataset.py stats  --dataset dataset     статистика и поиск ошибок в разметке
  python check_dataset.py draw   --dataset dataset     все кадры с нарисованными рамками -> dataset/check/frames/
                                                        + листы вырезок по классам -> dataset/check/sheets/
  python check_dataset.py view   --dataset dataset     просмотр в окне:
        → / D / пробел  следующий      ← / A  предыдущий
        Del / X        удалить кадр из датасета (картинка + txt + строки в train/valid)
        F              показывать только кадры с выбранным классом (по кругу)
        E              только пустые кадры (без рамок)
        Q / Esc        выход

Нужно только: pip install opencv-python numpy
"""
import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def colors(n):
    rs = np.random.RandomState(7)
    return [tuple(int(x) for x in rs.randint(60, 255, 3)) for _ in range(max(n, 1))]


def imread(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def imwrite(p, img):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(Path(p).suffix or ".jpg", img)
    if ok:
        buf.tofile(str(p))


def load(ds):
    ds = Path(ds)
    obj = ds / "obj" if (ds / "obj").is_dir() else ds
    names_f = ds / "obj.names" if (ds / "obj.names").exists() else obj / "obj.names"
    if not names_f.exists():
        sys.exit(f"[!] Нет obj.names в {ds}")
    names = [l.strip() for l in names_f.read_text(encoding="utf-8").splitlines() if l.strip()]
    imgs = sorted(p for p in obj.iterdir() if p.suffix.lower() in IMG_EXT)
    return ds, obj, names, imgs


def read_labels(img_path, n_cls):
    """-> (список (cls, xc, yc, w, h), список ошибок)"""
    txt = img_path.with_suffix(".txt")
    if not txt.exists():
        return [], ["нет .txt"]
    boxes, errs = [], []
    for i, line in enumerate(txt.read_text().splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            errs.append(f"строка {i}: {len(parts)} чисел вместо 5")
            continue
        try:
            c = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])
        except ValueError:
            errs.append(f"строка {i}: не числа")
            continue
        if not 0 <= c < n_cls:
            errs.append(f"строка {i}: класс {c} вне 0..{n_cls - 1}")
        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
            errs.append(f"строка {i}: координаты вне 0..1")
        boxes.append((c, xc, yc, w, h))
    return boxes, errs


def to_px(b, W, H):
    c, xc, yc, w, h = b
    return c, int((xc - w / 2) * W), int((yc - h / 2) * H), int((xc + w / 2) * W), int((yc + h / 2) * H)


def draw(img, boxes, names, cols):
    H, W = img.shape[:2]
    t = max(1, round(min(W, H) / 400))
    for b in boxes:
        c, x1, y1, x2, y2 = to_px(b, W, H)
        col = cols[c % len(cols)]
        name = names[c] if 0 <= c < len(names) else f"?{c}"
        cv2.rectangle(img, (x1, y1), (x2, y2), col, t + 1)
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t, t)
        ty = y1 - 4 if y1 - th - 6 > 0 else y2 + th + 4
        cv2.rectangle(img, (x1, ty - th - 3), (x1 + tw + 4, ty + 3), col, -1)
        cv2.putText(img, name, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t, (0, 0, 0), t)
    return img


# ============================================================================ stats

def cmd_stats(a):
    ds, obj, names, imgs = load(a.dataset)
    cnt, frames_with, sizes = Counter(), Counter(), defaultdict(list)
    empty, bad, tiny, dup = [], [], [], []
    for p in imgs:
        boxes, errs = read_labels(p, len(names))
        if errs:
            bad.append((p.name, errs))
        if not boxes:
            empty.append(p.name)
        seen = set()
        for b in boxes:
            c = b[0]
            cnt[c] += 1
            seen.add(c)
            sizes[c].append((b[3], b[4]))
            if b[3] * 416 < 6 or b[4] * 416 < 6:
                tiny.append((p.name, names[c] if c < len(names) else c))
        for c in seen:
            frames_with[c] += 1
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if boxes[i][1:] == boxes[j][1:]:
                    dup.append(p.name)
    orphan = [t.name for t in obj.glob("*.txt") if not any((obj / (t.stem + e)).exists() for e in IMG_EXT)]

    lists = {}
    for lst in ("train.txt", "valid.txt"):
        f = ds / lst
        if f.exists():
            rows = [r.strip() for r in f.read_text(encoding="utf-8").splitlines() if r.strip()]
            missing = [r for r in rows if not (obj / Path(r).name).exists()]
            lists[lst] = (len(rows), missing)

    print(f"\nДатасет: {ds.resolve()}")
    print(f"Кадров: {len(imgs)}   пустых (негативы): {len(empty)}   рамок: {sum(cnt.values())}\n")
    print(f"  {'ID':>2}  {'класс':<28} {'рамок':>6} {'кадров':>7}   средний размер (от кадра)")
    for i, n in enumerate(names):
        s = sizes[i]
        avg = f"{np.mean([w for w, _ in s]) * 100:4.1f}% x {np.mean([h for _, h in s]) * 100:4.1f}%" if s else "-"
        warn = "   ← нет совсем" if not cnt[i] else ("   ← мало (<100)" if cnt[i] < 100 else "")
        print(f"  {i:>2}  {n:<28} {cnt[i]:>6} {frames_with[i]:>7}   {avg}{warn}")
    for k in cnt:
        if not 0 <= k < len(names):
            print(f"  ??  класс {k} (нет в obj.names) {cnt[k]}")
    if cnt:
        mx, mn = max(cnt[i] for i in range(len(names))), min(cnt[i] for i in range(len(names)))
        if mn and mx / mn > 10:
            print(f"\n[!] Сильный перекос классов: {mx} vs {mn} — редкие классы будут распознаваться хуже")
    for lst, (n, missing) in lists.items():
        print(f"\n{lst}: {n} строк" + (f", {len(missing)} ссылаются на несуществующие файлы" if missing else ""))
    if len(empty) > 0.3 * max(len(imgs), 1):
        print(f"\n[!] Пустых кадров {len(empty)} из {len(imgs)} — больше 30%, это много")
    for title, items in (("Ошибки в .txt", bad), ("Совпадающие рамки", dup), ("Очень мелкие рамки (<6px при 416)", tiny),
                         (".txt без картинки", orphan)):
        if items:
            print(f"\n[!] {title}: {len(items)}")
            for it in items[:10]:
                print(f"     {it}")
            if len(items) > 10:
                print("     ...")
    if not (bad or dup or orphan):
        print("\n[✓] Ошибок формата нет")


# ============================================================================ draw

def cmd_draw(a):
    ds, obj, names, imgs = load(a.dataset)
    out = Path(a.out) if a.out else ds / "check"
    cols = colors(len(names))
    crops = defaultdict(list)
    for k, p in enumerate(imgs):
        img = imread(p)
        if img is None:
            continue
        boxes, errs = read_labels(p, len(names))
        H, W = img.shape[:2]
        for b in boxes:
            c, x1, y1, x2, y2 = to_px(b, W, H)
            pw, ph = int((x2 - x1) * 0.15) + 3, int((y2 - y1) * 0.15) + 3
            cr = img[max(0, y1 - ph):min(H, y2 + ph), max(0, x1 - pw):min(W, x2 + pw)]
            if cr.size:
                crops[c].append((cr.copy(), p.stem))
        vis = draw(img, boxes, names, cols)
        if errs:
            cv2.putText(vis, "ОШИБКА: " + errs[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        sub = "empty" if not boxes else "frames"
        if a.scale != 1:
            vis = cv2.resize(vis, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
        imwrite(out / sub / f"{p.stem}.jpg", vis)
        if (k + 1) % 200 == 0:
            print(f"  {k + 1}/{len(imgs)}")

    # листы вырезок: одна картинка = много объектов одного класса, ошибки видно сразу
    cell, cols_n = a.cell, a.grid
    for c, items in crops.items():
        name = names[c] if 0 <= c < len(names) else f"unknown_{c}"
        if a.max_sheet and len(items) > a.max_sheet * cols_n * cols_n:
            items = random.Random(0).sample(items, a.max_sheet * cols_n * cols_n)
        per = cols_n * cols_n
        for s in range(0, len(items), per):
            chunk = items[s:s + per]
            rows = (len(chunk) + cols_n - 1) // cols_n
            sheet = np.full((rows * (cell + 14), cols_n * cell, 3), 40, np.uint8)
            for i, (cr, stem) in enumerate(chunk):
                h, w = cr.shape[:2]
                k = min(cell / w, cell / h)
                im = cv2.resize(cr, (max(1, int(w * k)), max(1, int(h * k))))
                r, q = divmod(i, cols_n)
                y0, x0 = r * (cell + 14), q * cell
                sheet[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
                cv2.putText(sheet, stem.split("_")[0], (x0 + 2, y0 + cell + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                            (200, 200, 200), 1)
            imwrite(out / "sheets" / f"{c:02d}_{name.replace(' ', '_')}_{s // per + 1:03d}.jpg", sheet)
    print(f"[✓] {out / 'frames'} — кадры с рамками; {out / 'empty'} — пустые кадры")
    print(f"[✓] {out / 'sheets'} — листы вырезок по классам (подпись = номер кадра)")
    print("    Нашли ошибку — откройте кадр: python check_dataset.py view --dataset", a.dataset,
          "--start <номер кадра>")


# ============================================================================ view

def remove_frame(ds, p):
    p.unlink(missing_ok=True)
    p.with_suffix(".txt").unlink(missing_ok=True)
    for lst in ("train.txt", "valid.txt"):
        f = ds / lst
        if f.exists():
            rows = f.read_text(encoding="utf-8").splitlines()
            keep = [r for r in rows if Path(r.strip()).name != p.name]
            if len(keep) != len(rows):
                f.write_text("".join(r + "\n" for r in keep), encoding="utf-8")


def cmd_view(a):
    ds, obj, names, imgs = load(a.dataset)
    if not imgs:
        sys.exit("[!] Нет картинок")
    cols = colors(len(names))
    labels = {p: read_labels(p, len(names))[0] for p in imgs}
    flt = None  # None | int (класс) | "empty"
    i = 0
    if a.start:
        for k, p in enumerate(imgs):
            if a.start in p.stem:
                i = k
                break
    win = "check_dataset  (<- -> листать, Del удалить кадр, F фильтр класса, E пустые, Q выход)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def visible():
        if flt is None:
            return imgs
        if flt == "empty":
            return [p for p in imgs if not labels[p]]
        return [p for p in imgs if any(b[0] == flt for b in labels[p])]

    deleted = 0
    while True:
        vis_list = visible()
        if not vis_list:
            print("[i] нет кадров под фильтр — сбрасываю")
            flt, vis_list = None, imgs
            if not imgs:
                break
        i %= len(vis_list)
        p = vis_list[i]
        img = imread(p)
        if img is None:
            i += 1
            continue
        draw(img, labels[p], names, cols)
        fl = "все" if flt is None else ("пустые" if flt == "empty" else names[flt])
        info = f"{i + 1}/{len(vis_list)}  {p.name}  рамок: {len(labels[p])}  фильтр: {fl}"
        cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(img, info, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imshow(win, img)
        k = cv2.waitKeyEx(0)
        if k in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if k in (2555904, 65363, ord("d"), ord("D"), 32):          # →
            i += 1
        elif k in (2424832, 65361, ord("a"), ord("A")):            # ←
            i -= 1
        elif k in (3014656, 65535, ord("x"), ord("X")):            # Del
            remove_frame(ds, p)
            imgs.remove(p)
            labels.pop(p, None)
            deleted += 1
            print(f"  удалён {p.name}")
            if not imgs:
                break
        elif k in (ord("f"), ord("F")):
            flt = 0 if not isinstance(flt, int) else (flt + 1 if flt + 1 < len(names) else None)
            i = 0
        elif k in (ord("e"), ord("E")):
            flt = None if flt == "empty" else "empty"
            i = 0
    cv2.destroyAllWindows()
    if deleted:
        print(f"[✓] удалено кадров: {deleted}")


def main():
    if sys.platform == "win32":
        for s in (sys.stdout, sys.stderr):
            try:
                s.reconfigure(encoding="utf-8")
            except Exception:
                pass
    ap = argparse.ArgumentParser(description="Проверка датасета Darknet",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("stats", help="статистика и ошибки")
    p.add_argument("--dataset", default="dataset")
    p.set_defaults(func=cmd_stats)
    p = sub.add_parser("draw", help="нарисовать рамки на всех кадрах + листы вырезок")
    p.add_argument("--dataset", default="dataset")
    p.add_argument("--out", help="куда (по умолчанию dataset/check)")
    p.add_argument("--scale", type=float, default=1.0, help="масштаб картинок, например 0.5")
    p.add_argument("--cell", type=int, default=96, help="размер вырезки на листе, px")
    p.add_argument("--grid", type=int, default=10, help="вырезок в ряду (лист grid x grid)")
    p.add_argument("--max-sheet", type=int, default=0, help="макс. листов на класс (0 — все)")
    p.set_defaults(func=cmd_draw)
    p = sub.add_parser("view", help="просмотр в окне")
    p.add_argument("--dataset", default="dataset")
    p.add_argument("--start", help="начать с кадра (часть имени)")
    p.set_defaults(func=cmd_view)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
