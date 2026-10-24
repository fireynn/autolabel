#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autolabel Studio — оконное приложение для autolabel.py:
  1. Классы      — классы, фразы, описания, образцы (refs) и мусор (_reject) с картинками
  2. Разметка    — пресеты и настройки, запуск/остановка, живое превью со всеми рамками, лог
  3. Чистка      — сетка вырезок по классам: удалить, сменить класс, в мусор, в образцы,
                   выделить похожие, сортировка по уверенности, отмена (Ctrl+Z)
  4. Кадры       — редактор: рисовать/двигать/удалять рамки, менять класс, листать кадры
  5. Статистика  — классы, ошибки, перекос, train/valid, cfg, корзина
  6. Обучение    — сборка dataset.zip и инструкция для Google Colab
  7. Сеть        — проверка прокси/сети, скачивание моделей, проверка видеокарты

Запуск:  python autolabel_app.py      (autolabel.py должен лежать рядом)
"""
import json
import os
import random
import shutil
import sys
import time
import zipfile
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
SETTINGS_FILE = APP_DIR / "autolabel_app.json"


def no_proxy_enabled():
    """По умолчанию прокси НЕ используется (прямое подключение)."""
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8")).get("net", {}).get("no_proxy", True)
    except Exception:
        return True


if no_proxy_enabled():
    os.environ["AUTOLABEL_NO_PROXY"] = "1"
try:
    import autolabel as AL
except ImportError:
    sys.exit("[!] Рядом с autolabel_app.py должен лежать autolabel.py")

if AL.NO_PROXY_MODE:
    AL.disable_proxy()
AL.ensure_packages({"PyQt5": "PyQt5", "cv2": "opencv-python", "numpy": "numpy", "yaml": "pyyaml"})

import cv2  # noqa: E402
try:
    import vlm as VLM  # noqa: E402
except ImportError:
    VLM = None
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PyQt5.QtCore import (QPoint, QPointF, QProcess, QProcessEnvironment, QRectF, QSize, Qt, QThread,  # noqa: E402
                          QTimer, pyqtSignal)
from PyQt5.QtGui import (QBrush, QColor, QDesktopServices, QFont, QIcon, QImage, QKeySequence, QPainter,  # noqa: E402
                         QPalette, QPen, QPixmap, QTextCursor)
from PyQt5.QtCore import QUrl  # noqa: E402
from PyQt5.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox,  # noqa: E402
                             QFileDialog, QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout,
                             QHeaderView, QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                             QMainWindow, QMenu, QMessageBox, QPlainTextEdit, QProgressBar,
                             QProgressDialog, QPushButton, QScrollArea, QShortcut, QSizePolicy, QSlider,
                             QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QTextBrowser,
                             QToolButton, QVBoxLayout, QWidget)

IMG_EXT = AL.IMG_EXT
VID_EXT = AL.VID_EXT
PRESET_INFO = {
    "fast": "Быстро: Grounding DINO tiny + CLIP base. Для слабого ПК и проверки.",
    "balanced": "Баланс: Grounding DINO tiny + CLIP large. Обычный выбор.",
    "best": "Точно: DINO base + YOLO-World + CLIP large + нарезка на тайлы. Медленно, нужна видеокарта.",
}


# ============================================================================ общие функции

def class_bgr(i):
    return tuple(int(x) for x in np.random.RandomState(i + 7).randint(60, 255, 3))


def class_qcolor(i):
    b, g, r = class_bgr(i)
    return QColor(r, g, b)


def bgr_to_qimage(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def crop_box(img, xc, yc, w, h, pad=0.15):
    H, W = img.shape[:2]
    bw, bh = w * W, h * H
    x1, y1 = int(max(0, xc * W - bw / 2 - bw * pad - 3)), int(max(0, yc * H - bh / 2 - bh * pad - 3))
    x2, y2 = int(min(W, xc * W + bw / 2 + bw * pad + 3)), int(min(H, yc * H + bh / 2 + bh * pad + 3))
    return img[y1:y2, x1:x2]


def thumb_square(bgr, size):
    h, w = bgr.shape[:2]
    k = size / max(h, w, 1)
    im = cv2.resize(bgr, (max(1, int(w * k)), max(1, int(h * k))),
                    interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    out = np.full((size, size, 3), 45, np.uint8)
    y0, x0 = (size - im.shape[0]) // 2, (size - im.shape[1]) // 2
    out[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
    return out


def feature(bgr):
    """Маленький «отпечаток» вырезки для поиска похожих (без нейросетей)."""
    lab = cv2.cvtColor(cv2.resize(bgr, (20, 20), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2LAB)
    v = lab.astype(np.float32).ravel()
    v -= v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def load_settings():
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_settings(d):
    try:
        SETTINGS_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass


def hline():
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setFrameShadow(QFrame.Sunken)
    return f


def button(text, slot=None, tip=None, shortcut=None):
    b = QPushButton(text)
    if slot:
        b.clicked.connect(slot)
    if tip:
        b.setToolTip(tip)
    if shortcut:
        b.setShortcut(QKeySequence(shortcut))
    return b


def open_folder(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        os.startfile(str(path))  # noqa
    elif sys.platform == "darwin":
        QProcess.startDetached("open", [str(path)])
    else:
        QProcess.startDetached("xdg-open", [str(path)])


# ============================================================================ модель датасета

class Box:
    __slots__ = ("c", "xc", "yc", "w", "h", "conf", "alive")

    def __init__(self, c, xc, yc, w, h, conf=None, alive=True):
        self.c, self.xc, self.yc, self.w, self.h, self.conf, self.alive = c, xc, yc, w, h, conf, alive

    def tup(self):
        return (self.c, self.xc, self.yc, self.w, self.h, self.conf, self.alive)

    def key(self):
        return f"{self.xc:.4f},{self.yc:.4f},{self.w:.4f},{self.h:.4f}"


class Frame:
    __slots__ = ("stem", "img", "boxes", "errors")

    def __init__(self, stem, img):
        self.stem, self.img, self.boxes, self.errors = stem, img, [], []

    def alive(self):
        return [b for b in self.boxes if b.alive]


class Dataset:
    def __init__(self, root):
        self.root = Path(root)
        self.obj = self.root / "obj"
        self.trash = self.root / "_trash"
        self.names, self.frames = [], OrderedDict()
        self._cache = OrderedDict()
        self.version = 0

    # ---------------------------------------------------------------- загрузка/сохранение
    def load(self):
        nf = self.root / "obj.names"
        self.names = [l.strip() for l in nf.read_text(encoding="utf-8").splitlines() if l.strip()] \
            if nf.exists() else []
        meta = {}
        mf = self.root / "labels_meta.jsonl"
        if mf.exists():
            for line in mf.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                    meta[r["f"]] = r.get("conf") or []
                except Exception:
                    pass
        self.frames.clear()
        self._cache.clear()
        if not self.obj.is_dir():
            return
        for img in sorted(p for p in self.obj.iterdir() if p.suffix.lower() in IMG_EXT):
            fr = Frame(img.stem, img)
            txt = img.with_suffix(".txt")
            confs = meta.get(img.stem, [])
            if txt.exists():
                for k, line in enumerate(txt.read_text().splitlines()):
                    parts = line.split()
                    if not parts:
                        continue
                    try:
                        c = int(parts[0])
                        xc, yc, w, h = map(float, parts[1:5])
                        if len(parts) != 5:
                            raise ValueError
                    except ValueError:
                        fr.errors.append(f"строка {k + 1}: неверный формат")
                        continue
                    if not 0 <= c < max(len(self.names), 1):
                        fr.errors.append(f"строка {k + 1}: класс {c} вне списка")
                    if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        fr.errors.append(f"строка {k + 1}: координаты вне 0..1")
                    conf = confs[len(fr.boxes)] if len(fr.boxes) < len(confs) else None
                    fr.boxes.append(Box(c, xc, yc, w, h, conf))
            else:
                fr.errors.append("нет .txt")
            self.frames[img.stem] = fr
        self.version += 1

    def save(self, stems):
        for s in stems:
            fr = self.frames.get(s)
            if fr is None:
                continue
            with open(fr.img.with_suffix(".txt"), "w") as f:
                for b in fr.alive():
                    f.write(f"{b.c} {b.xc:.6f} {b.yc:.6f} {b.w:.6f} {b.h:.6f}\n")
        self.write_meta()
        self.version += 1

    def write_meta(self):
        try:
            with open(self.root / "labels_meta.jsonl", "w", encoding="utf-8") as f:
                for s, fr in self.frames.items():
                    f.write(json.dumps({"f": s, "conf": [b.conf for b in fr.alive()]}) + "\n")
        except Exception:
            pass

    def snapshot(self, stems):
        return {s: [b.tup() for b in self.frames[s].boxes] for s in stems if s in self.frames}

    def restore(self, snap):
        for s, boxes in snap.items():
            if s in self.frames:
                self.frames[s].boxes = [Box(*t) for t in boxes]
        self.save(list(snap))

    # ---------------------------------------------------------------- кадры
    def _lists(self, fn):
        for lst in ("train.txt", "valid.txt"):
            f = self.root / lst
            if f.exists():
                rows = f.read_text(encoding="utf-8").splitlines()
                new = fn(lst, rows)
                if new is not None:
                    f.write_text("".join(r + "\n" for r in new), encoding="utf-8")

    def delete_frames(self, stems):
        """В корзину _trash (можно вернуть)."""
        self.trash.mkdir(exist_ok=True)
        moved = []
        names = set()
        for s in stems:
            fr = self.frames.pop(s, None)
            if fr is None:
                continue
            names.add(fr.img.name)
            for p in (fr.img, fr.img.with_suffix(".txt")):
                if p.exists():
                    shutil.move(str(p), str(self.trash / p.name))
            moved.append(fr)
        removed = {}

        def fn(lst, rows):
            keep = [r for r in rows if Path(r.strip()).name not in names]
            for r in rows:
                if Path(r.strip()).name in names:
                    removed[Path(r.strip()).name] = (lst, r)
            return keep
        self._lists(fn)
        self.write_meta()
        self.version += 1
        return moved, removed

    def restore_frames(self, moved, removed):
        for fr in moved:
            for p in (fr.img, fr.img.with_suffix(".txt")):
                t = self.trash / p.name
                if t.exists():
                    shutil.move(str(t), str(p))
            self.frames[fr.stem] = fr
        self.frames = OrderedDict(sorted(self.frames.items()))
        for name, (lst, row) in removed.items():
            f = self.root / lst
            with open(f, "a", encoding="utf-8") as fh:
                fh.write(row + "\n")
        self.save([fr.stem for fr in moved])

    def rebuild_lists(self, val, prefix="data/obj", seed=0):
        files = [fr.img.name for fr in self.frames.values()]
        random.Random(seed).shuffle(files)
        n_val = max(1, int(len(files) * val)) if len(files) > 1 else 0
        prefix = prefix.rstrip("/")
        (self.root / "valid.txt").write_text("".join(f"{prefix}/{f}\n" for f in files[:n_val]), encoding="utf-8")
        (self.root / "train.txt").write_text("".join(f"{prefix}/{f}\n" for f in files[n_val:]), encoding="utf-8")
        return len(files) - n_val, n_val

    def image(self, stem):
        if stem in self._cache:
            self._cache.move_to_end(stem)
            return self._cache[stem]
        fr = self.frames.get(stem)
        img = AL.imread_u(fr.img) if fr else None
        if img is not None:
            self._cache[stem] = img
            while len(self._cache) > 40:
                self._cache.popitem(last=False)
        return img

    def count(self):
        cnt, frames_with = Counter(), Counter()
        for fr in self.frames.values():
            seen = set()
            for b in fr.alive():
                cnt[b.c] += 1
                seen.add(b.c)
            for c in seen:
                frames_with[c] += 1
        return cnt, frames_with


# ============================================================================ фоновые миниатюры

class ThumbWorker(QThread):
    ready = pyqtSignal(str, QImage, object)

    def __init__(self, jobs, size):
        super().__init__()
        self.jobs, self.size, self._stop = jobs, size, False

    def stop(self):
        self._stop = True

    def run(self):
        cache = OrderedDict()
        for key, path, (xc, yc, w, h) in sorted(self.jobs, key=lambda j: str(j[1])):
            if self._stop:
                return
            img = cache.get(path)
            if img is None:
                img = AL.imread_u(path)
                if img is None:
                    continue
                cache[path] = img
                while len(cache) > 8:
                    cache.popitem(last=False)
            cr = crop_box(img, xc, yc, w, h)
            if cr.size == 0:
                continue
            self.ready.emit(key, bgr_to_qimage(thumb_square(cr, self.size)), feature(cr))


class Task(QThread):
    """Выполнить функцию в фоне: fn(report) ; report(text) -> сигнал msg."""
    msg = pyqtSignal(str)
    item = pyqtSignal(object)
    done = pyqtSignal(object)

    def __init__(self, fn):
        super().__init__()
        self.fn, self.stop_flag = fn, False

    def run(self):
        try:
            res = self.fn(self)
        except Exception as e:
            res = e
        self.done.emit(res)


def normalize_classes(raw, names=None):
    """Классы из таблицы/yaml -> формат autolabel (prompts/desc — списки), в порядке names."""
    out = []
    by = {}
    for c in raw:
        c = dict(c)
        pr = c.get("prompts") or c.get("prompt") or c["name"]
        c["prompts"] = [pr] if isinstance(pr, str) else list(pr)
        d = c.get("desc") or []
        c["desc"] = [d] if isinstance(d, str) else list(d)
        by[c["name"]] = c
        out.append(c)
    if names:
        out = [by.get(n, {"name": n, "prompts": [n.replace("_", " ")], "desc": []}) for n in names]
    return out


# ============================================================================ запуск процессов

class Console(QPlainTextEdit):
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setMaximumBlockCount(8000)
        f = QFont("Consolas" if sys.platform == "win32" else "Monospace")
        f.setStyleHint(QFont.Monospace)
        f.setPointSize(9)
        self.setFont(f)

    def add(self, text):
        self.moveCursor(QTextCursor.End)
        self.insertPlainText(text)
        self.moveCursor(QTextCursor.End)


class Runner(QWidget):
    """QProcess + консоль. Запускает python autolabel.py <args>."""
    finished = pyqtSignal(int)
    line = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.console = Console()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.console)
        self.proc = None
        self._buf = ""

    def running(self):
        return self.proc is not None and self.proc.state() != QProcess.NotRunning

    def start(self, args, script="autolabel.py"):
        if self.running():
            return False
        self.proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUNBUFFERED", "1")
        key = os.environ.get("AUTOLABEL_VLM_KEY", "")
        if key:
            env.insert("AUTOLABEL_VLM_KEY", key)
        if AL.NO_PROXY_MODE:
            for k in AL.PROXY_KEYS:
                env.remove(k)
            env.insert("NO_PROXY", "*")
            env.insert("no_proxy", "*")
            env.insert("AUTOLABEL_NO_PROXY", "1")
            env.insert("PIP_PROXY", "")
        else:
            env.remove("AUTOLABEL_NO_PROXY")
        self.proc.setProcessEnvironment(env)
        self.proc.setWorkingDirectory(str(APP_DIR))
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(lambda code, _st: self._done(code))
        full = ["-u", str(APP_DIR / script)] + args if script else args
        self.console.add(f"\n$ python {' '.join(a if ' ' not in a else repr(a) for a in full[1:])}\n")
        self.proc.start(sys.executable, full)
        return True

    def _read(self):
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace").replace("\r\n", "\n")
        # строки прогресса загрузки с \r — оставляем последнюю версию
        parts = data.split("\r")
        for i, p in enumerate(parts):
            if i > 0:
                cur = self.console.textCursor()
                cur.movePosition(QTextCursor.End)
                cur.movePosition(QTextCursor.StartOfBlock, QTextCursor.KeepAnchor)
                cur.removeSelectedText()
            self.console.add(p)
        self._buf += data
        while "\n" in self._buf:
            l, self._buf = self._buf.split("\n", 1)
            self.line.emit(l)

    def _done(self, code):
        self._read()
        self.console.add(f"\n[процесс завершён, код {code}]\n")
        self.finished.emit(code)

    def kill(self):
        if self.running():
            self.proc.kill()


# ============================================================================ 1. Классы

class ClassesTab(QWidget):
    COLS = ["Имя класса", "Фраза для детектора (англ.)", "Описание для CLIP (англ.)", "Вытянутость",
            "По цвету", "Свой порог", "Образцов"]

    def __init__(self, main):
        super().__init__()
        self.main = main
        self.path = None
        self.extra = {}
        lay = QVBoxLayout(self)

        top = QHBoxLayout()
        self.path_lbl = QLabel("—")
        self.path_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        top.addWidget(QLabel("<b>Файл классов:</b>"))
        top.addWidget(self.path_lbl, 1)
        top.addWidget(button("Открыть...", self.open_dialog))
        top.addWidget(button("Сохранить", self.save, shortcut="Ctrl+S"))
        top.addWidget(button("Сохранить как...", self.save_as))
        top.addWidget(button("Пример (знаки)", self.new_example))
        lay.addLayout(top)

        split = QSplitter(Qt.Vertical)
        up = QWidget()
        ul = QHBoxLayout(up)
        ul.setContentsMargins(0, 0, 0, 0)
        tl = QVBoxLayout()
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Interactive)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.Stretch)
        for i in (3, 4, 5, 6):
            hh.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self.table.setColumnWidth(0, 190)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.itemSelectionChanged.connect(self.on_select)
        self.table.setToolTip("Порядок строк = номер класса в Darknet (0, 1, 2...).\n"
                              "Несколько фраз/описаний разделяйте знаком |")
        tl.addWidget(self.table)
        tb = QHBoxLayout()
        tb.addWidget(button("+ Класс", self.add_row))
        tb.addWidget(button("− Удалить", self.del_row))
        tb.addWidget(button("↑", lambda: self.move_row(-1)))
        tb.addWidget(button("↓", lambda: self.move_row(1)))
        tb.addStretch()
        tb.addWidget(QLabel("Вытянутость: 1.6 для знаков, 4 для светофора, 0 — не проверять.  "
                            "«По цвету» — подогнать рамку к цветной табличке."))
        tl.addLayout(tb)
        ul.addLayout(tl, 3)

        rl = QVBoxLayout()
        rl.addWidget(QLabel("<b>Общие фразы детектора</b> (по строке):"))
        self.generic = QPlainTextEdit()
        self.generic.setToolTip("Что искать «вообще» — потом классификатор разберёт, что это")
        rl.addWidget(self.generic)
        rl.addWidget(QLabel("<b>Мусор текстом</b> (по строке):"))
        self.reject = QPlainTextEdit()
        self.reject.setToolTip("Если классификатор выбирает одно из этого — рамка выбрасывается")
        rl.addWidget(self.reject)
        ul.addLayout(rl, 1)
        split.addWidget(up)

        # образцы
        down = QWidget()
        dl = QVBoxLayout(down)
        dl.setContentsMargins(0, 0, 0, 0)
        rb = QHBoxLayout()
        rb.addWidget(QLabel("<b>Образцы для:</b>"))
        self.ref_combo = QComboBox()
        self.ref_combo.setMinimumWidth(260)
        self.ref_combo.currentIndexChanged.connect(self.show_refs)
        rb.addWidget(self.ref_combo, 1)
        rb.addWidget(button("Добавить картинки...", self.add_refs))
        rb.addWidget(button("Удалить выбранные", self.del_refs))
        rb.addWidget(button("Открыть папку", lambda: open_folder(self.ref_dir())))
        self.refs_root_lbl = QLabel()
        rb.addWidget(self.refs_root_lbl)
        rb.addWidget(button("Папка refs...", self.choose_refs_root))
        dl.addLayout(rb)
        self.ref_list = QListWidget()
        self.ref_list.setViewMode(QListWidget.IconMode)
        self.ref_list.setIconSize(QSize(80, 80))
        self.ref_list.setResizeMode(QListWidget.Adjust)
        self.ref_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.ref_list.setMovement(QListWidget.Static)
        dl.addWidget(self.ref_list)
        dl.addWidget(QLabel("Образцы — 10–20 вырезок объекта (лучше из вкладки «Чистка» кнопкой «В образцы»). "
                            "_reject — вырезки ложных срабатываний: всё похожее будет выбрасываться."))
        split.addWidget(down)
        split.setSizes([450, 250])
        lay.addWidget(split)

    # ---------------------------------------------------------------- данные
    def refs_root(self):
        return Path(self.main.settings.get("refs") or APP_DIR / "refs")

    def choose_refs_root(self):
        d = QFileDialog.getExistingDirectory(self, "Папка образцов refs", str(self.refs_root()))
        if d:
            self.main.settings["refs"] = d
            self.refresh_ref_combo()

    def ref_dir(self):
        name = self.ref_combo.currentData()
        return self.refs_root() / (name or "_reject")

    def names(self):
        out = []
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if it and it.text().strip():
                out.append(it.text().strip())
        return out

    def load(self, path):
        path = Path(path)
        try:
            cfg = AL.load_config(path)
        except SystemExit as e:
            QMessageBox.warning(self, "Классы", str(e))
            return False
        except Exception as e:
            QMessageBox.warning(self, "Классы", f"Не удалось прочитать {path}:\n{e}")
            return False
        self.path = path
        self.path_lbl.setText(str(path))
        self.main.settings["classes"] = str(path)
        self.table.setRowCount(0)
        self.extra = {}
        for c in cfg["classes"]:
            self._append(c)
        self.generic.setPlainText("\n".join(cfg["generic"]))
        self.reject.setPlainText("\n".join(cfg["reject"]))
        self.refresh_ref_combo()
        return True

    def _append(self, c):
        r = self.table.rowCount()
        self.table.insertRow(r)
        name = c.get("name", f"class{r}")
        self.table.setItem(r, 0, QTableWidgetItem(name))
        self.table.setItem(r, 1, QTableWidgetItem(" | ".join(c.get("prompts") or [])))
        self.table.setItem(r, 2, QTableWidgetItem(" | ".join(c.get("desc") or [])))
        self.table.setItem(r, 3, QTableWidgetItem(str(c.get("max_aspect") or "")))
        chk = QTableWidgetItem()
        chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        chk.setCheckState(Qt.Checked if c.get("refine") == "color" else Qt.Unchecked)
        self.table.setItem(r, 4, chk)
        self.table.setItem(r, 5, QTableWidgetItem(str(c.get("thr") or "")))
        cnt = QTableWidgetItem("")
        cnt.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        self.table.setItem(r, 6, cnt)
        known = {"name", "prompt", "prompts", "desc", "max_aspect", "refine", "thr"}
        self.extra[name] = {k: v for k, v in c.items() if k not in known}

    def to_config(self):
        classes = []
        for r in range(self.table.rowCount()):
            txt = lambda col: (self.table.item(r, col).text().strip() if self.table.item(r, col) else "")  # noqa
            name = txt(0)
            if not name:
                continue
            c = {"name": name}
            pr = [x.strip() for x in txt(1).split("|") if x.strip()]
            if pr:
                c["prompt"] = pr[0] if len(pr) == 1 else pr
            ds = [x.strip() for x in txt(2).split("|") if x.strip()]
            if ds:
                c["desc"] = ds[0] if len(ds) == 1 else ds
            for col, key in ((3, "max_aspect"), (5, "thr")):
                try:
                    v = float(txt(col).replace(",", "."))
                    if v > 0:
                        c[key] = v
                except ValueError:
                    pass
            if self.table.item(r, 4) and self.table.item(r, 4).checkState() == Qt.Checked:
                c["refine"] = "color"
            c.update(self.extra.get(name, {}))
            classes.append(c)
        lines = lambda w: [l.strip() for l in w.toPlainText().splitlines() if l.strip()]  # noqa
        return {"generic": lines(self.generic), "reject": lines(self.reject), "classes": classes}

    def save(self, path=None):
        cfg = self.to_config()
        if not cfg["classes"]:
            QMessageBox.warning(self, "Классы", "Нет ни одного класса")
            return False
        if len({c["name"] for c in cfg["classes"]}) != len(cfg["classes"]):
            QMessageBox.warning(self, "Классы", "Имена классов повторяются")
            return False
        path = Path(path or self.path or APP_DIR / "classes.yaml")
        head = ("# Классы для autolabel.py (сохранено из Autolabel Studio).\n"
                "# Порядок задаёт ID класса в Darknet (0, 1, 2...).\n\n")
        path.write_text(head + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False, width=200),
                        encoding="utf-8")
        self.path = path
        self.path_lbl.setText(str(path))
        self.main.settings["classes"] = str(path)
        self.main.status(f"Классы сохранены: {path}")
        return True

    def save_as(self):
        p, _ = QFileDialog.getSaveFileName(self, "Сохранить классы", str(self.path or APP_DIR / "classes.yaml"),
                                           "YAML (*.yaml *.yml)")
        if p:
            self.save(p)

    def open_dialog(self):
        p, _ = QFileDialog.getOpenFileName(self, "Файл классов", str(APP_DIR), "YAML/JSON (*.yaml *.yml *.json)")
        if p:
            self.load(p)

    def new_example(self):
        if self.table.rowCount() and QMessageBox.question(
                self, "Пример", "Заменить текущие классы примером дорожных знаков?") != QMessageBox.Yes:
            return
        tmp = APP_DIR / "_example_classes.yaml"
        tmp.write_text(AL.EXAMPLE_YAML, encoding="utf-8")
        old = self.path
        self.load(tmp)
        tmp.unlink(missing_ok=True)
        self.path = old or APP_DIR / "classes.yaml"
        self.main.settings["classes"] = str(self.path)
        self.path_lbl.setText(str(self.path) + "  (не сохранено)")

    def add_row(self):
        n, ok = QInputDialog.getText(self, "Новый класс", "Имя класса (латиницей, как в obj.names):")
        if ok and n.strip():
            self._append({"name": n.strip(), "prompts": [n.strip()], "desc": []})
            self.table.selectRow(self.table.rowCount() - 1)
            self.refresh_ref_combo()

    def del_row(self):
        r = self.table.currentRow()
        if r >= 0 and QMessageBox.question(self, "Удалить", "Удалить класс? Номера следующих классов сдвинутся — "
                                                            "старый датасет станет несовместим.") == QMessageBox.Yes:
            self.table.removeRow(r)
            self.refresh_ref_combo()

    def move_row(self, d):
        r = self.table.currentRow()
        t = r + d
        if r < 0 or not 0 <= t < self.table.rowCount():
            return
        for col in range(self.table.columnCount()):
            a, b = self.table.takeItem(r, col), self.table.takeItem(t, col)
            self.table.setItem(r, col, b)
            self.table.setItem(t, col, a)
        self.table.selectRow(t)
        self.refresh_ref_combo()

    def on_select(self):
        r = self.table.currentRow()
        if r >= 0 and self.table.item(r, 0):
            i = self.ref_combo.findData(self.table.item(r, 0).text().strip())
            if i >= 0:
                self.ref_combo.setCurrentIndex(i)

    # ---------------------------------------------------------------- образцы
    def refresh_ref_combo(self):
        cur = self.ref_combo.currentData()
        self.ref_combo.blockSignals(True)
        self.ref_combo.clear()
        root = self.refs_root()
        self.refs_root_lbl.setText(f"  refs: {root}")
        for r, n in enumerate(self.names()):
            d = root / n
            k = len([p for p in d.iterdir() if p.suffix.lower() in IMG_EXT]) if d.is_dir() else 0
            self.ref_combo.addItem(f"{n}   ({k})", n)
            it = self.table.item(r, 6)
            if it:
                it.setText(str(k) if k else "—")
        d = root / "_reject"
        k = len([p for p in d.iterdir() if p.suffix.lower() in IMG_EXT]) if d.is_dir() else 0
        self.ref_combo.addItem(f"_reject — мусор   ({k})", "_reject")
        i = self.ref_combo.findData(cur)
        self.ref_combo.setCurrentIndex(max(0, i))
        self.ref_combo.blockSignals(False)
        self.show_refs()

    def show_refs(self):
        self.ref_list.clear()
        d = self.ref_dir()
        if not d.is_dir():
            return
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in IMG_EXT:
                continue
            img = AL.imread_u(p)
            if img is None:
                continue
            it = QListWidgetItem(QIcon(QPixmap.fromImage(bgr_to_qimage(thumb_square(img, 80)))), "")
            it.setData(Qt.UserRole, str(p))
            it.setToolTip(p.name)
            self.ref_list.addItem(it)

    def add_refs(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Картинки-образцы", str(APP_DIR),
                                                "Картинки (*.jpg *.jpeg *.png *.bmp *.webp)")
        if not files:
            return
        d = self.ref_dir()
        d.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, d / Path(f).name)
        self.refresh_ref_combo()

    def del_refs(self):
        items = self.ref_list.selectedItems()
        if not items:
            return
        for it in items:
            Path(it.data(Qt.UserRole)).unlink(missing_ok=True)
        self.refresh_ref_combo()

    def showEvent(self, e):
        super().showEvent(e)
        self.refresh_ref_combo()


# ============================================================================ 2. Разметка

class LiveView(QLabel):
    def __init__(self):
        super().__init__("Здесь будет живое превью: каждый обработанный кадр со всеми найденными рамками")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background:#1b1b1b; color:#888; border:1px solid #333;")
        self._pix = None

    def set_image(self, img):
        self._pix = QPixmap.fromImage(img)
        self._rescale()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._rescale()

    def _rescale(self):
        if self._pix:
            self.setPixmap(self._pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


class RunTab(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.live_mtime = 0
        self.stop_sent = 0
        self.last_state, self.last_saved = None, 0
        lay = QHBoxLayout(self)
        split = QSplitter(Qt.Horizontal)
        lay.addWidget(split)

        # ---------------------------------------------------------- настройки
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        form_w = QWidget()
        fl = QVBoxLayout(form_w)
        sc.setWidget(form_w)
        sc.setMinimumWidth(400)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        g = QGroupBox("Источники: видео, фото, папки")
        gl = QVBoxLayout(g)
        self.sources = QListWidget()
        self.sources.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.sources.setMaximumHeight(110)
        gl.addWidget(self.sources)
        b = QHBoxLayout()
        b.addWidget(button("Файлы...", self.add_files))
        b.addWidget(button("Папка...", self.add_dir))
        b.addWidget(button("Убрать", lambda: [self.sources.takeItem(self.sources.row(i))
                                              for i in self.sources.selectedItems()]))
        b.addWidget(button("↑", lambda: self.move_source(-1), "Выше — раньше разметится"))
        b.addWidget(button("↓", lambda: self.move_source(1)))
        gl.addLayout(b)
        gl.addWidget(QLabel("Размечаются по порядку списка, сверху вниз."))
        fl.addWidget(g)

        g = QGroupBox("Датасет")
        gf = QFormLayout(g)
        self.out = QLineEdit()
        ob = QHBoxLayout()
        ob.addWidget(self.out)
        ob.addWidget(button("...", self.choose_out))
        gf.addRow("Папка:", ob)
        self.append = QCheckBox("Дописывать в существующий датасет")
        gf.addRow(self.append)
        fl.addWidget(g)

        g = QGroupBox("Пресет и модели")
        gf = QFormLayout(g)
        self.preset = QComboBox()
        for p in ("fast", "balanced", "best"):
            self.preset.addItem(p, p)
        self.preset_info = QLabel()
        self.preset_info.setWordWrap(True)
        self.preset.currentIndexChanged.connect(lambda: self.preset_info.setText(
            PRESET_INFO[self.preset.currentData()]))
        gf.addRow("Пресет:", self.preset)
        gf.addRow(self.preset_info)
        self.clip = QComboBox()
        for t, v in (("по пресету", ""), ("CLIP base (быстрее)", "openai/clip-vit-base-patch32"),
                     ("CLIP large (точнее)", "openai/clip-vit-large-patch14"), ("выключен", "none")):
            self.clip.addItem(t, v)
        gf.addRow("Классификатор:", self.clip)
        self.yw = QComboBox()
        for t, v in (("по пресету", ""), ("выключен", "none"), ("YOLO-World s", "yolov8s-worldv2.pt"),
                     ("YOLO-World m", "yolov8m-worldv2.pt"), ("YOLO-World l", "yolov8l-worldv2.pt"),
                     ("YOLO-World x", "yolov8x-worldv2.pt")):
            self.yw.addItem(t, v)
        gf.addRow("2-й детектор:", self.yw)
        self.tiles = QComboBox()
        for t, v in (("по пресету", ""), ("нет", "0"), ("2 x 2", "2"), ("3 x 3", "3")):
            self.tiles.addItem(t, v)
        self.tiles.setToolTip("Нарезка кадра на части — для мелких/далёких объектов, в разы медленнее")
        gf.addRow("Тайлы:", self.tiles)
        self.device = QComboBox()
        for t, v in (("авто", ""), ("видеокарта (cuda)", "cuda"), ("процессор (cpu)", "cpu")):
            self.device.addItem(t, v)
        gf.addRow("Устройство:", self.device)
        self.vlm_lbl = QLabel("VLM: выключена (вкладка 8)")
        vb = QHBoxLayout()
        vb.addWidget(self.vlm_lbl, 1)
        vb.addWidget(button("Настроить", lambda: self.main.tabs.setCurrentWidget(self.main.vlm_tab)))
        gf.addRow("Судья:", vb)
        fl.addWidget(g)

        g = QGroupBox("Кадры")
        gf = QFormLayout(g)
        self.every = QSpinBox()
        self.every.setRange(1, 1000)
        self.every.setToolTip("Брать каждый N-й кадр видео. 1 — все кадры")
        gf.addRow("Каждый N-й кадр:", self.every)
        self.per_video = QSpinBox()
        self.per_video.setRange(0, 1_000_000)
        self.per_video.setSingleStep(50)
        self.per_video.setSpecialValueText("все (по шагу)")
        self.per_video.setToolTip("Сколько кадров взять с КАЖДОГО видео — равномерно по всей длине.\n"
                                  "Видео размечаются по порядку списка, каждое целиком.")
        gf.addRow("Кадров с каждого видео:", self.per_video)
        self.max_side = QSpinBox()
        self.max_side.setRange(0, 8000)
        self.max_side.setSingleStep(160)
        self.max_side.setToolTip("Уменьшать кадры крупнее этого (быстрее). 0 — не уменьшать")
        gf.addRow("Макс. сторона кадра, px:", self.max_side)
        self.dedup = QDoubleSpinBox()
        self.dedup.setRange(0, 50)
        self.dedup.setSingleStep(0.5)
        self.dedup.setSpecialValueText("выкл.")
        self.dedup.setToolTip("Пропускать почти одинаковые соседние кадры. 3 — разумно, 0 — выключено")
        gf.addRow("Отсев одинаковых:", self.dedup)
        self.keep_empty = QDoubleSpinBox()
        self.keep_empty.setRange(0, 1)
        self.keep_empty.setSingleStep(0.05)
        self.keep_empty.setToolTip("Доля кадров без объектов, которые попадут в датасет как негативы")
        gf.addRow("Доля пустых кадров:", self.keep_empty)
        fl.addWidget(g)

        g = QGroupBox("Точность")
        gf = QFormLayout(g)
        self.det_thr = QDoubleSpinBox()
        self.det_thr.setRange(0.05, 0.9)
        self.det_thr.setSingleStep(0.05)
        self.det_thr.setToolTip("Ниже — находит больше (и больше мусора)")
        gf.addRow("Порог детектора:", self.det_thr)
        self.accept = QDoubleSpinBox()
        self.accept.setRange(0.1, 0.99)
        self.accept.setSingleStep(0.05)
        self.accept.setToolTip("С такой уверенностью рамка принимается")
        gf.addRow("Принимать от:", self.accept)
        self.low = QDoubleSpinBox()
        self.low.setRange(0.0, 0.9)
        self.low.setSingleStep(0.05)
        self.low.setToolTip("Ниже — рамка выбрасывается; между этим и «Принимать от» — сомнительная")
        gf.addRow("Сомнительные от:", self.low)
        self.uncertain = QComboBox()
        for t, v in (("пропускать кадр целиком", "skip"), ("убирать только рамку", "drop"),
                     ("оставлять рамку", "keep")):
            self.uncertain.addItem(t, v)
        gf.addRow("Сомнительные:", self.uncertain)
        self.reject_sim = QDoubleSpinBox()
        self.reject_sim.setRange(0.5, 0.99)
        self.reject_sim.setSingleStep(0.01)
        self.reject_sim.setToolTip("Сходство с refs/_reject, от которого рамка считается мусором. Меньше — строже")
        gf.addRow("Мусор (_reject) от:", self.reject_sim)
        self.track = QCheckBox("Голосование по треку (точнее классы, метки пачками)")
        gf.addRow(self.track)
        self.preview = QCheckBox("Сохранять preview/ с рамками")
        gf.addRow(self.preview)
        fl.addWidget(g)

        g = QGroupBox("Дополнительно")
        gf = QFormLayout(g)
        self.extra = QLineEdit()
        self.extra.setPlaceholderText("например: --min-size 8 --contain 0.7")
        gf.addRow("Аргументы:", self.extra)
        fl.addWidget(g)
        fl.addStretch()
        split.addWidget(sc)

        # ---------------------------------------------------------- работа
        right = QWidget()
        rl = QVBoxLayout(right)
        bar = QHBoxLayout()
        self.btn_start = button("▶  Запустить разметку", self.start)
        self.btn_start.setStyleSheet("QPushButton{font-weight:bold; padding:6px 14px;}")
        self.btn_skip = button("⏭  Следующее видео", self.skip,
                               "Бросить текущее видео (размеченное с него сохранится) и перейти к следующему")
        self.btn_stop = button("■  Остановить", self.stop, "Мягко: доделать текущий кадр и сохранить всё")
        self.btn_kill = button("✖  Прервать", self.kill, "Сразу убить процесс (последние кадры могут потеряться)")
        self.btn_stop.setEnabled(False)
        self.btn_kill.setEnabled(False)
        self.btn_skip.setEnabled(False)
        bar.addWidget(self.btn_start)
        bar.addWidget(self.btn_skip)
        bar.addWidget(self.btn_stop)
        bar.addWidget(self.btn_kill)
        bar.addStretch()
        bar.addWidget(button("Открыть папку датасета", lambda: open_folder(self.out.text() or APP_DIR / "dataset")))
        rl.addLayout(bar)

        self.progress = QProgressBar()
        self.progress.setFormat("ожидание")
        self.progress.setTextVisible(True)
        rl.addWidget(self.progress)

        vs = QSplitter(Qt.Vertical)
        top = QWidget()
        tl = QHBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        lv = QVBoxLayout()
        self.live = LiveView()
        lv.addWidget(self.live, 1)
        legend = QLabel("<span style='color:#4caf50'>■</span> принято (цвет класса)   "
                        "<span style='color:#ffd700'>■</span> сомнительно   "
                        "<span style='color:#8c8c8c'>■</span> выброшено (мусор / не объект / форма / слабо)")
        lv.addWidget(legend)
        self.frame_lbl = QLabel()
        lv.addWidget(self.frame_lbl)
        tl.addLayout(lv, 3)
        self.counts = QTableWidget(0, 2)
        self.counts.setHorizontalHeaderLabels(["Класс", "Рамок"])
        self.counts.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.counts.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.counts.verticalHeader().setVisible(False)
        self.counts.setMaximumWidth(330)
        self.counts.setEditTriggers(QAbstractItemView.NoEditTriggers)
        tl.addWidget(self.counts, 1)
        vs.addWidget(top)
        self.runner = Runner()
        self.runner.finished.connect(self.on_finished)
        vs.addWidget(self.runner)
        vs.setSizes([520, 200])
        rl.addWidget(vs, 1)
        split.addWidget(right)
        split.setSizes([400, 1000])

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll_live)
        self.load_from_settings()

    # ---------------------------------------------------------- настройки
    def load_from_settings(self):
        s = self.main.settings.get("run", {})
        for p in s.get("sources", []):
            self.sources.addItem(p)
        self.out.setText(s.get("out", str(APP_DIR / "dataset")))
        self.append.setChecked(s.get("append", False))

        def setc(cb, v):
            i = cb.findData(v)
            cb.setCurrentIndex(max(0, i))
        setc(self.preset, s.get("preset", "balanced"))
        self.preset_info.setText(PRESET_INFO[self.preset.currentData()])
        setc(self.clip, s.get("clip", ""))
        setc(self.yw, s.get("yw", "none"))
        setc(self.tiles, s.get("tiles", ""))
        setc(self.device, s.get("device", ""))
        setc(self.uncertain, s.get("uncertain", "skip"))
        self.every.setValue(s.get("every", 3))
        self.per_video.setValue(s.get("per_video", 0))
        self.max_side.setValue(s.get("max_side", 1280))
        self.dedup.setValue(s.get("dedup", 0))
        self.keep_empty.setValue(s.get("keep_empty", 0.05))
        self.det_thr.setValue(s.get("det_thr", 0.25))
        self.accept.setValue(s.get("accept", 0.55))
        self.low.setValue(s.get("low", 0.30))
        rs = s.get("reject_sim", 0.90)
        self.reject_sim.setValue(0.90 if abs(rs - 0.85) < 1e-6 else rs)  # старое значение по умолчанию
        self.track.setChecked(s.get("track", False))
        self.preview.setChecked(s.get("preview", False))
        self.extra.setText(s.get("extra", ""))

    def store_settings(self):
        self.main.settings["run"] = {
            "sources": [self.source_path(i) for i in range(self.sources.count())],
            "out": self.out.text(), "append": self.append.isChecked(), "preset": self.preset.currentData(),
            "clip": self.clip.currentData(), "yw": self.yw.currentData(), "tiles": self.tiles.currentData(),
            "device": self.device.currentData(), "uncertain": self.uncertain.currentData(),
            "every": self.every.value(), "per_video": self.per_video.value(), "max_side": self.max_side.value(),
            "dedup": self.dedup.value(), "keep_empty": self.keep_empty.value(), "det_thr": self.det_thr.value(),
            "accept": self.accept.value(), "low": self.low.value(), "reject_sim": self.reject_sim.value(),
            "track": self.track.isChecked(), "preview": self.preview.isChecked(), "extra": self.extra.text()}

    def move_source(self, d):
        r = self.sources.currentRow()
        t = r + d
        if r < 0 or not 0 <= t < self.sources.count():
            return
        it = self.sources.takeItem(r)
        self.sources.insertItem(t, it)
        self.sources.setCurrentRow(t)

    def source_path(self, i):
        return self.sources.item(i).data(Qt.UserRole) or self.sources.item(i).text()

    def mark_sources(self, video_name=None, done_all=False):
        """✓ готово, ▶ сейчас — для файлов в списке."""
        cur_found = False
        for i in range(self.sources.count()):
            it = self.sources.item(i)
            p = self.source_path(i)
            it.setData(Qt.UserRole, p)
            mark = ""
            if done_all:
                mark = "✓ "
            elif video_name:
                if Path(p).name == video_name or (Path(p).is_dir() and not cur_found and
                                                   any(x.name == video_name for x in Path(p).rglob(video_name))):
                    mark, cur_found = "▶ ", True
                elif not cur_found:
                    mark = "✓ "
            it.setText(mark + p)
            f = it.font()
            f.setBold(mark == "▶ ")
            it.setFont(f)

    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Видео или фото", self.main.settings.get("last_dir", str(APP_DIR)),
            "Видео и фото (" + " ".join("*" + e for e in sorted(IMG_EXT | VID_EXT)) + ");;Все файлы (*)")
        for f in files:
            self.sources.addItem(f)
        if files:
            self.main.settings["last_dir"] = str(Path(files[0]).parent)

    def add_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Папка с видео/фото", self.main.settings.get("last_dir", str(APP_DIR)))
        if d:
            self.sources.addItem(d)
            self.main.settings["last_dir"] = d

    def choose_out(self):
        d = QFileDialog.getExistingDirectory(self, "Папка датасета", self.out.text() or str(APP_DIR))
        if d:
            self.out.setText(d)

    # ---------------------------------------------------------- запуск
    def build_args(self):
        out = Path(self.out.text() or APP_DIR / "dataset")
        args = ["run", "--source"] + [self.source_path(i) for i in range(self.sources.count())]
        args += ["--classes", str(self.main.classes_tab.path), "--out", str(out),
                 "--refs", str(self.main.classes_tab.refs_root()), "--live", "--no-review",
                 "--preset", self.preset.currentData(),
                 "--every", str(self.every.value()), "--per-video", str(self.per_video.value()),
                 "--max-side", str(self.max_side.value()), "--dedup", str(self.dedup.value()),
                 "--keep-empty", str(self.keep_empty.value()), "--det-thr", str(self.det_thr.value()),
                 "--accept", str(self.accept.value()), "--low", str(self.low.value()),
                 "--uncertain", self.uncertain.currentData(), "--reject-sim", str(self.reject_sim.value())]
        for flag, cb in (("--clip-model", self.clip), ("--yoloworld", self.yw), ("--tiles", self.tiles),
                         ("--device", self.device)):
            if cb.currentData():
                args += [flag, cb.currentData()]
        if self.append.isChecked():
            args.append("--append")
        if self.track.isChecked():
            args.append("--track")
        if self.preview.isChecked():
            args.append("--preview")
        args += self.main.vlm_tab.run_args()
        net = self.main.settings.get("net", {})
        if net.get("no_proxy", True):
            args.append("--no-proxy")
        elif net.get("proxy"):
            args += ["--proxy", net["proxy"]]
        if net.get("offline"):
            args.append("--offline")
        if net.get("models"):
            args += ["--models", net["models"]]
        args += self.extra.text().split()
        return args, out

    def start(self):
        if self.runner.running():
            return
        if not self.sources.count():
            QMessageBox.information(self, "Разметка", "Добавьте видео, фото или папку")
            return
        if not self.main.classes_tab.save():
            self.main.tabs.setCurrentWidget(self.main.classes_tab)
            return
        args, out = self.build_args()
        obj = out / "obj"
        if obj.is_dir() and any(obj.iterdir()) and not self.append.isChecked():
            box = QMessageBox(self)
            box.setWindowTitle("Папка датасета не пустая")
            box.setText(f"В {out} уже есть размеченные кадры.")
            b_app = box.addButton("Дописать к ним", QMessageBox.AcceptRole)
            b_clr = box.addButton("Очистить и начать заново", QMessageBox.DestructiveRole)
            box.addButton("Отмена", QMessageBox.RejectRole)
            box.exec_()
            if box.clickedButton() == b_app:
                args.append("--append")
            elif box.clickedButton() == b_clr:
                for p in ("obj", "preview", "review", "_trash"):
                    shutil.rmtree(out / p, ignore_errors=True)
                for p in ("train.txt", "valid.txt", "labels_meta.jsonl", "_live.json", "_live.jpg", "report.json"):
                    (out / p).unlink(missing_ok=True)
            else:
                return
        self.store_settings()
        self.main.save_settings()
        self.main.settings["dataset"] = str(out)
        self.live_out = out
        self.live_mtime = 0
        self.stop_sent = 0
        self._cur_video = None
        self.mark_sources()
        self.last_state, self.last_saved = None, 0
        if out.exists():
            (out / "_live.json").unlink(missing_ok=True)
        self.runner.start(args)
        self.started_at = time.time()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_kill.setEnabled(True)
        self.btn_skip.setEnabled(True)
        self.progress.setRange(0, 0)
        self.progress.setFormat("загрузка моделей...")
        self.timer.start(400)
        self.main.status("Разметка запущена")

    def skip(self):
        if not self.runner.running():
            return
        self.live_out.mkdir(parents=True, exist_ok=True)
        (self.live_out / "_SKIP").write_text("skip")
        self.main.status(f"Пропускаю видео {getattr(self, '_cur_video', '') or ''} — перехожу к следующему")

    def stop(self):
        if not self.runner.running():
            return
        self.live_out.mkdir(parents=True, exist_ok=True)
        (self.live_out / "_STOP").write_text("stop")
        self.stop_sent = time.time()
        self.progress.setFormat("останавливаю — доделываю кадр и сохраняю...")
        self.main.status("Остановка: дождитесь сохранения (или «Прервать»)")

    def kill(self):
        if self.runner.running() and QMessageBox.question(
                self, "Прервать", "Убить процесс сразу? Последние кадры могут не сохраниться.") == QMessageBox.Yes:
            self.runner.kill()

    def on_finished(self, code):
        self.timer.stop()
        self.poll_live(force=True)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_kill.setEnabled(False)
        self.btn_skip.setEnabled(False)
        if self.progress.maximum() == 0:
            self.progress.setRange(0, 1)
            self.progress.setValue(1 if code == 0 else 0)
        if code != 0:
            self.progress.setFormat(f"завершено с ошибкой (код {code}) — см. лог")
        else:
            self.progress.setFormat(("остановлено, размеченное сохранено" if self.last_state == "stopped" else
                                     "готово") + f"  |  сохранено кадров: {self.last_saved}")
        (self.live_out / "_STOP").unlink(missing_ok=True)
        (self.live_out / "_SKIP").unlink(missing_ok=True)
        self.main.dataset_changed(reload=True)
        self.main.status("Разметка завершена" if code == 0 else "Разметка завершилась с ошибкой — смотрите лог")
        if code == 0:
            QApplication.beep()

    def poll_live(self, force=False):
        out = getattr(self, "live_out", None)
        if not out:
            return
        jf, jp = out / "_live.json", out / "_live.jpg"
        try:
            mt = jf.stat().st_mtime
        except OSError:
            return
        if mt == self.live_mtime and not force:
            return
        self.live_mtime = mt
        try:
            info = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            return
        self.last_state, self.last_saved = info.get("state"), info.get("saved", 0)
        if info.get("state") == "running" and info.get("video") and info.get("video") != getattr(self, "_cur_video", None):
            self._cur_video = info.get("video")
            self.mark_sources(self._cur_video)
        elif info.get("state") == "done":
            self.mark_sources(done_all=True)
        if info.get("state") == "running" and jp.exists():
            img = QImage(str(jp))
            if not img.isNull():
                self.live.set_image(img)
            st = Counter(b["status"] for b in info.get("boxes", []))
            self.frame_lbl.setText(f"{info.get('video', '')}  →  кадр: {info.get('frame', '')}   "
                                   f"рамок: принято {st['ok']}, "
                                   f"сомнительно {st['unsure']}, выброшено "
                                   f"{st['reject'] + st['shape'] + st['low']}")
        seen, total, spf = info.get("seen", 0), info.get("total", 0), info.get("spf", 0)
        if total:
            self.progress.setRange(0, total)
            self.progress.setValue(min(seen, total))
            left = max(0, total - seen) * spf
            eta = f"{int(left // 3600)} ч {int(left % 3600 // 60)} мин" if left >= 3600 else \
                f"{int(left // 60)} мин {int(left % 60)} с"
            vid = f"видео {info.get('video_k')}/{info.get('videos')}  |  " if info.get("videos") else ""
            self.progress.setFormat(vid + f"кадр {seen} из ~{total}  |  сохранено {info.get('saved', 0)}  |  "
                                    f"{spf:.2f} с/кадр  |  осталось ~{eta}"
                                    + ("  |  ОСТАНАВЛИВАЮ..." if self.stop_sent and self.runner.running() else ""))
        stats = info.get("stats", {})
        names = info.get("classes", [])
        rows = [(n, stats.get(n, 0)) for n in names] + \
               [(k, v) for k, v in stats.items() if k not in names and v]
        self.counts.setRowCount(len(rows))
        for r, (k, v) in enumerate(rows):
            it = QTableWidgetItem(k)
            if k in names:
                it.setForeground(QBrush(class_qcolor(names.index(k))))
            else:
                it.setForeground(QBrush(QColor("#999")))
            self.counts.setItem(r, 0, it)
            self.counts.setItem(r, 1, QTableWidgetItem(str(v)))


# ============================================================================ 3. Чистка

class CleanTab(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.thumbs = {}      # key -> QPixmap
        self.feats = {}       # key -> вектор
        self.items = {}       # key -> QListWidgetItem
        self.worker = None
        self.undo_stack = []
        self.loaded_version = -1
        self.cur_class = None
        lay = QHBoxLayout(self)

        left = QVBoxLayout()
        left.addWidget(QLabel("<b>Классы</b>"))
        self.cls_list = QListWidget()
        self.cls_list.setMaximumWidth(260)
        self.cls_list.currentRowChanged.connect(self.on_class)
        left.addWidget(self.cls_list, 1)
        left.addWidget(button("Обновить из папки", lambda: self.main.dataset_changed(reload=True)))
        help_ = QLabel(
            "<b>Клавиши</b><br>"
            "Del — удалить рамки<br>"
            "1..9, 0 — сменить класс<br>"
            "R — в мусор (_reject) и удалить<br>"
            "S — в образцы класса<br>"
            "F — удалить кадры целиком<br>"
            "P — выделить похожие<br>"
            "Enter / двойной клик — открыть кадр<br>"
            "V — проверить через VLM<br>"
            "A — принять ответы VLM<br>"
            "Ctrl+A — всё, Ctrl+Z — отменить")
        help_.setStyleSheet("color:#aaa")
        left.addWidget(help_)
        lay.addLayout(left)

        center = QVBoxLayout()
        tb = QHBoxLayout()
        tb.addWidget(QLabel("Сортировка:"))
        self.sort = QComboBox()
        for t, v in (("по уверенности — сомнительные первыми", "conf"), ("по кадрам", "frame"),
                     ("по размеру — мелкие первыми", "size"), ("по вытянутости", "aspect")):
            self.sort.addItem(t, v)
        self.sort.currentIndexChanged.connect(self.rebuild)
        tb.addWidget(self.sort)
        tb.addWidget(QLabel("  Размер:"))
        self.size = QSlider(Qt.Horizontal)
        self.size.setRange(48, 200)
        self.size.setValue(96)
        self.size.setMaximumWidth(140)
        self.size.valueChanged.connect(lambda v: self.grid.setIconSize(QSize(v, v)))
        tb.addWidget(self.size)
        tb.addStretch()
        tb.addWidget(QLabel("Похожесть:"))
        self.sim = QSlider(Qt.Horizontal)
        self.sim.setRange(70, 99)
        self.sim.setValue(90)
        self.sim.setMaximumWidth(120)
        self.sim_lbl = QLabel("0.90")
        self.sim.valueChanged.connect(lambda v: self.sim_lbl.setText(f"{v / 100:.2f}"))
        tb.addWidget(self.sim)
        tb.addWidget(self.sim_lbl)
        tb.addWidget(button("Выделить похожие (P)", self.select_similar,
                            "Выделить все вырезки, похожие на уже выделенные (например, одну и ту же коробку)"))
        center.addLayout(tb)

        self.grid = QListWidget()
        self.grid.setViewMode(QListWidget.IconMode)
        self.grid.setIconSize(QSize(96, 96))
        self.grid.setResizeMode(QListWidget.Adjust)
        self.grid.setMovement(QListWidget.Static)
        self.grid.setSpacing(4)
        self.grid.setUniformItemSizes(True)
        self.grid.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.grid.itemSelectionChanged.connect(self.on_sel)
        self.grid.itemDoubleClicked.connect(self.open_frame)
        self.grid.setContextMenuPolicy(Qt.CustomContextMenu)
        self.grid.customContextMenuRequested.connect(self.context_menu)
        center.addWidget(self.grid, 1)

        act = QHBoxLayout()
        self.sel_lbl = QLabel("выбрано: 0")
        act.addWidget(self.sel_lbl)
        act.addStretch()
        act.addWidget(button("Удалить рамки (Del)", self.act_delete))
        self.move_cb = QComboBox()
        self.move_cb.setMinimumWidth(170)
        act.addWidget(self.move_cb)
        act.addWidget(button("Сменить класс", lambda: self.act_class(self.move_cb.currentData())))
        act.addWidget(button("В мусор (R)", self.act_reject, "Скопировать в refs/_reject и удалить рамки"))
        act.addWidget(button("В образцы (S)", self.act_sample, "Скопировать в refs/<класс> (рамки остаются)"))
        act.addWidget(button("Удалить кадры (F)", self.act_frames, "Кадры целиком — в корзину _trash"))
        act.addWidget(button("Отменить (Ctrl+Z)", self.undo))
        center.addLayout(act)
        act = QHBoxLayout()
        self.vlm_btn = button("🤖 Проверить через VLM (V)", self.vlm_check,
                              "Отправить выделенные вырезки VLM-судье; несогласные будут подсвечены и выделены")
        act.addWidget(self.vlm_btn)
        act.addWidget(button("Принять ответы VLM (A)", self.vlm_apply,
                             "Для выделенных: сменить класс на ответ VLM, а «ничего» — удалить рамку"))
        self.vlm_lbl = QLabel("")
        act.addWidget(self.vlm_lbl, 1)
        center.addLayout(act)
        lay.addLayout(center, 1)

        self.vlm_ans = {}     # key -> (класс|-1, уверенность)
        self.vlm_task = None
        for key, fn in (("Delete", self.act_delete), ("R", self.act_reject), ("S", self.act_sample),
                        ("V", self.vlm_check), ("A", self.vlm_apply),
                        ("F", self.act_frames), ("P", self.select_similar), ("Ctrl+Z", self.undo),
                        ("Return", lambda: self.open_frame(self.grid.currentItem()))):
            sc = QShortcut(QKeySequence(key), self.grid)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)
        for d in range(10):
            sc = QShortcut(QKeySequence(str(d)), self.grid)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(lambda d=d: self.act_class((d - 1) % 10))

    # ---------------------------------------------------------------- загрузка
    @property
    def ds(self):
        return self.main.dataset

    def showEvent(self, e):
        super().showEvent(e)
        if self.ds is None or self.loaded_version != self.ds.version:
            self.refresh()

    def refresh(self):
        ds = self.ds
        cur = self.cls_list.currentRow()
        self.cls_list.blockSignals(True)
        self.cls_list.clear()
        self.move_cb.clear()
        if ds is None:
            self.cls_list.blockSignals(False)
            self.grid.clear()
            return
        cnt, _ = ds.count()
        total = sum(cnt.values())
        self.cls_list.addItem(f"Все классы   ({total})")
        for i, n in enumerate(ds.names):
            it = QListWidgetItem(f"{i + 1 if i < 9 else 0 if i == 9 else '-'}  {n}   ({cnt[i]})")
            it.setForeground(QBrush(class_qcolor(i)))
            self.cls_list.addItem(it)
            self.move_cb.addItem(f"→ {n}", i)
        bad = [k for k in cnt if not 0 <= k < len(ds.names)]
        if bad:
            self.cls_list.addItem(f"Неизвестный класс   ({sum(cnt[k] for k in bad)})")
        self.cls_list.blockSignals(False)
        self.loaded_version = ds.version
        self.cls_list.setCurrentRow(cur if 0 <= cur < self.cls_list.count() else 0)
        self.on_class(self.cls_list.currentRow())

    def on_class(self, row):
        if self.ds is None:
            return
        n = len(self.ds.names)
        self.cur_class = None if row <= 0 else (row - 1 if row - 1 < n else "bad")
        self.rebuild()

    def boxes_for_view(self):
        ds, out = self.ds, []
        for fr in ds.frames.values():
            for i, b in enumerate(fr.boxes):
                if not b.alive:
                    continue
                if self.cur_class is None or b.c == self.cur_class or \
                        (self.cur_class == "bad" and not 0 <= b.c < len(ds.names)):
                    out.append((fr, i, b))
        mode = self.sort.currentData()
        if mode == "conf":
            out.sort(key=lambda t: (t[2].conf if t[2].conf is not None else 9, t[0].stem))
        elif mode == "size":
            out.sort(key=lambda t: t[2].w * t[2].h)
        elif mode == "aspect":
            out.sort(key=lambda t: -max(t[2].w / max(t[2].h, 1e-6), t[2].h / max(t[2].w, 1e-6)))
        return out

    def rebuild(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait(2000)
            self.worker = None
        self.grid.clear()
        self.items = {}
        if self.ds is None:
            return
        placeholder = QPixmap(96, 96)
        placeholder.fill(QColor(50, 50, 50))
        ph_icon = QIcon(placeholder)
        jobs = []
        self.grid.setUpdatesEnabled(False)
        for fr, i, b in self.boxes_for_view():
            key = f"{fr.stem}|{i}|{b.key()}"
            label = f"{b.conf:.2f}" if b.conf is not None else fr.stem.split("_")[0]
            it = QListWidgetItem(QIcon(self.thumbs[key]) if key in self.thumbs else ph_icon, label)
            it.setData(Qt.UserRole, (fr.stem, i))
            it.setData(Qt.UserRole + 1, key)
            it.setData(Qt.UserRole + 2, label)
            name = self.ds.names[b.c] if 0 <= b.c < len(self.ds.names) else f"?{b.c}"
            it.setToolTip(f"{name}\nкадр {fr.stem}\nуверенность: {b.conf if b.conf is not None else '—'}")
            if self.cur_class is None:
                it.setForeground(QBrush(class_qcolor(b.c)))
            if b.conf is not None and b.conf < 0.5:
                it.setBackground(QBrush(QColor(90, 70, 20)))
            self.grid.addItem(it)
            self.items[key] = it
            if key in self.vlm_ans:
                self._mark_vlm(it, b.c, *self.vlm_ans[key])
            if key not in self.thumbs:
                jobs.append((key, fr.img, (b.xc, b.yc, b.w, b.h)))
        self.grid.setUpdatesEnabled(True)
        if jobs:
            self.worker = ThumbWorker(jobs, 160)
            self.worker.ready.connect(self.on_thumb)
            self.worker.start()
        self.on_sel()

    def on_thumb(self, key, qimg, feat):
        pix = QPixmap.fromImage(qimg)
        self.thumbs[key] = pix
        self.feats[key] = feat
        it = self.items.get(key)
        if it is not None:
            it.setIcon(QIcon(pix))

    def on_sel(self):
        self.sel_lbl.setText(f"выбрано: {len(self.grid.selectedItems())} из {self.grid.count()}")

    def selection(self):
        return [it.data(Qt.UserRole) for it in self.grid.selectedItems()]

    def remove_items(self, items):
        for it in items:
            self.items.pop(it.data(Qt.UserRole + 1), None)
            self.grid.takeItem(self.grid.row(it))
        self.update_counts()
        self.on_sel()

    def update_counts(self):
        cnt, _ = self.ds.count()
        self.cls_list.item(0).setText(f"Все классы   ({sum(cnt.values())})")
        for i, n in enumerate(self.ds.names):
            if i + 1 < self.cls_list.count():
                self.cls_list.item(i + 1).setText(f"{i + 1 if i < 9 else 0 if i == 9 else '-'}  {n}   ({cnt[i]})")
        self.loaded_version = self.ds.version

    # ---------------------------------------------------------------- действия
    def _crop(self, stem, i, pad=0.15):
        img = self.ds.image(stem)
        b = self.ds.frames[stem].boxes[i]
        return crop_box(img, b.xc, b.yc, b.w, b.h, pad) if img is not None else None

    def _boxes_action(self, fn, remove=True):
        items = self.grid.selectedItems()
        if not items or self.ds is None:
            return []
        sel = [it.data(Qt.UserRole) for it in items]
        stems = sorted({s for s, _ in sel})
        self.undo_stack.append(("boxes", self.ds.snapshot(stems)))
        for s, i in sel:
            fn(self.ds.frames[s].boxes[i], s, i)
        self.ds.save(stems)
        if remove:
            self.remove_items(items)
        self.main.frames_tab.mark_dirty()
        return sel

    def act_delete(self):
        sel = self._boxes_action(lambda b, s, i: setattr(b, "alive", False))
        if sel:
            self.main.status(f"Удалено рамок: {len(sel)}  (Ctrl+Z — отменить)")

    def act_class(self, c):
        if c is None or self.ds is None or not 0 <= c < len(self.ds.names):
            return
        stay = self.cur_class is None or self.cur_class == c

        def fn(b, s, i):
            b.c = c
        sel = self._boxes_action(fn, remove=not stay)
        if sel:
            if stay:
                for it in self.grid.selectedItems():
                    it.setForeground(QBrush(class_qcolor(c)))
                self.update_counts()
            self.main.status(f"Класс «{self.ds.names[c]}» для {len(sel)} рамок")

    def _copy_to_refs(self, sel, folder_fn):
        root = self.main.classes_tab.refs_root()
        n = 0
        for s, i in sel:
            cr = self._crop(s, i)
            if cr is None or cr.size == 0:
                continue
            d = root / folder_fn(s, i)
            d.mkdir(parents=True, exist_ok=True)
            if AL.imwrite_u(d / f"{s}_{i}.jpg", cr):
                n += 1
        return n

    def act_reject(self):
        sel = self.selection()
        if not sel:
            return
        n = self._copy_to_refs(sel, lambda s, i: "_reject")
        self._boxes_action(lambda b, s, i: setattr(b, "alive", False))
        self.main.status(f"В _reject добавлено {n} вырезок, рамки удалены. При следующей разметке похожее "
                         f"будет выбрасываться")

    def act_sample(self):
        sel = self.selection()
        if not sel:
            return
        names = self.ds.names
        n = self._copy_to_refs(sel, lambda s, i: names[self.ds.frames[s].boxes[i].c]
                               if 0 <= self.ds.frames[s].boxes[i].c < len(names) else "_unknown")
        self.main.status(f"В образцы скопировано {n} вырезок (вкладка «Классы» — посмотреть)")

    def act_frames(self):
        sel = self.selection()
        stems = sorted({s for s, _ in sel})
        if not stems:
            return
        if len(stems) > 20 and QMessageBox.question(
                self, "Удалить кадры", f"Убрать {len(stems)} кадров в корзину?") != QMessageBox.Yes:
            return
        moved, removed = self.ds.delete_frames(stems)
        self.undo_stack.append(("frames", (moved, removed)))
        gone = set(stems)
        self.remove_items([self.grid.item(r) for r in range(self.grid.count())
                           if self.grid.item(r).data(Qt.UserRole)[0] in gone])
        self.main.frames_tab.mark_dirty()
        self.main.status(f"Кадров в корзине: {len(moved)}  (Ctrl+Z — вернуть)")

    def undo(self):
        if not self.undo_stack or self.ds is None:
            self.main.status("Нечего отменять")
            return
        kind, data = self.undo_stack.pop()
        if kind == "boxes":
            self.ds.restore(data)
        else:
            self.ds.restore_frames(*data)
        self.main.frames_tab.mark_dirty()
        self.refresh()
        self.main.status("Отменено")

    def select_similar(self):
        items = self.grid.selectedItems()
        keys = [it.data(Qt.UserRole + 1) for it in items if it.data(Qt.UserRole + 1) in self.feats]
        if not keys:
            self.main.status("Выделите одну-две вырезки (дождитесь загрузки миниатюр)")
            return
        thr = self.sim.value() / 100
        S = np.stack([self.feats[k] for k in keys])
        n = 0
        self.grid.blockSignals(True)
        for key, it in self.items.items():
            f = self.feats.get(key)
            if f is not None and not it.isSelected() and float((S @ f).max()) >= thr:
                it.setSelected(True)
                n += 1
        self.grid.blockSignals(False)
        self.on_sel()
        self.main.status(f"Добавлено к выделению похожих: {n}. Проверьте глазами и жмите действие")

    # ---------------------------------------------------------------- VLM
    def _mark_vlm(self, it, cur_c, k, q):
        names = self.ds.names
        ans = "ничего" if k == -1 else (names[k] if 0 <= k < len(names) else "?")
        agree = k == cur_c
        it.setBackground(QBrush(QColor(30, 90, 40) if agree else QColor(140, 70, 20)))
        it.setText(f"✓ {it.data(Qt.UserRole + 2)}" if agree else f"VLM: {ans}")
        it.setToolTip(it.toolTip().split("\nVLM:")[0] + f"\nVLM: {ans} ({q:.2f})")

    def vlm_check(self):
        if self.ds is None:
            return
        if self.vlm_task and self.vlm_task.isRunning():
            self.vlm_task.stop_flag = True
            self.vlm_btn.setText("🤖 Проверить через VLM (V)")
            return
        items = self.grid.selectedItems()
        if not items:
            self.main.status("Выделите вырезки (Ctrl+A — все на экране)")
            return
        judge = self.main.vlm_tab.make_judge(self.ds.names)
        if judge is None:
            return
        jobs = []
        for it in items:
            s_, i = it.data(Qt.UserRole)
            cr = self._crop(s_, i, pad=0.25)
            if cr is not None and cr.size:
                jobs.append((it.data(Qt.UserRole + 1), self.ds.frames[s_].boxes[i].c, cr.copy()))
        self.vlm_lbl.setText(f"VLM: 0 / {len(jobs)} ...")
        self.vlm_btn.setText("■ Остановить VLM")
        state = {"n": 0, "dis": 0, "err": 0}

        def fn(t):
            step = max(1, judge.workers)
            for k0 in range(0, len(jobs), step):
                if t.stop_flag:
                    break
                chunk = jobs[k0:k0 + step]
                for (key, c, _), res in zip(chunk, judge.ask_many([j[2] for j in chunk])):
                    t.item.emit((key, c, res))
                if judge.fails >= 5:
                    return RuntimeError(judge.last_error)
            return None

        def on_item(v):
            key, c, (k, q, raw) = v
            state["n"] += 1
            if k is None:
                state["err"] += 1
            else:
                self.vlm_ans[key] = (k, q)
                state["dis"] += k != c
                it = self.items.get(key)
                if it is not None:
                    self._mark_vlm(it, c, k, q)
            self.vlm_lbl.setText(f"VLM: {state['n']} / {len(jobs)}, не согласна: {state['dis']}"
                                 + (f", ошибок: {state['err']}" if state["err"] else ""))

        def done(res):
            self.vlm_btn.setText("🤖 Проверить через VLM (V)")
            self.grid.clearSelection()
            self.grid.blockSignals(True)
            for key, (k, q) in self.vlm_ans.items():
                it = self.items.get(key)
                if it is not None:
                    s_, i = it.data(Qt.UserRole)
                    if self.ds.frames.get(s_) and k != self.ds.frames[s_].boxes[i].c:
                        it.setSelected(True)
            self.grid.blockSignals(False)
            self.on_sel()
            msg = f"VLM проверила {state['n']}: не согласна с {state['dis']} — они выделены (оранжевые). " \
                  f"Посмотрите и нажмите «Принять ответы VLM» (A) или исправьте вручную"
            if isinstance(res, Exception):
                msg = f"VLM остановлена: {res}"
            self.main.status(msg)
        self.vlm_task = Task(fn)
        self.vlm_task.item.connect(on_item)
        self.vlm_task.done.connect(done)
        self.vlm_task.start()

    def vlm_apply(self):
        items = [it for it in self.grid.selectedItems() if it.data(Qt.UserRole + 1) in self.vlm_ans]
        if not items or self.ds is None:
            self.main.status("Выделите вырезки, проверенные VLM (оранжевые)")
            return
        sel = [(it.data(Qt.UserRole), self.vlm_ans[it.data(Qt.UserRole + 1)][0]) for it in items]
        stems = sorted({s_ for (s_, _), _ in sel})
        self.undo_stack.append(("boxes", self.ds.snapshot(stems)))
        nd = nc = 0
        for (s_, i), k in sel:
            b = self.ds.frames[s_].boxes[i]
            if k == -1:
                b.alive = False
                nd += 1
            elif k != b.c:
                b.c = k
                nc += 1
        self.ds.save(stems)
        self.main.frames_tab.mark_dirty()
        self.refresh()
        self.main.status(f"Ответы VLM приняты: класс изменён у {nc}, удалено {nd}  (Ctrl+Z — отменить)")

    def open_frame(self, item):
        if item is None:
            return
        s, i = item.data(Qt.UserRole)
        self.main.tabs.setCurrentWidget(self.main.frames_tab)
        self.main.frames_tab.open(s, i)

    def context_menu(self, pos):
        if not self.grid.selectedItems():
            return
        m = QMenu(self)
        m.addAction("Открыть кадр", lambda: self.open_frame(self.grid.currentItem()))
        m.addAction("Выделить похожие", self.select_similar)
        m.addAction("Проверить через VLM", self.vlm_check)
        m.addAction("Принять ответы VLM", self.vlm_apply)
        m.addSeparator()
        m.addAction("Удалить рамки", self.act_delete)
        sub = m.addMenu("Сменить класс")
        for i, n in enumerate(self.ds.names):
            sub.addAction(n, lambda i=i: self.act_class(i))
        m.addAction("В мусор (_reject)", self.act_reject)
        m.addAction("В образцы класса", self.act_sample)
        m.addAction("Удалить кадры целиком", self.act_frames)
        m.exec_(self.grid.mapToGlobal(pos))


# ============================================================================ 4. Кадры (редактор)

class Canvas(QWidget):
    changed = pyqtSignal(object)      # snapshot до изменения
    nav = pyqtSignal(int)
    selected_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setMinimumSize(400, 300)
        self.pix = None
        self.frame = None
        self.names = []
        self.sel = None
        self.cur_class = 0
        self.mode = None
        self.drag_start = None
        self.orig = None
        self.new_rect = None
        self.snap = None

    def set_frame(self, frame, img, names, select=None):
        self.frame, self.names = frame, names
        self.pix = QPixmap.fromImage(bgr_to_qimage(img)) if img is not None else None
        self.sel = select
        self.update()
        self.selected_changed.emit()

    # ---------------------------------------------------------------- геометрия
    def geom(self):
        if not self.pix:
            return 1.0, 0.0, 0.0
        s = min(self.width() / self.pix.width(), self.height() / self.pix.height())
        return s, (self.width() - self.pix.width() * s) / 2, (self.height() - self.pix.height() * s) / 2

    def to_img(self, p):
        s, ox, oy = self.geom()
        return ((p.x() - ox) / s / self.pix.width(), (p.y() - oy) / s / self.pix.height())

    def box_rect(self, b):
        s, ox, oy = self.geom()
        W, H = self.pix.width() * s, self.pix.height() * s
        return QRectF(ox + (b.xc - b.w / 2) * W, oy + (b.yc - b.h / 2) * H, b.w * W, b.h * H)

    def hit(self, p):
        """-> (индекс, режим) режим: move / tl / tr / bl / br"""
        if not self.frame:
            return None, None
        best = None
        for i, b in enumerate(self.frame.boxes):
            if not b.alive:
                continue
            r = self.box_rect(b)
            for name, c in (("tl", r.topLeft()), ("tr", r.topRight()), ("bl", r.bottomLeft()),
                            ("br", r.bottomRight())):
                if abs(c.x() - p.x()) <= 7 and abs(c.y() - p.y()) <= 7:
                    return i, name
            if r.contains(QPointF(p)):
                if best is None or r.width() * r.height() < self.box_rect(self.frame.boxes[best]).width() * \
                        self.box_rect(self.frame.boxes[best]).height():
                    best = i
        return (best, "move") if best is not None else (None, None)

    # ---------------------------------------------------------------- рисование
    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(27, 27, 27))
        if not self.pix:
            p.setPen(QColor(140, 140, 140))
            p.drawText(self.rect(), Qt.AlignCenter, "Выберите кадр слева")
            return
        s, ox, oy = self.geom()
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.drawPixmap(QRectF(ox, oy, self.pix.width() * s, self.pix.height() * s), self.pix,
                     QRectF(0, 0, self.pix.width(), self.pix.height()))
        f = p.font()
        f.setPointSize(9)
        f.setBold(True)
        p.setFont(f)
        for i, b in enumerate(self.frame.boxes if self.frame else []):
            if not b.alive:
                continue
            r = self.box_rect(b)
            col = class_qcolor(b.c) if 0 <= b.c < len(self.names) else QColor(255, 0, 0)
            p.setPen(QPen(col, 3 if i == self.sel else 2))
            p.setBrush(QColor(col.red(), col.green(), col.blue(), 45) if i == self.sel else Qt.NoBrush)
            p.drawRect(r)
            name = self.names[b.c] if 0 <= b.c < len(self.names) else f"?{b.c}"
            txt = name + (f" {b.conf:.2f}" if b.conf is not None else "")
            fm = p.fontMetrics()
            tw = (fm.horizontalAdvance(txt) if hasattr(fm, "horizontalAdvance") else fm.width(txt)) + 6
            ty = r.top() - 16 if r.top() > 18 else r.bottom() + 1
            p.fillRect(QRectF(r.left(), ty, tw, 15), col)
            p.setPen(QColor(0, 0, 0))
            p.drawText(QRectF(r.left() + 3, ty, tw, 15), Qt.AlignVCenter, txt)
            if i == self.sel:
                p.setBrush(col)
                for c in (r.topLeft(), r.topRight(), r.bottomLeft(), r.bottomRight()):
                    p.drawRect(QRectF(c.x() - 4, c.y() - 4, 8, 8))
        if self.new_rect is not None:
            col = class_qcolor(self.cur_class)
            p.setPen(QPen(col, 2, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawRect(self.new_rect)

    # ---------------------------------------------------------------- мышь
    def mousePressEvent(self, e):
        if not self.pix or e.button() != Qt.LeftButton:
            return
        self.setFocus()
        i, mode = self.hit(e.pos())
        self.snap = [b.tup() for b in self.frame.boxes]
        if i is not None:
            self.sel, self.mode = i, mode
            b = self.frame.boxes[i]
            self.orig = (b.xc - b.w / 2, b.yc - b.h / 2, b.xc + b.w / 2, b.yc + b.h / 2)
        else:
            self.sel, self.mode = None, "new"
            self.new_rect = QRectF(QPointF(e.pos()), QPointF(e.pos()))
        self.drag_start = e.pos()
        self.moved = False
        self.selected_changed.emit()
        self.update()

    def mouseMoveEvent(self, e):
        if not self.pix:
            return
        if self.mode is None:
            _, m = self.hit(e.pos())
            self.setCursor({"move": Qt.SizeAllCursor, "tl": Qt.SizeFDiagCursor, "br": Qt.SizeFDiagCursor,
                            "tr": Qt.SizeBDiagCursor, "bl": Qt.SizeBDiagCursor}.get(m, Qt.CrossCursor))
            return
        self._drag(e.pos())

    def _drag(self, pos):
        if (pos - self.drag_start).manhattanLength() < 3:
            return
        self.moved = True
        if self.mode == "new":
            self.new_rect = QRectF(QPointF(self.drag_start), QPointF(pos)).normalized()
        else:
            b = self.frame.boxes[self.sel]
            x0, y0 = self.to_img(self.drag_start)
            x1, y1 = self.to_img(pos)
            dx, dy = x1 - x0, y1 - y0
            l, t, r, bt = self.orig
            if self.mode == "move":
                l, r, t, bt = l + dx, r + dx, t + dy, bt + dy
            else:
                if "l" in self.mode:
                    l += dx
                if "r" in self.mode:
                    r += dx
                if "t" in self.mode:
                    t += dy
                if "b" in self.mode:
                    bt += dy
            l, r = sorted((min(max(l, 0), 1), min(max(r, 0), 1)))
            t, bt = sorted((min(max(t, 0), 1), min(max(bt, 0), 1)))
            b.xc, b.yc, b.w, b.h = (l + r) / 2, (t + bt) / 2, max(r - l, 1e-4), max(bt - t, 1e-4)
        self.update()

    def mouseReleaseEvent(self, e):
        if self.mode is not None:
            self._drag(e.pos())
        if self.mode == "new" and self.new_rect is not None and self.new_rect.width() > 5 and \
                self.new_rect.height() > 5:
            x1, y1 = self.to_img(self.new_rect.topLeft())
            x2, y2 = self.to_img(self.new_rect.bottomRight())
            x1, x2 = sorted((min(max(x1, 0), 1), min(max(x2, 0), 1)))
            y1, y2 = sorted((min(max(y1, 0), 1), min(max(y2, 0), 1)))
            if x2 - x1 > 1e-3 and y2 - y1 > 1e-3:
                self.frame.boxes.append(Box(self.cur_class, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
                self.sel = len(self.frame.boxes) - 1
                self.changed.emit(self.snap)
        elif self.mode in ("move", "tl", "tr", "bl", "br") and self.moved:
            self.changed.emit(self.snap)
        self.mode, self.new_rect = None, None
        self.selected_changed.emit()
        self.update()

    def contextMenuEvent(self, e):
        i, _ = self.hit(e.pos())
        if i is None:
            return
        self.sel = i
        self.update()
        m = QMenu(self)
        for c, n in enumerate(self.names):
            m.addAction(f"{c + 1}. {n}", lambda c=c: self.set_class(c))
        m.addSeparator()
        m.addAction("Удалить рамку (Del)", self.delete_sel)
        m.exec_(e.globalPos())

    # ---------------------------------------------------------------- клавиатура
    def keyPressEvent(self, e):
        k = e.key()
        if k in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_sel()
        elif k in (Qt.Key_Right, Qt.Key_D, Qt.Key_Space):
            self.nav.emit(1)
        elif k in (Qt.Key_Left, Qt.Key_A):
            self.nav.emit(-1)
        elif Qt.Key_0 <= k <= Qt.Key_9:
            c = (k - Qt.Key_0 - 1) % 10
            if c < len(self.names):
                if self.sel is not None:
                    self.set_class(c)
                else:
                    self.cur_class = c
                    self.selected_changed.emit()
        elif k == Qt.Key_Escape:
            self.sel = None
            self.update()
            self.selected_changed.emit()
        else:
            super().keyPressEvent(e)

    def delete_sel(self):
        if self.sel is None or not self.frame:
            return
        snap = [b.tup() for b in self.frame.boxes]
        self.frame.boxes[self.sel].alive = False
        self.sel = None
        self.changed.emit(snap)
        self.selected_changed.emit()
        self.update()

    def set_class(self, c):
        if self.sel is None or not self.frame:
            return
        snap = [b.tup() for b in self.frame.boxes]
        self.frame.boxes[self.sel].c = c
        self.cur_class = c
        self.changed.emit(snap)
        self.selected_changed.emit()
        self.update()


class FramesTab(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.dirty = True
        self.undo_stack = []
        self.cur = None
        lay = QHBoxLayout(self)

        left = QVBoxLayout()
        self.filter = QComboBox()
        self.filter.currentIndexChanged.connect(self.fill)
        left.addWidget(self.filter)
        self.search = QLineEdit()
        self.search.setPlaceholderText("поиск по имени / номеру кадра")
        self.search.textChanged.connect(self.fill)
        left.addWidget(self.search)
        self.list = QListWidget()
        self.list.setMaximumWidth(290)
        self.list.currentItemChanged.connect(lambda cur, _p: self.show_frame(cur.data(Qt.UserRole)) if cur else None)
        left.addWidget(self.list, 1)
        self.list_lbl = QLabel()
        left.addWidget(self.list_lbl)
        lay.addLayout(left)

        self.canvas = Canvas()
        self.canvas.changed.connect(self.on_changed)
        self.canvas.nav.connect(self.step)
        self.canvas.selected_changed.connect(self.update_palette)
        lay.addWidget(self.canvas, 1)

        right = QVBoxLayout()
        right.addWidget(QLabel("<b>Класс новых рамок</b>"))
        self.palette = QListWidget()
        self.palette.setMaximumWidth(230)
        self.palette.itemClicked.connect(self.on_palette)
        right.addWidget(self.palette, 1)
        right.addWidget(button("Удалить кадр (в корзину)", self.delete_frame))
        right.addWidget(button("Отменить (Ctrl+Z)", self.undo))
        hint = QLabel("<b>Мышь</b><br>тянуть по пустому — новая рамка<br>тянуть рамку — сдвинуть<br>"
                      "тянуть угол — размер<br>правый клик — класс/удалить<br><br><b>Клавиши</b><br>"
                      "← → (A/D, пробел) — кадры<br>1..9, 0 — класс выбранной<br>(или класс новых)<br>"
                      "Del — удалить рамку<br>Esc — снять выделение<br>Ctrl+Z — отменить<br><br>"
                      "Всё сохраняется сразу.")
        hint.setStyleSheet("color:#aaa")
        right.addWidget(hint)
        lay.addLayout(right)
        sc = QShortcut(QKeySequence("Ctrl+Z"), self)
        sc.activated.connect(self.undo)

    @property
    def ds(self):
        return self.main.dataset

    def mark_dirty(self):
        self.dirty = True

    def showEvent(self, e):
        super().showEvent(e)
        if self.dirty:
            self.reload()

    def reload(self):
        self.dirty = False
        ds = self.ds
        self.filter.blockSignals(True)
        cur = self.filter.currentData()
        self.filter.clear()
        self.filter.addItem("Все кадры", "all")
        self.filter.addItem("Пустые (без рамок)", "empty")
        self.filter.addItem("С ошибками разметки", "err")
        self.filter.addItem("С сомнительными (уверенность < 0.6)", "unsure")
        if ds:
            for i, n in enumerate(ds.names):
                self.filter.addItem(f"С классом: {n}", i)
        i = self.filter.findData(cur)
        self.filter.setCurrentIndex(max(0, i))
        self.filter.blockSignals(False)
        self.palette.clear()
        if ds:
            for i, n in enumerate(ds.names):
                it = QListWidgetItem(f"{i + 1 if i < 9 else 0 if i == 9 else '-'}  {n}")
                pm = QPixmap(14, 14)
                pm.fill(class_qcolor(i))
                it.setIcon(QIcon(pm))
                self.palette.addItem(it)
        self.fill()
        self.update_palette()

    def fill(self):
        ds = self.ds
        cur = self.cur
        self.list.blockSignals(True)
        self.list.clear()
        if not ds:
            self.list.blockSignals(False)
            self.canvas.set_frame(None, None, [])
            return
        f, q = self.filter.currentData(), self.search.text().strip().lower()
        for s, fr in ds.frames.items():
            al = fr.alive()
            if f == "empty" and al:
                continue
            if f == "err" and not fr.errors:
                continue
            if f == "unsure" and not any(b.conf is not None and b.conf < 0.6 for b in al):
                continue
            if isinstance(f, int) and not any(b.c == f for b in al):
                continue
            if q and q not in s.lower():
                continue
            it = QListWidgetItem(f"{s}   [{len(al)}]")
            it.setData(Qt.UserRole, s)
            if fr.errors:
                it.setForeground(QBrush(QColor(255, 90, 90)))
            elif not al:
                it.setForeground(QBrush(QColor(140, 140, 140)))
            self.list.addItem(it)
        self.list.blockSignals(False)
        self.list_lbl.setText(f"кадров: {self.list.count()} из {len(ds.frames)}")
        row = 0
        if cur:
            for r in range(self.list.count()):
                if self.list.item(r).data(Qt.UserRole) == cur:
                    row = r
                    break
        if self.list.count():
            self.list.setCurrentRow(row)
            self.show_frame(self.list.item(row).data(Qt.UserRole))
        else:
            self.canvas.set_frame(None, None, ds.names)

    def show_frame(self, stem, select=None):
        ds = self.ds
        if not ds or stem not in ds.frames:
            return
        self.cur = stem
        self.canvas.set_frame(ds.frames[stem], ds.image(stem), ds.names, select)

    def open(self, stem, box_i=None):
        if self.dirty:
            self.reload()
        if self.filter.currentData() != "all" or self.search.text():
            self.search.blockSignals(True)
            self.search.clear()
            self.search.blockSignals(False)
            self.filter.setCurrentIndex(0)
        for r in range(self.list.count()):
            if self.list.item(r).data(Qt.UserRole) == stem:
                self.list.blockSignals(True)
                self.list.setCurrentRow(r)
                self.list.blockSignals(False)
                break
        self.show_frame(stem, box_i)
        self.canvas.setFocus()

    def step(self, d):
        r = self.list.currentRow() + d
        if 0 <= r < self.list.count():
            self.list.setCurrentRow(r)
            self.canvas.setFocus()

    def on_changed(self, snap):
        if not self.cur:
            return
        self.undo_stack.append((self.cur, snap))
        self.ds.save([self.cur])
        it = self.list.currentItem()
        if it:
            it.setText(f"{self.cur}   [{len(self.ds.frames[self.cur].alive())}]")
        self.main.dataset_changed(reload=False)

    def undo(self):
        if not self.undo_stack:
            return
        stem, snap = self.undo_stack.pop()
        if stem in self.ds.frames:
            self.ds.restore({stem: snap})
            self.open(stem)
            self.main.dataset_changed(reload=False)

    def on_palette(self, it):
        c = self.palette.row(it)
        if self.canvas.sel is not None:
            self.canvas.set_class(c)
        else:
            self.canvas.cur_class = c
        self.update_palette()
        self.canvas.setFocus()

    def update_palette(self):
        c = self.canvas.cur_class
        if self.canvas.sel is not None and self.canvas.frame:
            c = self.canvas.frame.boxes[self.canvas.sel].c
        for r in range(self.palette.count()):
            f = self.palette.item(r).font()
            f.setBold(r == c)
            self.palette.item(r).setFont(f)
            self.palette.item(r).setSelected(r == c)

    def delete_frame(self):
        if not self.cur:
            return
        r = self.list.currentRow()
        self.ds.delete_frames([self.cur])
        self.cur = None
        self.main.dataset_changed(reload=False)
        self.list.takeItem(r)
        if self.list.count():
            self.list.setCurrentRow(min(r, self.list.count() - 1))
        self.main.status("Кадр перемещён в корзину _trash (вернуть — вкладка «Статистика»)")


# ============================================================================ 5. Статистика

class Bars(QWidget):
    def __init__(self):
        super().__init__()
        self.data = []
        self.setMinimumHeight(180)

    def set(self, data):
        self.data = data
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(30, 30, 30))
        if not self.data:
            return
        mx = max(v for _, v, _ in self.data) or 1
        n = len(self.data)
        lw = 190
        bh = max(12, min(26, (self.height() - 10) // max(n, 1) - 4))
        for i, (name, v, col) in enumerate(self.data):
            y = 6 + i * (bh + 4)
            p.setPen(QColor(220, 220, 220))
            p.drawText(QRectF(4, y, lw - 8, bh), Qt.AlignVCenter | Qt.AlignRight, name)
            w = (self.width() - lw - 70) * v / mx
            p.fillRect(QRectF(lw, y, max(w, 1), bh), col)
            p.drawText(QRectF(lw + w + 6, y, 64, bh), Qt.AlignVCenter, str(v))
        self.setMinimumHeight(12 + n * (bh + 4))


class StatsTab(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(button("Обновить", lambda: self.main.dataset_changed(reload=True)))
        self.summary = QLabel()
        top.addWidget(self.summary, 1)
        lay.addLayout(top)
        split = QSplitter(Qt.Horizontal)
        lv = QWidget()
        ll = QVBoxLayout(lv)
        ll.setContentsMargins(0, 0, 0, 0)
        self.bars = Bars()
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(self.bars)
        ll.addWidget(sc, 2)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Класс", "Рамок", "Кадров", "Средний размер", "Сомнительных (<0.6)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        ll.addWidget(self.table, 2)
        split.addWidget(lv)

        rv = QWidget()
        rl = QVBoxLayout(rv)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(QLabel("<b>Проверки</b>"))
        self.warn = QTextBrowser()
        rl.addWidget(self.warn, 1)
        g = QGroupBox("Обслуживание")
        gl = QGridLayout(g)
        self.val = QSpinBox()
        self.val.setRange(5, 50)
        self.val.setValue(15)
        self.val.setSuffix(" % valid")
        gl.addWidget(self.val, 0, 0)
        gl.addWidget(button("Пересобрать train/valid", self.rebuild_lists), 0, 1)
        self.wh = QComboBox()
        for v in (320, 416, 512, 608, 640):
            self.wh.addItem(f"{v} x {v}", v)
        self.wh.setCurrentIndex(1)
        self.subdiv = QComboBox()
        for v in (8, 16, 32, 64):
            self.subdiv.addItem(f"subdivisions={v}", v)
        self.subdiv.setCurrentIndex(1)
        gl.addWidget(self.wh, 1, 0)
        gl.addWidget(self.subdiv, 1, 1)
        gl.addWidget(button("Пересоздать obj.data, obj.names и cfg", self.rebuild_cfg), 2, 0, 1, 2)
        self.trash_lbl = QLabel()
        gl.addWidget(self.trash_lbl, 3, 0)
        tb = QHBoxLayout()
        tb.addWidget(button("Вернуть всё", self.restore_trash))
        tb.addWidget(button("Очистить", self.empty_trash))
        gl.addLayout(tb, 3, 1)
        rl.addWidget(g)
        split.addWidget(rv)
        split.setSizes([700, 500])
        lay.addWidget(split, 1)

    @property
    def ds(self):
        return self.main.dataset

    def showEvent(self, e):
        super().showEvent(e)
        self.refresh()

    def refresh(self):
        ds = self.ds
        if ds is None:
            self.summary.setText("Датасет не выбран")
            return
        cnt, fw = ds.count()
        sizes, unsure = defaultdict(list), Counter()
        empty = errs = 0
        tiny = []
        for fr in ds.frames.values():
            al = fr.alive()
            empty += not al
            errs += bool(fr.errors)
            for b in al:
                sizes[b.c].append((b.w, b.h))
                if b.conf is not None and b.conf < 0.6:
                    unsure[b.c] += 1
                if b.w * 416 < 6 or b.h * 416 < 6:
                    tiny.append(fr.stem)
        n = len(ds.frames)
        self.summary.setText(f"<b>{ds.root}</b> — кадров {n}, рамок {sum(cnt.values())}, пустых {empty}")
        self.bars.set([(nm, cnt[i], class_qcolor(i)) for i, nm in enumerate(ds.names)])
        self.table.setRowCount(len(ds.names))
        for i, nm in enumerate(ds.names):
            s = sizes[i]
            avg = f"{np.mean([w for w, _ in s]) * 100:.1f}% x {np.mean([h for _, h in s]) * 100:.1f}%" if s else "—"
            for col, v in enumerate((nm, cnt[i], fw[i], avg, unsure[i])):
                it = QTableWidgetItem(str(v))
                if col == 0:
                    it.setForeground(QBrush(class_qcolor(i)))
                self.table.setItem(i, col, it)
        w = []
        if not ds.names:
            w.append("<span style='color:#f66'>✖ Нет obj.names</span>")
        if errs:
            w.append(f"<span style='color:#f66'>✖ Кадров с ошибками разметки: {errs}</span> — фильтр «С ошибками» "
                     f"во вкладке «Кадры»")
        for i, nm in enumerate(ds.names):
            if cnt[i] == 0:
                w.append(f"<span style='color:#f66'>✖ Класс «{nm}»: нет ни одной рамки</span> — сеть его не выучит")
            elif cnt[i] < 100:
                w.append(f"<span style='color:#fc6'>▲ Класс «{nm}»: всего {cnt[i]} рамок</span> — желательно 300+")
        vals = [cnt[i] for i in range(len(ds.names)) if cnt[i]]
        if vals and max(vals) / min(vals) > 10:
            w.append("<span style='color:#fc6'>▲ Сильный перекос классов</span> — редкие будут распознаваться хуже")
        if n and empty / n > 0.3:
            w.append(f"<span style='color:#fc6'>▲ Пустых кадров {empty / n:.0%}</span> — многовато (норма до 10–20%)")
        if n and empty == 0:
            w.append("<span style='color:#fc6'>▲ Нет пустых кадров</span> — немного негативов снижает ложные "
                     "срабатывания")
        if tiny:
            w.append(f"<span style='color:#fc6'>▲ Очень мелких рамок: {len(tiny)}</span> (меньше 6 px при 416) — "
                     f"сортировка «по размеру» во вкладке «Чистка»")
        su = sum(unsure.values())
        if su:
            w.append(f"▲ Сомнительных рамок (уверенность &lt; 0.6): {su} — проверьте во вкладке «Чистка» "
                     f"(сортировка по уверенности)")
        lists = {}
        for lst in ("train.txt", "valid.txt"):
            f = ds.root / lst
            rows = [r.strip() for r in f.read_text(encoding="utf-8").splitlines() if r.strip()] if f.exists() else []
            miss = [r for r in rows if not (ds.obj / Path(r).name).exists()]
            lists[lst] = len(rows)
            if not f.exists():
                w.append(f"<span style='color:#f66'>✖ Нет {lst}</span> — нажмите «Пересобрать train/valid»")
            elif miss:
                w.append(f"<span style='color:#f66'>✖ {lst}: {len(miss)} строк на несуществующие файлы</span> — "
                         f"«Пересобрать train/valid»")
        listed = lists.get("train.txt", 0) + lists.get("valid.txt", 0)
        if listed and listed != n:
            w.append(f"▲ В train+valid {listed} строк, а кадров {n} — «Пересобрать train/valid»")
        if not (ds.root / "yolov4-tiny-custom.cfg").exists():
            w.append("▲ Нет yolov4-tiny-custom.cfg — «Пересоздать obj.data, obj.names и cfg»")
        if not w:
            w.append("<span style='color:#6c6'>✔ Ошибок не найдено</span>")
        w.append(f"<br>train: {lists.get('train.txt', 0)}, valid: {lists.get('valid.txt', 0)}")
        self.warn.setHtml("<br>".join(w))
        k = len(list(ds.trash.glob("*.jpg"))) if ds.trash.is_dir() else 0
        self.trash_lbl.setText(f"Корзина: {k} кадров")

    def rebuild_lists(self):
        if self.ds is None:
            return
        tr, va = self.ds.rebuild_lists(self.val.value() / 100)
        self.main.status(f"train {tr}, valid {va}")
        self.refresh()

    def rebuild_cfg(self):
        ds = self.ds
        if ds is None:
            return
        names = ds.names or self.main.classes_tab.names()
        if not names:
            return
        a = type("A", (), {})()
        a.width = a.height = self.wh.currentData()
        a.batch, a.subdivisions = 64, self.subdiv.currentData()
        AL.write_meta(ds.root, names, a)
        self.main.status(f"obj.data, obj.names и yolov4-tiny-custom.cfg пересозданы ({len(names)} классов)")
        self.refresh()

    def restore_trash(self):
        ds = self.ds
        if ds is None or not ds.trash.is_dir():
            return
        n = 0
        for p in list(ds.trash.iterdir()):
            shutil.move(str(p), str(ds.obj / p.name))
            n += p.suffix.lower() in IMG_EXT
        self.main.dataset_changed(reload=True)
        self.main.status(f"Возвращено кадров: {n}. Нажмите «Пересобрать train/valid»")

    def empty_trash(self):
        ds = self.ds
        if ds is None or not ds.trash.is_dir():
            return
        if QMessageBox.question(self, "Корзина", "Удалить кадры из корзины навсегда?") == QMessageBox.Yes:
            shutil.rmtree(ds.trash, ignore_errors=True)
            self.refresh()


# ============================================================================ 6. Обучение

COLAB = """<h3>Обучение YOLOv4-tiny в Google Colab (бесплатная видеокарта)</h3>
<ol>
<li>Нажмите <b>«Собрать dataset.zip»</b> (ниже) и проверьте вкладку «Статистика» — без красных ошибок.</li>
<li>Откройте <a href="https://colab.research.google.com">colab.research.google.com</a> → «Создать блокнот» →
«Среда выполнения → Сменить среду → T4 GPU».</li>
<li>Выполните ячейки по очереди:</li></ol>
<pre>
!git clone https://github.com/AlexeyAB/darknet
%cd darknet
!sed -i 's/OPENCV=0/OPENCV=1/; s/GPU=0/GPU=1/; s/CUDNN=0/CUDNN=1/; s/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
!make -j4
!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29

from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/yolo_backup
</pre>
<p>Загрузите <b>dataset.zip</b> в папку <code>darknet</code> (панель файлов слева), затем:</p>
<pre>
!unzip -q -o dataset.zip -d data/
!cp data/yolov4-tiny-custom.cfg cfg/
!sed -i 's#backup = backup/#backup = /content/drive/MyDrive/yolo_backup/#' data/obj.data
!./darknet detector train data/obj.data cfg/yolov4-tiny-custom.cfg yolov4-tiny.conv.29 -dont_show -map
</pre>
<p>Продолжить после отключения Colab:</p>
<pre>
!./darknet detector train data/obj.data cfg/yolov4-tiny-custom.cfg /content/drive/MyDrive/yolo_backup/yolov4-tiny-custom_last.weights -dont_show -map
</pre>
<p>Проверка:</p>
<pre>
!./darknet detector map data/obj.data cfg/yolov4-tiny-custom.cfg /content/drive/MyDrive/yolo_backup/yolov4-tiny-custom_best.weights
!./darknet detector demo data/obj.data cfg/yolov4-tiny-custom.cfg /content/drive/MyDrive/yolo_backup/yolov4-tiny-custom_best.weights test.mp4 -dont_show -out_filename result.avi
</pre>
<p>Результат: <b>yolov4-tiny-custom_best.weights</b> + <b>yolov4-tiny-custom.cfg</b> + <b>obj.names</b>.
mAP@0.5 выше 0.8–0.9 — хорошо.</p>
"""


class TrainTab(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        lay = QVBoxLayout(self)
        b = QHBoxLayout()
        b.addWidget(button("Собрать dataset.zip", self.make_zip))
        b.addWidget(button("Открыть папку датасета", lambda: self.main.dataset and open_folder(self.main.dataset.root)))
        b.addStretch()
        lay.addLayout(b)
        t = QTextBrowser()
        t.setOpenExternalLinks(True)
        t.setHtml(COLAB)
        lay.addWidget(t)

    def make_zip(self):
        ds = self.main.dataset
        if ds is None:
            return
        if not (ds.root / "train.txt").exists():
            ds.rebuild_lists(0.15)
        files = [ds.root / n for n in ("obj.data", "obj.names", "train.txt", "valid.txt", "yolov4-tiny-custom.cfg")
                 if (ds.root / n).exists()]
        objs = [p for p in ds.obj.iterdir() if p.suffix.lower() in IMG_EXT | {".txt"}]
        dst, _ = QFileDialog.getSaveFileName(self, "Сохранить архив", str(ds.root.parent / "dataset.zip"), "ZIP (*.zip)")
        if not dst:
            return
        dlg = QProgressDialog("Собираю архив...", "Отмена", 0, len(objs) + len(files), self)
        dlg.setWindowModality(Qt.WindowModal)
        with zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED) as z:
            for k, p in enumerate(files + objs):
                if dlg.wasCanceled():
                    break
                z.write(p, p.name if p.parent == ds.root else f"obj/{p.name}")
                if k % 50 == 0:
                    dlg.setValue(k)
                    QApplication.processEvents()
        dlg.setValue(dlg.maximum())
        self.main.status(f"Архив готов: {dst}")


# ============================================================================ 7. Сеть и модели

class NetTab(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        lay = QVBoxLayout(self)
        g = QGroupBox("Сеть")
        gf = QFormLayout(g)
        s = self.main.settings.get("net", {})
        self.no_proxy = QCheckBox("Не использовать прокси (прямое подключение)")
        self.no_proxy.setChecked(s.get("no_proxy", True))
        self.no_proxy.setToolTip("Игнорировать прокси из настроек Windows, переменных среды и pip")
        gf.addRow(self.no_proxy)
        self.proxy = QLineEdit(s.get("proxy", ""))
        self.proxy.setPlaceholderText("пусто — искать автоматически; например 10.255.201.171:8080")
        self.proxy.setEnabled(not self.no_proxy.isChecked())
        gf.addRow("Прокси:", self.proxy)
        self.no_proxy.toggled.connect(self.on_no_proxy)
        self.offline = QCheckBox("Ничего не скачивать (только уже скачанные модели)")
        self.offline.setChecked(s.get("offline", False))
        gf.addRow(self.offline)
        self.models = QLineEdit(s.get("models", ""))
        self.models.setPlaceholderText(f"по умолчанию {APP_DIR / 'models'}")
        mb = QHBoxLayout()
        mb.addWidget(self.models)
        mb.addWidget(button("...", self.choose_models))
        mb.addWidget(button("Открыть", lambda: open_folder(self.models.text() or APP_DIR / "models")))
        gf.addRow("Папка моделей:", mb)
        for w in (self.proxy, self.models):
            w.editingFinished.connect(self.store)
        self.offline.toggled.connect(self.store)
        lay.addWidget(g)
        b = QHBoxLayout()
        b.addWidget(button("Проверить сеть и прокси", lambda: self.run(["net"] + self.net_args(False))))
        self.preset = QComboBox()
        for p in ("fast", "balanced", "best"):
            self.preset.addItem(f"модели пресета {p}", p)
        self.preset.setCurrentIndex(1)
        b.addWidget(self.preset)
        b.addWidget(button("Скачать модели", lambda: self.run(["download", "--preset", self.preset.currentData()]
                                                               + self.net_args(True))))
        b.addWidget(button("Проверить видеокарту", self.check_gpu))
        b.addStretch()
        lay.addLayout(b)
        self.runner = Runner()
        lay.addWidget(self.runner, 1)

    def store(self):
        self.main.settings["net"] = {"proxy": self.proxy.text().strip(), "offline": self.offline.isChecked(),
                                     "models": self.models.text().strip(),
                                     "no_proxy": self.no_proxy.isChecked()}

    def on_no_proxy(self, on):
        self.proxy.setEnabled(not on)
        self.store()
        if on:
            AL.disable_proxy()
        else:
            AL.NO_PROXY_MODE = False
            os.environ.pop("AUTOLABEL_NO_PROXY", None)
            for k in ("NO_PROXY", "no_proxy", "PIP_PROXY"):
                os.environ.pop(k, None)
        self.main.status("Прокси не используется" if on else "Прокси: автоматически или из поля «Прокси»")

    def choose_models(self):
        d = QFileDialog.getExistingDirectory(self, "Папка моделей", self.models.text() or str(APP_DIR))
        if d:
            self.models.setText(d)
            self.store()

    def net_args(self, with_models):
        self.store()
        a = []
        if self.no_proxy.isChecked():
            a.append("--no-proxy")
        elif self.proxy.text().strip():
            a += ["--proxy", self.proxy.text().strip()]
        if with_models and self.models.text().strip():
            a += ["--models", self.models.text().strip()]
        return a

    def run(self, args):
        if not self.runner.start(args):
            self.main.status("Уже что-то выполняется")

    def check_gpu(self):
        code = ("import sys\n"
                "try:\n import torch\nexcept ImportError:\n print('PyTorch не установлен'); sys.exit()\n"
                "print('Python', sys.version.split()[0], '| PyTorch', torch.__version__, '| CUDA в сборке:', "
                "torch.version.cuda)\n"
                "ok = torch.cuda.is_available()\n"
                "print('Видеокарта доступна:', ok)\n"
                "print(torch.cuda.get_device_name(0) if ok else 'Работать будет на процессоре. Если есть NVIDIA — '"
                "'поставьте PyTorch с CUDA (сборка cu118/cu121) и при необходимости обновите драйвер.')\n")
        self.runner.start(["-c", code], script=None)


# ============================================================================ 8. VLM-судья

class VLMTab(QWidget):
    MODES = (("Выключена", "off"), ("Локально — Ollama (бесплатно, на вашей видеокарте)", "ollama"),
             ("API — агрегатор / OpenAI-совместимый (платно, нужен интернет)", "openai"))

    def __init__(self, main):
        super().__init__()
        self.main = main
        self.task = None
        s = self.main.settings.get("vlm", {})
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        inner = QWidget()
        scroll.setWidget(inner)
        lay = QVBoxLayout(inner)
        if VLM is None:
            lay.addWidget(QLabel("<b style='color:#f66'>Рядом нет vlm.py — положите его в папку с приложением.</b>"))

        info = QLabel("<b>VLM-судья</b> — визуальная языковая модель смотрит на вырезку и говорит, какой это класс "
                      "или «ничего из списка». При разметке ей отдаются <b>только сомнительные рамки</b> — "
                      "это убирает путаницу похожих знаков и ложные срабатывания почти без потери скорости. "
                      "Во вкладке «Чистка» можно проверить выделенные вырезки кнопкой «Проверить через VLM».")
        info.setWordWrap(True)
        lay.addWidget(info)

        top = QFormLayout()
        self.mode = QComboBox()
        for t, v in self.MODES:
            self.mode.addItem(t, v)
        self.mode.setCurrentIndex(max(0, self.mode.findData(s.get("mode", "off"))))
        self.mode.currentIndexChanged.connect(self.on_mode)
        top.addRow("<b>Режим:</b>", self.mode)
        self.check = QComboBox()
        self.check.addItem("только сомнительные рамки (рекомендуется)", "uncertain")
        self.check.addItem("все рамки (медленно, но строже)", "all")
        self.check.setCurrentIndex(max(0, self.check.findData(s.get("check", "uncertain"))))
        top.addRow("При разметке проверять:", self.check)
        self.context = QLineEdit(s.get("context", ""))
        self.context.setPlaceholderText("подсказка о сцене (англ.), напр.: Russian road signs on a toy model track")
        top.addRow("Подсказка о сцене:", self.context)
        lay.addLayout(top)

        # ---------------- локально
        self.g_local = QGroupBox("Локально через Ollama")
        gl = QVBoxLayout(self.g_local)
        st = QHBoxLayout()
        self.ollama_lbl = QLabel("Ollama: не проверено")
        st.addWidget(self.ollama_lbl, 1)
        st.addWidget(button("Проверить", self.check_ollama))
        st.addWidget(button("Установить Ollama", self.install_ollama, "Скачать и запустить установщик с ollama.com"))
        gl.addLayout(st)
        self.models = QTableWidget(0, 5)
        self.models.setHorizontalHeaderLabels(["Модель", "Скачать", "Видеопамять", "Описание", "Скачана"])
        self.models.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        for i in (0, 1, 2, 4):
            self.models.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self.models.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.models.setSelectionMode(QAbstractItemView.SingleSelection)
        self.models.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.models.verticalHeader().setVisible(False)
        for row in (VLM.LOCAL_MODELS if VLM else []):
            r = self.models.rowCount()
            self.models.insertRow(r)
            for c, v in enumerate(row):
                self.models.setItem(r, c, QTableWidgetItem(v))
            self.models.setItem(r, 4, QTableWidgetItem(""))
        self.models.itemSelectionChanged.connect(self.on_pick_local)
        self.models.setMinimumHeight(30 + 30 * self.models.rowCount())
        gl.addWidget(self.models)
        f = QFormLayout()
        self.local_model = QLineEdit(s.get("local_model", "qwen3-vl:4b"))
        self.local_model.setToolTip("Можно вписать любую модель Ollama с поддержкой картинок")
        mb = QHBoxLayout()
        mb.addWidget(self.local_model)
        mb.addWidget(button("Скачать модель", self.pull))
        f.addRow("Модель:", mb)
        self.local_url = QLineEdit(s.get("local_url", "http://localhost:11434"))
        f.addRow("Адрес Ollama:", self.local_url)
        gl.addLayout(f)
        self.pull_bar = QProgressBar()
        self.pull_bar.setVisible(False)
        gl.addWidget(self.pull_bar)
        lay.addWidget(self.g_local)

        # ---------------- API
        self.g_api = QGroupBox("API — российские агрегаторы (оплата в рублях) и другие OpenAI-совместимые")
        ga = QFormLayout(self.g_api)
        self.provider = QComboBox()
        for name, url, model, site in (VLM.API_PROVIDERS if VLM else []):
            self.provider.addItem(name, (url, model, site))
        i = self.provider.findText(s.get("provider", "ProxyAPI"))
        self.provider.setCurrentIndex(max(0, i))
        self.provider.currentIndexChanged.connect(self.on_provider)
        pb = QHBoxLayout()
        pb.addWidget(self.provider, 1)
        self.site_btn = button("Сайт / получить ключ", self.open_site)
        pb.addWidget(self.site_btn)
        ga.addRow("Сервис:", pb)
        self.url = QLineEdit(s.get("url", ""))
        ga.addRow("Адрес (base_url):", self.url)
        self.key = QLineEdit(s.get("key", ""))
        self.key.setEchoMode(QLineEdit.Password)
        self.key.setPlaceholderText("ключ API из личного кабинета сервиса")
        kb = QHBoxLayout()
        kb.addWidget(self.key)
        show = QToolButton()
        show.setText("👁")
        show.setCheckable(True)
        show.toggled.connect(lambda on: self.key.setEchoMode(QLineEdit.Normal if on else QLineEdit.Password))
        kb.addWidget(show)
        ga.addRow("Ключ:", kb)
        self.api_model = QComboBox()
        self.api_model.setEditable(True)
        for m in (VLM.API_MODEL_HINTS if VLM else []):
            self.api_model.addItem(m)
        self.api_model.setCurrentText(s.get("model", "openai/gpt-4o-mini"))
        mb = QHBoxLayout()
        mb.addWidget(self.api_model, 1)
        mb.addWidget(button("Список моделей", self.list_models, "Загрузить список моделей с сервиса"))
        ga.addRow("Модель (с картинками):", mb)
        self.workers = QSpinBox()
        self.workers.setRange(1, 16)
        self.workers.setValue(s.get("workers", 4))
        ga.addRow("Параллельных запросов:", self.workers)
        note = QLabel("Ключ хранится только в autolabel_app.json на этом компьютере. Модель должна понимать "
                      "картинки (vision): gpt-4o-mini, gpt-4.1-mini, claude, gemini, qwen-vl. Точное имя "
                      "у каждого сервиса своё — нажмите «Список моделей». Через API уходят только вырезки "
                      "объектов, не целые кадры.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#aaa")
        ga.addRow(note)
        lay.addWidget(self.g_api)

        tb = QHBoxLayout()
        tb.addWidget(button("Проверить на картинке...", self.test_image,
                            "Выберите вырезку объекта — модель скажет, какой это класс"))
        tb.addStretch()
        lay.addLayout(tb)
        self.out = Console()
        self.out.setMaximumHeight(170)
        lay.addWidget(self.out)
        lay.addStretch()
        if not self.url.text():
            self.on_provider()
        self.on_mode()

    # ---------------------------------------------------------------- настройки
    def store(self):
        self.main.settings["vlm"] = {
            "mode": self.mode.currentData(), "check": self.check.currentData(), "context": self.context.text(),
            "local_model": self.local_model.text().strip(), "local_url": self.local_url.text().strip(),
            "provider": self.provider.currentText(), "url": self.url.text().strip(), "key": self.key.text().strip(),
            "model": self.api_model.currentText().strip(), "workers": self.workers.value()}
        os.environ["AUTOLABEL_VLM_KEY"] = self.key.text().strip()

    def on_mode(self):
        m = self.mode.currentData()
        self.g_local.setEnabled(m == "ollama")
        self.g_api.setEnabled(m == "openai")
        self.store()
        rt = getattr(self.main, "run_tab", None)
        if rt is not None and hasattr(rt, "vlm_lbl"):
            rt.vlm_lbl.setText(self.summary())

    def summary(self):
        m = self.mode.currentData()
        if m == "ollama":
            return f"VLM: {self.local_model.text()} (локально)"
        if m == "openai":
            return f"VLM: {self.api_model.currentText()} ({self.provider.currentText()})"
        return "VLM: выключена (вкладка 8)"

    def on_provider(self):
        url, model, site = self.provider.currentData() or ("", "", "")
        if url:
            self.url.setText(url)
        if model:
            self.api_model.setCurrentText(model)
        self.site_btn.setEnabled(bool(site))
        self.store()

    def open_site(self):
        _, _, site = self.provider.currentData() or ("", "", "")
        if site:
            QDesktopServices.openUrl(QUrl(site))

    def on_pick_local(self):
        r = self.models.currentRow()
        if r >= 0:
            self.local_model.setText(self.models.item(r, 0).text())
            self.store()

    def run_args(self):
        self.store()
        m = self.mode.currentData()
        if m == "off" or VLM is None:
            return []
        a = ["--vlm", m, "--vlm-mode", self.check.currentData()]
        if self.context.text().strip():
            a += ["--vlm-context", self.context.text().strip()]
        if m == "ollama":
            a += ["--vlm-model", self.local_model.text().strip(), "--vlm-url", self.local_url.text().strip()]
        else:
            a += ["--vlm-model", self.api_model.currentText().strip(), "--vlm-url", self.url.text().strip(),
                  "--vlm-workers", str(self.workers.value())]
        return a

    def make_judge(self, names=None):
        """VLMJudge по текущим настройкам; names — порядок классов датасета."""
        self.store()
        m = self.mode.currentData()
        if VLM is None or m == "off":
            QMessageBox.information(self, "VLM", "Выберите режим VLM во вкладке «8. VLM-судья»")
            self.main.tabs.setCurrentWidget(self)
            return None
        classes = normalize_classes(self.main.classes_tab.to_config()["classes"], names)
        if m == "ollama":
            return VLM.VLMJudge("ollama", self.local_model.text().strip(), classes, self.local_url.text().strip(),
                                timeout=180, context=self.context.text().strip())
        if not self.url.text().strip():
            QMessageBox.warning(self, "VLM", "Укажите адрес API")
            return None
        proxy = None if self.main.settings.get("net", {}).get("no_proxy", True) else \
            (self.main.settings.get("net", {}).get("proxy") or None)
        return VLM.VLMJudge("openai", self.api_model.currentText().strip(), classes, self.url.text().strip(),
                            self.key.text().strip(), proxy, workers=self.workers.value(),
                            context=self.context.text().strip())

    # ---------------------------------------------------------------- действия
    def _run(self, fn, done):
        if self.task and self.task.isRunning():
            self.main.status("Подождите, предыдущая операция ещё идёт")
            return
        self.task = Task(fn)
        self.task.msg.connect(lambda t: self.out.add(t + "\n"))
        self.task.done.connect(done)
        self.task.start()

    def check_ollama(self):
        url = self.local_url.text().strip()
        self.ollama_lbl.setText("Ollama: проверяю...")

        def done(res):
            ok, have, err = res if isinstance(res, tuple) else (False, [], str(res))
            if ok:
                self.ollama_lbl.setText(f"<span style='color:#6c6'>Ollama работает</span>, скачано моделей: {len(have)}")
            else:
                self.ollama_lbl.setText("<span style='color:#f66'>Ollama не найдена</span> — установите и запустите "
                                        "(значок появится в трее)")
            for r in range(self.models.rowCount()):
                tag = self.models.item(r, 0).text()
                self.models.item(r, 4).setText("✓" if tag in have else "")
        self._run(lambda t: VLM.ollama_status(url), done)

    def install_ollama(self):
        if sys.platform != "win32":
            QDesktopServices.openUrl(QUrl("https://ollama.com/download"))
            return
        dst = Path(os.environ.get("TEMP", str(APP_DIR))) / "OllamaSetup.exe"
        self.out.add("Скачиваю установщик Ollama...\n")

        def fn(t):
            s = VLM._session()
            with s.get("https://ollama.com/download/OllamaSetup.exe", stream=True, timeout=(15, 600)) as r:
                r.raise_for_status()
                with open(dst, "wb") as f:
                    for ch in r.iter_content(1 << 20):
                        f.write(ch)
            return dst

        def done(res):
            if isinstance(res, Exception):
                self.out.add(f"Не скачалось ({res}). Открываю сайт — скачайте вручную.\n")
                QDesktopServices.openUrl(QUrl("https://ollama.com/download"))
            else:
                self.out.add("Запускаю установщик. После установки нажмите «Проверить».\n")
                QProcess.startDetached(str(res), [])
        self._run(fn, done)

    def pull(self):
        model, url = self.local_model.text().strip(), self.local_url.text().strip()
        if not model:
            return
        self.pull_bar.setVisible(True)
        self.pull_bar.setRange(0, 100)
        self.out.add(f"Скачиваю {model} в Ollama...\n")

        def fn(t):
            last = [0.0]

            def prog(st, pct):
                if time.time() - last[0] > 0.5:
                    last[0] = time.time()
                    t.item.emit((st, pct))
            VLM.ollama_pull(model, url, prog)
            return True

        def done(res):
            self.pull_bar.setVisible(False)
            self.out.add(("Готово.\n" if res is True else f"Ошибка: {res}\nЗапущена ли Ollama?\n"))
            self.check_ollama()
        self._run(fn, done)
        self.task.item.connect(lambda v: (self.pull_bar.setValue(int(v[1] or 0)),
                                          self.pull_bar.setFormat(f"{v[0]} {v[1] or 0:.0f}%")))

    def list_models(self):
        url, key = self.url.text().strip(), self.key.text().strip()

        def done(res):
            if isinstance(res, Exception):
                self.out.add(f"Не удалось получить список: {res}\n")
                return
            cur = self.api_model.currentText()
            self.api_model.clear()
            vis = [m for m in res if any(k in m.lower() for k in ("gpt-4o", "gpt-4.1", "gpt-5", "claude", "gemini",
                                                                   "vl", "vision", "pixtral", "llava", "gemma-3",
                                                                   "gemma3", "grok", "glm-4v"))]
            for m in vis + [m for m in res if m not in vis]:
                self.api_model.addItem(m)
            self.api_model.setCurrentText(cur)
            self.out.add(f"Моделей: {len(res)}, из них похожих на «с картинками» — {len(vis)} (вверху списка)\n")
        self._run(lambda t: VLM.api_models(url, key), done)

    def test_image(self):
        f, _ = QFileDialog.getOpenFileName(self, "Вырезка объекта", str(APP_DIR), "Картинки (*.jpg *.jpeg *.png *.bmp)")
        if not f:
            return
        img = AL.imread_u(f)
        judge = self.make_judge(self.main.dataset.names if self.main.dataset and self.main.dataset.names else None)
        if img is None or judge is None:
            return
        self.out.add(f"Спрашиваю {judge.describe()}...\n")
        t0 = time.time()

        def done(res):
            k, q, raw = res if isinstance(res, tuple) else (None, 0, str(res))
            name = "ОШИБКА" if k is None else ("ничего из списка" if k == -1 else judge.names[k])
            self.out.add(f"→ {name}  (уверенность {q:.2f}, {time.time() - t0:.1f} с)\n   ответ: {raw}\n")
        self._run(lambda t: judge.ask(img), done)

    def showEvent(self, e):
        super().showEvent(e)
        if self.mode.currentData() == "ollama":
            self.check_ollama()


# ============================================================================ главное окно

class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = load_settings()
        self.dataset = None
        self.setWindowTitle("Autolabel Studio — авторазметка датасета для Darknet")
        self.resize(1500, 920)

        top = QWidget()
        tl = QHBoxLayout(top)
        tl.setContentsMargins(6, 4, 6, 0)
        tl.addWidget(QLabel("<b>Датасет:</b>"))
        self.ds_edit = QLineEdit()
        self.ds_edit.setReadOnly(True)
        tl.addWidget(self.ds_edit, 1)
        tl.addWidget(button("Открыть другой...", self.choose_dataset))
        tl.addWidget(button("Перечитать", lambda: self.dataset_changed(reload=True)))

        self.tabs = QTabWidget()
        self.classes_tab = ClassesTab(self)
        self.run_tab = RunTab(self)
        self.clean_tab = CleanTab(self)
        self.frames_tab = FramesTab(self)
        self.stats_tab = StatsTab(self)
        self.train_tab = TrainTab(self)
        self.net_tab = NetTab(self)
        self.vlm_tab = VLMTab(self)
        for w, t in ((self.classes_tab, "1. Классы и образцы"), (self.run_tab, "2. Разметка"),
                     (self.clean_tab, "3. Чистка"), (self.frames_tab, "4. Кадры"),
                     (self.stats_tab, "5. Статистика"), (self.train_tab, "6. Обучение"),
                     (self.net_tab, "7. Сеть и модели"), (self.vlm_tab, "8. VLM-судья")):
            self.tabs.addTab(w, t)
        central = QWidget()
        cl = QVBoxLayout(central)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(top)
        cl.addWidget(self.tabs)
        self.setCentralWidget(central)
        self.statusBar()

        cls = self.settings.get("classes")
        for p in ([cls] if cls else []) + [str(APP_DIR / "classes.yaml")]:
            if p and Path(p).exists() and self.classes_tab.load(p):
                break
        else:
            self.classes_tab.new_example()
        self.set_dataset(self.settings.get("dataset") or str(APP_DIR / "dataset"))
        self.tabs.setCurrentIndex(self.settings.get("tab", 1))

    def status(self, text):
        self.statusBar().showMessage(text, 15000)

    def set_dataset(self, path):
        path = Path(path)
        self.settings["dataset"] = str(path)
        self.ds_edit.setText(str(path))
        self.dataset = Dataset(path)
        self.dataset.load()
        self.frames_tab.mark_dirty()
        self.clean_tab.loaded_version = -1
        cur = self.tabs.currentWidget()
        if cur in (self.clean_tab, self.frames_tab, self.stats_tab):
            if cur is self.clean_tab:
                self.clean_tab.refresh()
            elif cur is self.frames_tab:
                self.frames_tab.reload()
            else:
                self.stats_tab.refresh()

    def choose_dataset(self):
        d = QFileDialog.getExistingDirectory(self, "Папка датасета (где obj/ и obj.names)",
                                             self.settings.get("dataset", str(APP_DIR)))
        if d:
            self.set_dataset(d)
            self.run_tab.out.setText(d)

    def dataset_changed(self, reload=False):
        """reload=True — перечитать с диска (после разметки); иначе изменения уже в памяти."""
        if reload or self.dataset is None or str(self.dataset.root) != self.settings.get("dataset"):
            self.set_dataset(self.settings.get("dataset") or self.run_tab.out.text())
        else:
            self.dataset.version += 1
            if self.tabs.currentWidget() is not self.frames_tab:
                self.frames_tab.mark_dirty()

    def save_settings(self):
        self.run_tab.store_settings()
        self.vlm_tab.store()
        self.net_tab.store()
        self.settings["tab"] = self.tabs.currentIndex()
        save_settings(self.settings)

    def closeEvent(self, e):
        busy = self.run_tab.runner.running()
        if busy and QMessageBox.question(self, "Выход", "Разметка ещё идёт. Остановить и выйти?") != QMessageBox.Yes:
            e.ignore()
            return
        if busy:
            self.run_tab.stop()
            self.run_tab.runner.proc.waitForFinished(15000)
            self.run_tab.runner.kill()
        self.net_tab.runner.kill()
        self.save_settings()
        e.accept()


def dark_palette(app):
    app.setStyle("Fusion")
    p = QPalette()
    for role, c in ((QPalette.Window, "#2b2b2b"), (QPalette.WindowText, "#e6e6e6"), (QPalette.Base, "#1f1f1f"),
                    (QPalette.AlternateBase, "#2b2b2b"), (QPalette.ToolTipBase, "#333"),
                    (QPalette.ToolTipText, "#eee"), (QPalette.Text, "#e6e6e6"), (QPalette.Button, "#353535"),
                    (QPalette.ButtonText, "#e6e6e6"), (QPalette.BrightText, "#ff5555"),
                    (QPalette.Highlight, "#2f6fb3"), (QPalette.HighlightedText, "#ffffff"),
                    (QPalette.Link, "#6aa9ff")):
        p.setColor(role, QColor(c))
    p.setColor(QPalette.Disabled, QPalette.Text, QColor("#777"))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#777"))
    app.setPalette(p)
    app.setStyleSheet("QGroupBox{font-weight:bold; border:1px solid #444; border-radius:4px; margin-top:8px;"
                      "padding-top:6px} QGroupBox::title{subcontrol-origin:margin; left:8px}"
                      "QToolTip{color:#eee; background:#333; border:1px solid #555}")


def main():
    if hasattr(Qt, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    dark_palette(app)
    w = Main()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
