#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
webcam_test.py — проверка обученной нейросети YOLO на веб-камере, видео или картинках.

Поддерживает:
  • YOLOv4 / YOLOv4-tiny / YOLOv3 (Darknet): .weights + .cfg + .names — через OpenCV DNN,
    Darknet собирать не нужно, на процессоре ~20–60 кадр/с (нужен opencv-python 4.x)
  • PyTorch .pt (YOLOv5u / v8 / v9 / v10 / 11 / 12 от Ultralytics), а также .onnx — через ultralytics
    (pip install ultralytics; видеокарта используется сама, если есть). Старые .pt из репозитория
    yolov5 тоже пробуются (через torch.hub, нужен интернет при первом запуске).

Запуск:
  python webcam_test.py                         ищет модель сам (папка model/ или рядом), камера 0
  python webcam_test.py --cam 1                 другая камера
  python webcam_test.py --source video.mp4      видео (или картинка, или папка с картинками)
  python webcam_test.py --weights model/yolov4-tiny-custom_best.weights --cfg model/yolov4-tiny-custom.cfg
                        --names model/obj.names
  python webcam_test.py --weights best.pt       модель PyTorch (YOLOv8/11 и т.п.)

Клавиши в окне:
  Q / Esc  выход           Пробел  пауза          S  снимок в shots/
  R        запись видео    + / -   порог ±0.05    F  во весь экран
  ← / →    (пауза, видео) кадр назад/вперёд
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_model(a):
    """Ищет cfg/weights/names: аргументы → папка model/ → рядом со скриптом → dataset/."""
    places = [Path(a.model_dir)] if a.model_dir else []
    places += [HERE / "model", HERE, HERE / "dataset", Path.cwd() / "model", Path.cwd()]

    def pick(pattern, prefer=()):
        for d in places:
            if not d.is_dir():
                continue
            files = sorted(d.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
            for key in prefer:
                for f in files:
                    if key in f.name:
                        return f
            if files:
                return files[0]
        return None

    if a.weights:
        weights = Path(a.weights)
    else:  # самый свежий из .weights / .pt / .onnx, «best» в приоритете
        cands = [f for f in (pick("*.weights", ("_best", "_last", "_final")), pick("*.pt", ("best", "last")),
                             pick("*.onnx", ("best",))) if f]
        cands.sort(key=lambda f: ("best" not in f.name, -f.stat().st_mtime))
        weights = cands[0] if cands else None
    if not weights or not weights.exists():
        sys.exit("[!] Модель не найдена. Положите в папку model рядом со скриптом:\n"
                 "    Darknet: yolov4-tiny-custom_best.weights + yolov4-tiny-custom.cfg + obj.names\n"
                 "    или PyTorch: best.pt\n"
                 "    или укажите --weights путь")
    names = Path(a.names) if a.names else pick("*.names")
    if weights.suffix.lower() != ".weights":
        return None, weights, names
    cfg = Path(a.cfg) if a.cfg else pick("*.cfg", ("custom",))
    if not cfg or not cfg.exists():
        sys.exit("[!] Для .weights нужен .cfg (yolov4-tiny-custom.cfg) — положите рядом или укажите --cfg")
    return cfg, weights, names


def load_net(cfg, weights, gpu):
    if not hasattr(cv2.dnn, "readNetFromDarknet"):
        sys.exit(f"[!] В OpenCV {cv2.__version__} убрана поддержка Darknet. Поставьте 4.x:\n"
                 f'    pip install "opencv-python>=4.8,<5"')
    # через буферы — работает и с русскими буквами в пути
    net = cv2.dnn.readNetFromDarknet(np.fromfile(str(cfg), np.uint8), np.fromfile(str(weights), np.uint8))
    backend = "CPU"
    if gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        backend = "CUDA"
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net, backend


def cfg_size(cfg):
    w = h = 416
    for line in Path(cfg).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.split("#")[0].strip().replace(" ", "")
        if line.startswith("width="):
            w = int(line[6:])
        elif line.startswith("height="):
            h = int(line[7:])
        elif line.startswith("[") and line != "[net]":
            break
    return w, h


class Detector:
    def __init__(self, net, size, n_cls):
        self.net, self.size, self.n_cls = net, size, n_cls
        self.out_names = net.getUnconnectedOutLayersNames()

    def __call__(self, img, conf, nms):
        H, W = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, self.size, swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.out_names)
        boxes, scores, ids = [], [], []
        for out in outs:
            out = out.reshape(-1, out.shape[-1])
            cls_scores = out[:, 5:]
            cid = cls_scores.argmax(1)
            sc = cls_scores[np.arange(len(out)), cid] * out[:, 4]
            keep = sc >= conf
            for (cx, cy, bw, bh), s, c in zip(out[keep, :4], sc[keep], cid[keep]):
                boxes.append([int((cx - bw / 2) * W), int((cy - bh / 2) * H), int(bw * W), int(bh * H)])
                scores.append(float(s))
                ids.append(int(c))
        res = []
        if boxes:
            idx = cv2.dnn.NMSBoxes(boxes, scores, conf, nms)
            for i in np.array(idx).flatten():
                res.append((boxes[i], scores[i], ids[i]))
        return res


class UltraDetector:
    """.pt / .onnx через ultralytics (YOLOv5u, v8, v9, v10, 11, 12...)."""

    def __init__(self, path, size, device):
        try:
            import torch
            from ultralytics import YOLO
        except ImportError:
            sys.exit("[!] Для .pt/.onnx нужен пакет ultralytics:  pip install ultralytics")
        self.hub = None
        try:
            self.model = YOLO(str(path))
            names = self.model.names
        except Exception as e:
            # старый формат репозитория yolov5 — пробуем через torch.hub
            print(f"[i] ultralytics не открыл модель ({str(e)[:120]}) — пробую как YOLOv5 через torch.hub...")
            self.hub = torch.hub.load("ultralytics/yolov5", "custom", path=str(path), trust_repo=True)
            names = self.hub.names
        self.names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
        self.size = size
        self.device = device or (0 if torch.cuda.is_available() else "cpu")
        self.backend = (f"CUDA ({torch.cuda.get_device_name(0)})" if self.device != "cpu" and torch.cuda.is_available()
                        else "CPU") + (" · torch.hub" if self.hub else " · ultralytics")

    def __call__(self, img, conf, nms):
        res = []
        if self.hub is not None:
            self.hub.conf, self.hub.iou = conf, nms
            out = self.hub(img[:, :, ::-1], size=self.size or 640).xyxy[0].cpu().numpy()
            rows = [(r[:4], r[4], r[5]) for r in out]
        else:
            kw = {"imgsz": self.size} if self.size else {}
            r = self.model.predict(img, conf=conf, iou=nms, device=self.device, verbose=False, **kw)[0]
            b = r.boxes
            rows = zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()) if b is not None else []
        for (x1, y1, x2, y2), s, c in rows:
            res.append(([int(x1), int(y1), int(x2 - x1), int(y2 - y1)], float(s), int(c)))
        return res


def colors(n):
    rs = np.random.RandomState(7)
    return [tuple(int(x) for x in rs.randint(60, 255, 3)) for _ in range(max(n, 1))]


def draw(img, dets, names, cols):
    t = max(1, round(min(img.shape[:2]) / 360))
    for (x, y, w, h), s, c in dets:
        col = cols[c % len(cols)]
        label = f"{names[c] if c < len(names) else c} {s:.2f}"
        cv2.rectangle(img, (x, y), (x + w, y + h), col, t + 1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t, t)
        ty = y - 5 if y - th - 8 > 0 else y + h + th + 5
        cv2.rectangle(img, (x, ty - th - 4), (x + tw + 6, ty + 4), col, -1)
        cv2.putText(img, label, (x + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t, (0, 0, 0), t, cv2.LINE_AA)


def hud(img, lines):
    y = 22
    for l in lines:
        cv2.putText(img, l, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, l, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24


def open_source(a):
    """-> (тип, объект): camera/video -> VideoCapture, images -> список путей"""
    if a.source is None:
        api = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(a.cam, api)
        if not cap.isOpened():
            cap = cv2.VideoCapture(a.cam)
        if not cap.isOpened():
            sys.exit(f"[!] Камера {a.cam} не открылась. Попробуйте --cam 1 или закройте другие программы с камерой")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
        return "camera", cap
    p = Path(a.source)
    if p.is_dir():
        return "images", sorted(x for x in p.iterdir() if x.suffix.lower() in IMG_EXT)
    if p.suffix.lower() in IMG_EXT:
        return "images", [p]
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        sys.exit(f"[!] Не открылось: {p}")
    return "video", cap


def main():
    ap = argparse.ArgumentParser(description="Тест YOLOv4-tiny на веб-камере/видео",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--weights", "--model", dest="weights", help=".weights, .pt или .onnx")
    ap.add_argument("--cfg")
    ap.add_argument("--names")
    ap.add_argument("--model-dir", help="папка, где лежат weights/cfg/names")
    ap.add_argument("--cam", type=int, default=0, help="номер камеры")
    ap.add_argument("--source", help="видео, картинка или папка вместо камеры")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--conf", type=float, default=0.4, help="порог уверенности")
    ap.add_argument("--nms", type=float, default=0.45)
    ap.add_argument("--size", type=int, default=0, help="размер входа сети (0 — из cfg / из модели)")
    ap.add_argument("--gpu", action="store_true", help="Darknet: CUDA, если OpenCV собран с ней")
    ap.add_argument("--device", help=".pt: cpu, 0, 1... (по умолчанию видеокарта, если есть)")
    ap.add_argument("--headless", action="store_true", help="без окна (для проверки), пишет out.avi")
    ap.add_argument("--frames", type=int, default=0, help="остановиться после N кадров")
    a = ap.parse_args()

    cfg, weights, names_f = find_model(a)
    names = [l.strip() for l in names_f.read_text(encoding="utf-8").splitlines() if l.strip()] if names_f else []
    if cfg is None:  # .pt / .onnx
        print(f"[i] модель:  {weights}")
        det = UltraDetector(weights, a.size, a.device)
        if not names or len(names) != len(det.names):
            names = det.names
        backend = det.backend
        print(f"[i] классы:  {len(names)}: {', '.join(names[:12])}{' ...' if len(names) > 12 else ''}")
    else:
        size = (a.size, a.size) if a.size else cfg_size(cfg)
        print(f"[i] cfg:     {cfg}\n[i] weights: {weights}\n[i] names:   {names_f} ({len(names)} классов)\n"
              f"[i] вход сети {size[0]}x{size[1]}")
        net, backend = load_net(cfg, weights, a.gpu)
        det = Detector(net, size, len(names))
    cols = colors(max(len(names), 1))
    print(f"[i] вычисления: {backend}")

    kind, src = open_source(a)
    win = "YOLO test  (Q - exit, Space - pause, S - snapshot, R - record, +/- threshold)"
    if not a.headless:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    conf, paused, writer, rec_path = a.conf, False, None, None
    shots = HERE / "shots"
    fps, t_last, n, img_i = 0.0, time.time(), 0, 0
    frame, step = None, False
    if a.headless:
        rec_path = HERE / "out.avi"

    while True:
        if not paused or frame is None or step:
            if kind == "images":
                if img_i >= len(src):
                    if a.headless:
                        break
                    img_i = len(src) - 1
                    paused = True
                frame = cv2.imdecode(np.fromfile(str(src[img_i]), np.uint8), cv2.IMREAD_COLOR)
                img_i += 1
                if frame is None:
                    continue
            else:
                ok, frame = src.read()
                if not ok:
                    break
        t0 = time.time()
        dets = det(frame, conf, a.nms)
        dt = time.time() - t0
        vis = frame.copy()
        draw(vis, dets, names, cols)
        n += 1
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1 / max(now - t_last, 1e-6)) if n > 1 else 0
        t_last = now
        cnt = {}
        for _, _, c in dets:
            nm = names[c] if c < len(names) else str(c)
            cnt[nm] = cnt.get(nm, 0) + 1
        hud(vis, [f"{fps:4.1f} FPS | сеть {dt * 1000:.0f} мс | {backend} | порог {conf:.2f}"
                  .replace("сеть", "net").replace("порог", "thr"),
                  "  ".join(f"{k}:{v}" for k, v in cnt.items()) or "-"]
            + (["REC"] if writer else []) + (["PAUSE"] if paused else []))
        if rec_path and writer is None:
            writer = cv2.VideoWriter(str(rec_path), cv2.VideoWriter_fourcc(*"MJPG"), 20,
                                     (vis.shape[1], vis.shape[0]))
        if writer:
            writer.write(vis)
        if a.headless:
            if n % 10 == 0:
                print(f"[{n}] {dt * 1000:.0f} мс, найдено {len(dets)}: {cnt}")
            if a.frames and n >= a.frames:
                break
            continue
        cv2.imshow(win, vis)
        k = cv2.waitKeyEx(0 if (paused and kind != "camera") else 1)
        if k in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if k == 32:
            paused = not paused
        elif k in (ord("s"), ord("S")):
            shots.mkdir(exist_ok=True)
            p = shots / f"shot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imencode(".jpg", vis)[1].tofile(str(p))
            print(f"[i] снимок: {p}")
        elif k in (ord("r"), ord("R")):
            if writer:
                writer.release()
                writer, rec_path = None, None
                print("[i] запись остановлена")
            else:
                shots.mkdir(exist_ok=True)
                rec_path = shots / f"rec_{time.strftime('%Y%m%d_%H%M%S')}.avi"
                print(f"[i] запись: {rec_path}")
        elif k in (ord("+"), ord("="), 43):
            conf = min(0.95, conf + 0.05)
        elif k in (ord("-"), ord("_")):
            conf = max(0.05, conf - 0.05)
        elif k in (ord("f"), ord("F")):
            full = cv2.getWindowProperty(win, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_NORMAL if full else cv2.WINDOW_FULLSCREEN)
        elif paused and kind != "camera" and k in (2555904, 65363, 2424832, 65361):   # → / ←
            if k in (2424832, 65361):
                if kind == "images":
                    img_i = max(0, img_i - 2)
                else:
                    src.set(cv2.CAP_PROP_POS_FRAMES, max(0, src.get(cv2.CAP_PROP_POS_FRAMES) - 2))
            step = True
            continue
        step = False
    if writer:
        writer.release()
        print(f"[i] видео сохранено: {rec_path}")
    if kind != "images":
        src.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
