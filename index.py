import sys
import os
import time
import threading
import argparse

import cv2
import torch
import numpy as np
import pygame
import face_recognition

from pathlib import Path
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from utils.torch_utils import select_device

def parse_args():
    parser = argparse.ArgumentParser(
        description="Intrusion detection with YOLOv5 + face recognition")
    parser.add_argument("--video", default="Input.mp4")
    parser.add_argument("--output", default="Output.mp4")
    parser.add_argument("--weights", default="yolov5n.pt", help="YOLOv5 weights file")
    parser.add_argument("--known-dir", default="Faces", help="Directory of authorized face images")
    parser.add_argument("--siren", default="siren.wav", help="Path to siren sound file")
    parser.add_argument("--logo", default="logo.jpg", help="Overlay logo image (PNG with alpha)")
    parser.add_argument("--shape", type=int, default=320,
                        help="Shorter side size for YOLO downscaling")
    parser.add_argument("--dist", type=float, default=30.0,
                        help="Pixel threshold for tripwire breach")
    parser.add_argument("--tol", type=float, default=0.6,
                        help="Face matching tolerance")
    parser.add_argument("--cooldown", type=float, default=5.0,
                        help="Siren cooldown in seconds")
    return parser.parse_args()

def init_audio(siren_path: str):
    pygame.mixer.init()

    def play_siren():
        pygame.mixer.music.load(siren_path)
        pygame.mixer.music.play()

    return play_siren


def load_logo(logo_path: str):
    img = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None
    if img.shape[2] == 4:
        rgb = img[...,:3].astype(float)
        alpha = (img[..., 3:].astype(float) / 255.0)
    else:
        rgb = img.astype(float)
        alpha = np.ones(rgb.shape[:2] + (1,), float)
    return rgb, alpha


def load_known_faces(known_dir: str):
    encodings, names = [], []
    for fn in os.listdir(known_dir):
        if fn.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(known_dir, fn)
            img = face_recognition.load_image_file(path)
            results = face_recognition.face_encodings(img)
            if results:
                encodings.append(results[0])
                names.append(os.path.splitext(fn)[0])
    return encodings, names


def define_tripwire(cap, max_display: int=960):
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not grab first frame for tripwire definition")
    h, w = frame.shape[:2]
    scale = min(max_display / max(w, h), 1.0)
    disp = cv2.resize(frame, None, fx=scale, fy=scale)
    points = []

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            ox, oy = int(x / scale), int(y / scale)
            points.append((ox, oy))
            cv2.circle(disp, (x, y), 5, (0, 255, 0), -1)

    cv2.namedWindow("Define Tripwire")
    cv2.setMouseCallback("Define Tripwire", click)
    print("Click two points to set the tripwire. Press 'q' to abort.")
    while len(points) < 2:
        cv2.imshow("Define Tripwire", disp)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            cv2.destroyAllWindows()
            sys.exit(0)
    cv2.destroyAllWindows()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print(f"Tripwire: {points[0]} to {points[1]}")
    return points[0], points[1]


def point_line_distance(p, a, b):
    # p, a, b are (x,y)
    px, py = p
    x1, y1 = a
    x2, y2 = b
    return abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1) / np.hypot(y2 - y1, x2 - x1)


def overlay_logo(frame, logo_rgb, logo_alpha, margin=10):
    if logo_rgb is None or logo_alpha is None:
        return frame
    h, w = frame.shape[:2]
    lh, lw = logo_rgb.shape[:2]
    scale = min((w - 2 * margin) / lw, (h - 2 * margin) / lh, 1.0)
    oh, ow = int(lh * scale), int(lw * scale)

    # resized logo and alpha
    lr = cv2.resize(logo_rgb, (ow, oh), interpolation=cv2.INTER_AREA)
    la = cv2.resize(logo_alpha, (ow, oh), interpolation=cv2.INTER_AREA)
    if la.ndim == 2:
        la = la[:,:, np.newaxis]

    # blend into frame
    roi = frame[margin:margin + oh, margin:margin + ow].astype(float)
    frame[margin:margin + oh, margin:margin + ow] = (la * lr + (1 - la) * roi).astype(np.uint8)
    return frame


def main():
    args = parse_args()

    play_siren = init_audio(args.siren)
    logo_rgb, logo_alpha = load_logo(args.logo)
    known_encs, known_names = load_known_faces(args.known_dir)
    print(f"Loaded {len(known_encs)} known faces from {args.known_dir}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error opening video {args.video}")
        sys.exit(1)

    line_a, line_b = define_tripwire(cap)

    device = select_device('')
    model = DetectMultiBackend(args.weights, device=device)
    model.eval()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    last_siren = time.time() - args.cooldown
    print("Starting processing…")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        original = frame.copy()

        # YOLO detect
        h0, w0 = frame.shape[:2]
        scale = args.shape / min(h0, w0)
        small = cv2.resize(frame, None, fx=scale, fy=scale)
        img = letterbox(small, new_shape=(640, 640))[0]
        img = img[:,:,::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        tensor = torch.from_numpy(img).to(device).float() / 255.0
        det = non_max_suppression(model(tensor.unsqueeze(0)), 0.5, 0.4)[0]

        persons = []
        if det is not None and len(det):
            det[:,:4] = scale_boxes(tensor.shape[1:], det[:,:4], small.shape).round()
            for * xyxy, conf, cls in det:
                if int(cls) != 0:
                    continue
                x1, y1, x2, y2 = map(int, xyxy)
                x1, y1 = int(x1 / scale), int(y1 / scale)
                x2, y2 = int(x2 / scale), int(y2 / scale)
                persons.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(frame,
                "Detecting: Person",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2)


        # face recognition & intrusion logic
        for (x1, y1, x2, y2) in persons:
            crop = original[y1:y2, x1:x2]
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb, model='hog')
            encs = face_recognition.face_encodings(rgb, locs)
            for (top, right, bottom, left), enc in zip(locs, encs):
                cx = x1 + (left + right) // 2
                cy = y1 + (top + bottom) // 2
                d = point_line_distance((cx, cy), line_a, line_b)

                # match
                match_idx = -1
                if known_encs:
                    dists = face_recognition.face_distance(known_encs, enc)
                    match_idx = int(np.argmin(dists))
                if match_idx >= 0 and dists[match_idx] <= args.tol:
                    authorized = True
                    label = f"AUTHORIZED: {known_names[match_idx]}"
                    color = (0, 255, 0)
                else:
                    authorized = False
                    if d < args.dist and time.time() - last_siren > args.cooldown:
                        threading.Thread(target=play_siren, daemon=True).start()
                        last_siren = time.time()
                    label = "INTRUDER"
                    color = (0, 0, 255)

                fx1, fy1 = x1 + left, y1 + top
                fx2, fy2 = x1 + right, y1 + bottom
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), color, 2)
                cv2.putText(frame, label, (fx1, fy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # logo overlay
        frame = overlay_logo(frame, logo_rgb, logo_alpha)

        writer.write(frame)
        cv2.imshow("Output", frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
    print(f"Saved output to {args.output}")


if __name__ == "__main__":
    main()