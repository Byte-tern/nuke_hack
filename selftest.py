"""Самопроверка логики склейки на синтетических данных (без видео и GPU): python selftest.py"""
import numpy as np
from fusion import fuse, P

rng = np.random.default_rng(0)


def build(rows, N, sig=None):
    rows.sort(key=lambda r: r[0])
    arr = np.array(rows, dtype=object)
    d = {"fps": 25.0, "n_frames": N, "start": 0, "cameras": [1, 2, 3, 4, 5], "video_size": {},
         "f": arr[:, 0].astype(np.int32), "cam": arr[:, 1].astype(np.int8), "tid": arr[:, 2].astype(np.int32),
         "box": arr[:, 3:7].astype(np.float32), "plan": arr[:, 8:10].astype(np.float32),
         "foot_ok": arr[:, 10].astype(bool)}
    if sig is not None:
        d["sig"] = np.full(len(rows), sig, np.float32)
    return d


def add(rows, f, cam, tid, pos, noise=1.5):
    p = np.array(pos) + rng.normal(scale=noise, size=2)
    rows.append((f, cam, tid, .1, .1, .2, .4, .9, p[0], p[1], True))


# ---- тест 1: основные правила
rows, N = [], 700
for f in range(0, 400):                      # A: cam1 -> cam2, смена локального id в cam2 на кадре 300
    pos = (50, 50 + 0.7 * f)
    if f <= 220: add(rows, f, 1, 10, pos)
    if 150 <= f < 300: add(rows, f, 2, 5, pos)
    if 300 <= f: add(rows, f, 2, 6, pos)
for f in range(100, 250): add(rows, f, 1, 11, (120, 100 + 0.2 * (f - 100)))      # B
for f in range(350, 450): add(rows, f, 1, 12, (120, 130))                         # B' вернулся спустя 100 кадров -> НОВЫЙ id
for f in range(500, 503): add(rows, f, 3, 99, (30, 400))                         # ложный 3-кадровый трек
for f in range(600, 620): add(rows, f, 3, 77, (30, 380))                         # 20 кадров (<1 с) - отбрасывается
for f in range(500, N):                                                           # C: две камеры до конца видео
    add(rows, f, 4, 20, (150, 300 + 0.1 * (f - 500))); add(rows, f, 5, 30, (151, 301 + 0.1 * (f - 500)))
res = fuse(build(rows, N), None, P)
tr = res["tracks"]
assert len(tr) == 4, tr
assert tr[1]["start"] == 0 and tr[1]["last"] == 399 and tr[1]["finished"]
assert tr[2]["last"] == 249 and tr[2]["finished"]
assert tr[3]["start"] == 350 and tr[3]["finished"], "вернувшийся после >30 кадров получает новый ID"
assert tr[4]["start"] == 500 and not tr[4]["finished"], "активный в конце видео - не finished"
assert (res["gid"][res["cam"] == 3]).size == 0, "ложные/короткие (<1 с) треки отфильтрованы"
assert set(res["gid"][(res["cam"] == 4) | (res["cam"] == 5)]) == {4}
print("тест 1 OK")

# ---- тест 2: один человек получил два ID (далёкая зона, проекции расходятся на ~22 px) -> должен стать одним ID
class P2(P):
    k_gate = 0.0                                   # без ворот по неопределённости и без онлайн-слияния -> работает только итоговый проход
    merge_frames = 10 ** 6
rows, N = [], 300
for f in range(0, 300):
    add(rows, f, 5, 6, (170, 330), noise=1.0)      # камера 5
for f in range(40, 300):
    add(rows, f, 3, 7, (170, 352), noise=1.0)      # камера 3: тот же человек, смещение 22 px
res = fuse(build(rows, N, sig=14.0), None, P2)
assert len(res["tracks"]) == 1, res["tracks"]
print("тест 2 OK (дубль слит итоговым проходом)")

# ---- тест 3: двое рядом на ОДНОЙ камере - разные люди (не сливать)
rows = []
for f in range(0, 200):
    add(rows, f, 4, 1, (100, 200), noise=0.5); add(rows, f, 4, 2, (108, 204), noise=0.5)
res = fuse(build(rows, 200, sig=14.0), None, P)
assert len(res["tracks"]) == 2, res["tracks"]
print("тест 3 OK (двое на одной камере остаются двумя ID)")
print("OK")

# ---- тест 4: синхронизация камер с разным fps (шкала времени)
from common import build_timeline
info = {1: (25.0, 30001), 2: (25.0, 30001), 3: (15.0, 18001), 4: (30.0, 36001), 5: (25.0, 30001)}   # ~20 мин
N, idx = build_timeline(info, 25.0)
t = (np.arange(N) + 0.5) / 25.0
for c, (fps_c, n_c) in info.items():
    err = np.abs((idx[c] + 0.5) / fps_c - t)              # рассинхрон момента кадра и шкалы
    assert err.max() <= 0.5 / fps_c + 1e-6, (c, err.max())  # не больше полукадра исходного видео и НЕ растёт со временем
    assert idx[c][-1] <= n_c - 1 and np.all(np.diff(idx[c]) >= 0)
assert np.mean(np.diff(idx[3]) == 0) > 0.3 and np.mean(np.diff(idx[4]) > 1) > 0.1      # 15 fps - дубли, 30 fps - пропуски
print(f"тест 4a OK: кадров шкалы {N}, макс. рассинхрон за ~20 мин = {max(np.abs((idx[c] + 0.5) / info[c][0] - t).max() for c in info) * 1000:.0f} мс (без накопления)")

import cv2, os, tempfile
from stage1_detect_track import Reader
tmp = tempfile.mkdtemp()
def mk(path, fps, n):
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (160, 120))
    for i in range(n):
        w.write(np.full((120, 160, 3), 5 + i * 4, np.uint8))     # яркость кодирует номер кадра
    w.release()
mk(os.path.join(tmp, "a.avi"), 25, 50); mk(os.path.join(tmp, "b.avi"), 10, 20)       # обе по 2 секунды
N, idx = build_timeline({1: (25.0, 50), 2: (10.0, 20)}, 25.0)
assert N == 50
got = {}
for cam, name in ((1, "a.avi"), (2, "b.avi")):
    rd = Reader(os.path.join(tmp, name), idx[cam]); out = []
    while True:
        it = rd.q.get()
        if it is None: break
        out.append((int(round((it[0].mean() - 5) / 4)), it[2], it[3]))
    got[cam] = out
assert len(got[1]) == len(got[2]) == 50
assert all(abs(v - s) <= 1 for v, s, _ in got[1] + got[2]), "кадр прочитан не тот, что запланирован"
assert sum(n for _, _, n in got[2]) == 20 and sum(n for _, _, n in got[1]) == 50      # дубли не декодируются повторно
print("тест 4b OK: Reader отдаёт кадры 10 fps и 25 fps синхронно (дубли без повторного чтения)")
print("OK")
