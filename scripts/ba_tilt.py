import numpy as np, sys, os, itertools
from scipy.spatial.transform import Rotation as Rt
def readers(a):
    yield 'rotvec', lambda v: Rt.from_rotvec(v)
    for order in ('xyz', 'zyx', 'ZYX', 'XYZ', 'zxy', 'ZXY'):
        for deg in (False, True):
            yield 'euler-%s-%s' % (order, 'deg' if deg else 'rad'), (lambda o, d: (lambda v: Rt.from_euler(o, v, degrees=d)))(order, deg)
def best_reading(ci):
    best = None
    for (name, f), inv, ax in itertools.product(list(readers(ci)), (False, True), (1, -1)):
        try:
            R = f(ci[:, 3:6]); R = R.inv() if inv else R
            d = R.apply([0, 0, ax])
        except Exception:
            continue
        score = np.mean(d[:, 2] < -0.97)            # 연직 아래(ENU -z)를 향한 비율
        if best is None or score > best[0]:
            best = (score, name, inv, ax, f)
    return best
def shift(npz, h):
    z = np.load(npz, allow_pickle=True); c, ci = z['cams_opt'], z['initial']
    score, name, inv, ax, f = best_reading(ci)
    if score < 0.8:
        return None, (score, name, inv, ax)
    Ri, Ro = f(ci[:, 3:6]), f(c[:, 3:6])
    if inv: Ri, Ro = Ri.inv(), Ro.inv()
    di, do = Ri.apply([0, 0, ax]), Ro.apply([0, 0, ax])
    gi = di[:, :2] / -di[:, 2:3] * h; go = do[:, :2] / -do[:, 2:3] * h
    d = go - gi
    tilt = np.degrees(np.arccos(np.clip(np.sum(di * do, axis=1), -1, 1)))
    return (d.mean(axis=0), np.median(np.hypot(d[:, 0], d[:, 1])), np.median(tilt)), (score, name, inv, ax)
h = float(os.environ.get('H', 45))
for p in sys.argv[1:]:
    run = os.path.basename(os.path.dirname(p)); site = os.path.basename(os.path.dirname(os.path.dirname(p)))
    if not os.path.exists(p):
        print('%-20s %-14s  cameras.npz 없음' % (site, run)); continue
    r, (score, name, inv, ax) = shift(p, h)
    how = '%s%s, 광축 %sz, 아래 향함 %.0f%%' % (name, ' (역)' if inv else '', '+' if ax > 0 else '-', 100 * score)
    if r is None:
        print('%-20s %-14s  판별 실패 — 가장 나은 해석: %s' % (site, run, how)); continue
    (e, n), dist, tl = r
    print('%-20s %-14s  옮겨짐 평균 동 %+.2f · 북 %+.2f m  (프레임별 중앙 %.2f m, 기울기 중앙 %.2f°)   [%s]' % (site, run, e, n, dist, tl, how))
