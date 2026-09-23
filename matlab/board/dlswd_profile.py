#!/usr/bin/env python3
"""
dlswd_profile.py  --  run ON the KR260, not on the host.

Profiles all four pipeline stages separately so the paper can report a real
end-to-end latency instead of CNN-inference-only throughput:

    preprocess : |x| -> min-max normalise -> resize to 75x75
    transfer   : host <-> DPU buffer copies
    cnn        : DPU execution of the quantised classifier
    denoise    : the selected wavelet configuration applied to the frame

Writes hw_measured.json in the format expected by s07_hardware_timing.m.

    python3 dlswd_profile.py --platform KR260 --xmodel dlswd_cnn.xmodel \
            --frames frames.npy --labels labels.npy --out hw_measured.json
"""
import argparse, json, time, os
import numpy as np
import pywt
import vart, xir


def load_dpu(xmodel):
    g = xir.Graph.deserialize(xmodel)
    subs = [s for s in g.get_root_subgraph().toposort_child_subgraph()
            if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]
    assert len(subs) == 1, "expected exactly one DPU subgraph"
    return vart.Runner.create_runner(subs[0], "run"), subs[0]


def dpu_meta(sub):
    a = sub.get_attrs()
    return (a.get("dpu_name", "unknown"), int(a.get("dpu_fingerprint", 0)))


CFGS = {  # must mirror the MCS retained by cbcSelect
    0: ("fk4", 6, "universal", "soft"),
    1: ("db8", 5, "bayes",     "soft"),
    2: ("db4", 7, "sure",      "soft"),
}


def threshold(c, rule, lam):
    if rule == "soft":
        return np.sign(c) * np.maximum(np.abs(c) - lam, 0.0)
    return c * (np.abs(c) > lam)


def denoise(frame, cfg):
    wav, lev, meth, rule = cfg
    out = np.empty_like(frame)
    for col in range(frame.shape[1]):
        for part, sl in ((frame[:, col].real, "re"), (frame[:, col].imag, "im")):
            co = pywt.wavedec(part, wav, level=lev, mode="symmetric")
            sigma = np.median(np.abs(co[-1])) / 0.6745
            n = part.size
            if meth == "universal":
                lam = sigma * np.sqrt(2 * np.log(n))
            elif meth == "minimax":
                lam = sigma * (0.3936 + 0.1829 * np.log2(n))
            elif meth == "bayes":
                lam = sigma ** 2 / max(np.sqrt(max(np.var(co[-1]) - sigma ** 2, 1e-12)), 1e-12)
            else:  # sure
                lam = sigma * np.sqrt(2 * np.log(n * np.log2(max(n, 2))))
            co[1:] = [threshold(c, rule, lam) for c in co[1:]]
            r = pywt.waverec(co, wav, mode="symmetric")[:n]
            if sl == "re":
                re = r
            else:
                out[:, col] = re + 1j * r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True, choices=["KR260"])
    ap.add_argument("--xmodel", required=True)
    ap.add_argument("--frames", required=True, help=".npy complex [N,slow,fast]")
    ap.add_argument("--labels", required=True, help=".npy int64 [N] ground truth")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--power-w", type=float, default=None,
                    help="measured rail power in W; omit to record 'not measured'")
    ap.add_argument("--out", default="hw_measured.json")
    a = ap.parse_args()

    frames = np.load(a.frames)
    labels = np.load(a.labels)
    runner, sub = load_dpu(a.xmodel)
    it, ot = runner.get_input_tensors(), runner.get_output_tensors()
    ishape = tuple(it[0].dims)
    oshape = tuple(ot[0].dims)
    arch, fp = dpu_meta(sub)

    t_pre = t_tx = t_cnn = t_den = 0.0
    correct = 0
    N = len(frames)

    for i in range(N):
        f = frames[i]

        t0 = time.perf_counter()
        img = np.abs(f).astype(np.float32)
        img = (img - img.min()) / max(img.ptp(), 1e-12)
        yi = np.linspace(0, img.shape[0] - 1, 75).astype(np.int32)
        xi = np.linspace(0, img.shape[1] - 1, 75).astype(np.int32)
        img = img[np.ix_(yi, xi)]
        t_pre += time.perf_counter() - t0

        t0 = time.perf_counter()
        ib = [np.empty(ishape, dtype=np.float32, order="C")]
        ob = [np.empty(oshape, dtype=np.float32, order="C")]
        ib[0][0, ..., 0] = img
        t_tx += time.perf_counter() - t0

        t0 = time.perf_counter()
        jid = runner.execute_async(ib, ob)
        runner.wait(jid)
        t_cnn += time.perf_counter() - t0

        t0 = time.perf_counter()
        pred = int(np.argmax(ob[0][0]))
        t_tx += time.perf_counter() - t0

        t0 = time.perf_counter()
        denoise(f, CFGS[pred])
        t_den += time.perf_counter() - t0

        correct += int(pred == int(labels[i]))

    res = {
        a.platform: {
            "n_frames": N,
            "preprocess_ms": 1e3 * t_pre / N,
            "transfer_ms": 1e3 * t_tx / N,
            "cnn_ms": 1e3 * t_cnn / N,
            "denoise_ms": 1e3 * t_den / N,
            "accuracy_pct": 100.0 * correct / N,
            "dpu_arch": arch,
            "dpu_fingerprint": hex(fp),
            "dpu_clock_mhz": int(os.popen(
                "xbutil examine 2>/dev/null | awk '/DPU/&&/MHz/{print $NF}'"
            ).read().strip() or 0),
            "threads": a.threads,
            "power_w": a.power_w if a.power_w is not None else "not measured",
            "power_method": "measured rail (INA226)" if a.power_w is not None
                            else "NOT MEASURED - do not quote as measured",
        }
    }

    merged = {}
    if os.path.exists(a.out):
        merged = json.load(open(a.out))
    merged.update(res)
    json.dump(merged, open(a.out, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
