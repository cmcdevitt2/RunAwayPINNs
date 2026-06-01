from pathlib import Path

import numpy as np
import onnxruntime as ort

MODELS = [
    Path("models/rpf_forward.onnx"),
    Path("models/rpf_residual.onnx"),
]

for path in MODELS:
    print(f"checking {path.resolve()}")
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    outs = sess.get_outputs()
    print("input:", inp.name, inp.shape, inp.type)
    print("outputs:", [(o.name, o.shape, o.type) for o in outs])
    x = np.zeros((128, 5), dtype=np.float32)
    x[:, 0] = np.linspace(0.0, 1.0, 128, dtype=np.float32)
    x[:, 1] = np.linspace(0.0, 1.0, 128, dtype=np.float32)
    x[:, 2] = 0.5
    x[:, 3] = 0.0
    x[:, 4] = 0.0
    y = sess.run(None, {inp.name: x})
    for i, yy in enumerate(y):
        print(f"output[{i}] shape={yy.shape} dtype={yy.dtype} finite={np.isfinite(yy).all()}")
    print("ok\n")
