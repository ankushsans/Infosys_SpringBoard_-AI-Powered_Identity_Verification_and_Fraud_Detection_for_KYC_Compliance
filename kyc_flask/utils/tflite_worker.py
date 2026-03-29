"""
utils/tflite_worker.py
─────────────────────
Runs as a subprocess. Reads a JSON request from stdin, runs TFLite
inference, writes a JSON response to stdout. PyTorch is never imported
in this process, so the pthreadpool double-free cannot happen.
"""

import json
import sys
import numpy as np
from PIL import Image
import io
import base64


def preprocess(image_bytes: bytes, h: int, w: int) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((w, h), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    arr = arr / 127.5 - 1.0
    return np.expand_dims(arr, axis=0)


def load_interpreter(model_path: str):
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter(model_path=model_path)
    except ImportError:
        pass
    try:
        import tflite_runtime.interpreter as tflite
        return tflite.Interpreter(model_path=model_path)
    except ImportError:
        pass
    import tensorflow as tf
    return tf.lite.Interpreter(model_path=model_path)


def main():
    try:
        req = json.loads(sys.stdin.buffer.read())
        model_path   = req["model_path"]
        image_bytes  = base64.b64decode(req["image_b64"])

        interp = load_interpreter(model_path)
        interp.allocate_tensors()

        in_det  = interp.get_input_details()[0]
        out_det = interp.get_output_details()[0]

        h, w = int(in_det["shape"][1]), int(in_det["shape"][2])
        arr  = preprocess(image_bytes, h, w).astype(in_det["dtype"])

        interp.set_tensor(in_det["index"], arr)
        interp.invoke()
        probs = interp.get_tensor(out_det["index"])[0].tolist()

        print(json.dumps({"probs": probs}))

    except Exception as exc:
        print(json.dumps({"error": str(exc)}))


if __name__ == "__main__":
    main()
