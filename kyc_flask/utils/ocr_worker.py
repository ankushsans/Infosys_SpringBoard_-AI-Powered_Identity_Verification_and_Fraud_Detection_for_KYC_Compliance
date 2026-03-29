"""
utils/ocr_worker.py
────────────────────
Runs as a subprocess. Receives a JSON request on stdin, runs OCR,
returns JSON on stdout. Isolated from the GNN's PyTorch session.
"""


import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import json

def run_aadhaar(image_path: str) -> dict:
    import openbharatocr
    result = openbharatocr.front_aadhaar(image_path)
    return {
        "Full Name":          result.get("Full Name", ""),
        "Date/Year of Birth": result.get("Date/Year of Birth", ""),
        "Gender":             result.get("Gender", ""),
        "Aadhaar Number":     result.get("Aadhaar Number", ""),
    }


def run_pan(image_path: str) -> dict:
    import openbharatocr
    result = openbharatocr.pan(image_path)
    return {
        "Name":          result.get("Full Name", ""),
        "Parent's Name": result.get("Parent's Name", ""),
        "Date of Birth": result.get("Date of Birth", ""),
        "PAN Number":    result.get("PAN Number", ""),
    }


def run_passport(image_path: str) -> dict:
    import re
    from PIL import Image

    MONTH_MAP = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5,  "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10,"NOV": 11, "DEC": 12,
    }

    result = {
        "given_name": "", "surname": "", "date_of_birth": "",
        "sex": "", "nationality": "", "place_of_birth": "", "place_of_issue": "",
    }

    try:
        import pytesseract
        img = Image.open(image_path).convert("RGB")
        raw = pytesseract.image_to_string(img, config="--psm 6")
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        mrz = [l for l in lines if re.fullmatch(r"[A-Z0-9<]{44}", l)]

        if len(mrz) >= 2:
            line1, line2 = mrz[0], mrz[1]
            m = re.match(r"P[A-Z<]([A-Z]{3})<<([A-Z<]+)", line1)
            if m:
                result["nationality"] = m.group(1).replace("<", "")
                parts = m.group(2).split("<<", 1)
                result["surname"]    = parts[0].replace("<", " ").strip()
                result["given_name"] = parts[1].replace("<", " ").strip() if len(parts) > 1 else ""
            if len(line2) >= 21:
                dob_str = line2[13:19]
                sex_chr = line2[20]
                try:
                    yy, mm, dd = int(dob_str[:2]), int(dob_str[2:4]), int(dob_str[4:])
                    year = 1900 + yy if yy >= 30 else 2000 + yy
                    mon_name = list(MONTH_MAP.keys())[mm - 1]
                    result["date_of_birth"] = f"{dd:02d} {mon_name} {year}"
                except Exception:
                    pass
                result["sex"] = sex_chr if sex_chr in ("M", "F") else ""
    except Exception as exc:
        result["_error"] = str(exc)

    return result


def main():
    try:
        req        = json.loads(sys.stdin.buffer.read())
        doc_type   = req["doc_type"]
        image_path = req["image_path"]

        if doc_type == "aadhaar":
            fields = run_aadhaar(image_path)
        elif doc_type == "pan":
            fields = run_pan(image_path)
        elif doc_type == "passport":
            fields = run_passport(image_path)
        else:
            fields = {"_error": f"Unknown doc_type: {doc_type}"}

        print(json.dumps({"fields": fields}))

    except Exception as exc:
        print(json.dumps({"error": str(exc)}))


if __name__ == "__main__":
    main()
