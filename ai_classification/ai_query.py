
import base64
from openai import OpenAI
from pathlib import Path

client = OpenAI()  # requires OPENAI_API_KEY env var

# Get path to PROMPT.txt relative to this file's location
SCRIPT_DIR = Path(__file__).parent
PROMPT_PATH = SCRIPT_DIR / 'PROMPT.txt'

with open(PROMPT_PATH, 'r') as f:
    PROMPT = f.read()


#TODO: pass live DB into prompt
CALL_TYPES = r"""
N01i: (BB) (SHORT ASC FAST | FLAT) (PEAK FLAT PEAK | PEAK FLAT)
N01ii: (BB SHORT) (SHORT ASC FAST | GAP) (PEAK FLAT)
N01iii: (BB) (SHORT FLAT ?) (PEAK FLAT)
N02: (BB SBI_TIGHT) (SQUIGGLE) (SHORT ASC FAST)
N03: (BB SHORT) (GAP SHORT) (DESC SBI_DEC)
N04: (LEFT PEAK LARGE , FLAT) (SHORT ASC SBI_TIGHT)
N05i: (UPSWEEP FLAT | UPSWEEP FLAT , ASC) (SHORT SBI_TIGHT , ASC)
N05ii: (FLAT | ASC SLOW) (BB SHORT) (SHORT PEAK) (BB SHORT)
N07i: (BB) (UPSWEEP , FLAT)
N07ii: (BB) (UPSWEEP FLAT) (ASC SBI_INC)
N07iii: (BB) (UPSWEEP FLAT) (ASC SBI_INC)
N07iv: (BB) (UPSWEEP FLAT) (ASC SBI_WIDE)
N08i: (PULSED) (LEFT PEAK)
N08ii: (PULSED) (SHORT FLAT SBI_TIGHT)
N08iii: (PULSED) (PEAK , FLAT | DESC)
N08iv: (PULSED) (SHORT SMALL PEAK) (BB)
N09i: (BB) (SHORT ASC FAST) (FLAT | ASC SLOW) (SHORT ASC FAST)
N10: (BB) (SHORT ?) (LARGE PEAK , FLAT) (BB SHORT ?)
N12: (BB SHORT) (FLAT SBI_TIGHT) (ASC SBI_INC | ASC , FLAT)
N13: (BB) (SHORT ASC) (SQUIGGLE | PEAK PEAK) (BB SHORT)
N16i: (ASC SLOW , ASC FAST) (BB SHORT) (ASC SBI_WIDE | LARGE RIGHT PEAK SBI_WIDE) (BB SHORT)
N16ii: (ASC | FLAT , ASC) (SHORT SBI_TIGHT) (ASC SBI_WIDE | LARGE RIGHT PEAK SBI_WIDE) (SHORT SBI_TIGHT)
N16iii: (PEAK , FLAT , ASC) () (ASC SBI_WIDE) (SHORT SBI_TIGHT)
N16iv: (FLAT | ASC SLOW) (ASC SBI_WIDE) (BB SBI_TIGHT)
N18: (FLAT) (BB SHORT) (SHORT PEAK) (BB SHORT)
N20: (PEAK LARGE)
N21: (BB SHORT) (PEAK) (BB | SINGLE_TONE)
N23ii: (FLAT SBI_WIDE) (DOWNSWEEP FLAT DOWNSWEEP)
N25: (SBI_TIGHT ASC) (DESC SBI_WIDE BIPHO) (SHORT DIP) (SBI_TIGHT FLAT | PEAK)
N32i: (ASC SBI_INC , BIPHO) (SHORT PEAK)
"""


PROMPT = PROMPT.replace("{{vocab}}", CALL_TYPES.strip())

def encode_image_to_data_url(image_path: str) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def annotate_spectrogram(image_path: str, display_image: bool = True) -> str:
    img_url = encode_image_to_data_url(image_path)
    
    if display_image:
        from PIL import Image
        import IPython.display as display
        img = Image.open(image_path)
        display.display(img)

    resp = client.responses.create(
        model="gpt-5.2",
        input=[
            {
                "role": "system",
                "content": [
                    {"type": "input_text", "text": PROMPT }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Annotate this spectrogram."},
                    {"type": "input_image", "image_url": img_url},
                ],
            },
        ],
        # optional: keep it short + deterministic
        temperature=0,
        max_output_tokens=400,
    )

    # Parse resp.output_text to extract tokens and prediction separately
    # Example output: '(FLAT SBI_TIGHT) (ASC SBI_INC)\n\nPREDICT: N12'
    text = resp.output_text.strip()
    import re

    # Find prediction ("PREDICT: ...")
    prediction_match = re.search(r'PREDICT:\s*([^\n\r]+)', text)
    prediction = prediction_match.group(1).strip() if prediction_match else ""

    # Get annotation, which is everything before 'PREDICT:'
    annotation = text.split("PREDICT:")[0].strip() if "PREDICT:" in text else text

    return annotation, prediction
