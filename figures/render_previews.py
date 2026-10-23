"""Render every figures/*.pdf to PNG for visual checks. Usage: python -I figures/render_previews.py OUT_DIR   (needs pypdfium2; previews are not deliverables)"""
import glob
import os
import sys

import pypdfium2 as pdfium

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
for f in sorted(glob.glob("figures/*.pdf")):
    page = pdfium.PdfDocument(f)[0]
    w, h = page.get_size()
    page.render(scale=3).to_pil().save(os.path.join(out, os.path.basename(f)[:-4] + ".png"))
    print(os.path.basename(f), "%.2f x %.2f cm" % (w / 72 * 2.54, h / 72 * 2.54))
