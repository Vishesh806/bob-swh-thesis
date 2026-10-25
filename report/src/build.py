"""Build the thesis .docx. Usage: python build.py OUT.docx [pages.json]"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from builder import ThesisBuilder
from front import front_matter
from refs import REFS
import content_a, content_b, content_c, content_d

BLOCKS = content_a.BLOCKS + content_b.BLOCKS + content_c.BLOCKS + content_d.BLOCKS

if __name__ == "__main__":
    out = Path(sys.argv[1])
    pages = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}
    out.parent.mkdir(parents=True, exist_ok=True)
    b = ThesisBuilder(BLOCKS, REFS, Path(__file__).parent / "eq", pages)
    b.build(front_matter, out, watermark=Path(__file__).resolve().parents[2] / "assets" / "iitkgp_logo_watermark.png")
    words = sum(len(t.split()) for blk in BLOCKS for t in ([blk[1]] if blk[0] == "p" else blk[1] if blk[0] in ("bullets", "numbered") else []))
    print(f"wrote {out}; {len(b.num.cite_order)} references cited of {len(REFS)}; body prose ≈ {words:,} words")
    unused = sorted(set(REFS) - set(b.num.cite_order))
    if unused:
        print("uncited (left out of the list):", unused)
