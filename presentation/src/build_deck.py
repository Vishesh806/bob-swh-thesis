import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from deckkit import build_pptx, render_preview
from slides import SLIDES
out = Path(sys.argv[1]); prev = Path(sys.argv[2]); prev.mkdir(parents=True, exist_ok=True)
build_pptx(SLIDES, out)
for i, sp in enumerate(SLIDES, 1):
    issues = render_preview(sp, prev / f"slide-{i:02d}.png")
    for m in issues:
        print(f"slide {i}: {m}")
print(f"built {len(SLIDES)} slides -> {out}")
