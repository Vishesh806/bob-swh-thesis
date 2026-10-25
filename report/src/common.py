"""Block constructors shared by the content files."""
from pathlib import Path

ROOT = Path("/Users/visheshagarwal/Desktop/bob_swh_thesis")
FIG = ROOT / "figures"
RES = ROOT / "results"


def C(key, title):            return ("chapter", key, title)
def H2(key, title):           return ("h2", key, title)
def H3(key, title):           return ("h3", key, title)
def P(text):                  return ("p", text)
def B(items):                 return ("bullets", items)
def N(items):                 return ("numbered", items)
def EQ(key, latex):           return ("eq", key, latex)
def F(key, name, width, cap): return ("fig", key, FIG / name, width, cap)
def T(key, cap, header, rows, widths): return ("table", key, cap, header, rows, widths)
