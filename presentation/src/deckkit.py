"""A tiny slide spec -> (python-pptx deck, PIL preview, text-fit report)."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt, Emu

W, H = 13.333, 7.5
NAVY, TEAL, TINT, AMBER, GREY, INK, WHITE, PALE = "0B2545", "1B7F8C", "E3F2F3", "E8A33D", "5B6770", "1D2A33", "FFFFFF", "F3F6F7"
HEAD, BODY = "Georgia", "Arial"
FONTDIR = "/System/Library/Fonts/Supplemental/"
FONTFILES = {("Georgia", False): "Georgia.ttf", ("Georgia", True): "Georgia Bold.ttf",
             ("Arial", False): "Arial.ttf", ("Arial", True): "Arial Bold.ttf"}
_fonts = {}


def pil_font(face, bold, size_pt, px_per_in):
    key = (face, bold, size_pt, px_per_in)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(FONTDIR + FONTFILES[(face, bold)], max(1, int(round(size_pt / 72 * px_per_in))))
    return _fonts[key]


def rgb(h):
    return RGBColor.from_string(h)


# ----------------------------------------------------------------- element constructors
def text(x, y, w, h, paras, size=16, color=INK, face=BODY, bold=False, align="l", valign="t", bullets=False,
         spacing=6, italic=False, name="text"):
    """paras: str or list of str or list of (str, {"bold":..,"color":..,"size":..}) runs-lists."""
    if isinstance(paras, str):
        paras = [paras]
    return dict(kind="text", x=x, y=y, w=w, h=h, paras=paras, size=size, color=color, face=face, bold=bold,
                align=align, valign=valign, bullets=bullets, spacing=spacing, italic=italic, name=name)


def rect(x, y, w, h, fill, rounded=True, name="card", line=None):
    return dict(kind="rect", x=x, y=y, w=w, h=h, fill=fill, rounded=rounded, name=name, line=line)


def circle(x, y, d, fill, label="", size=16, color=WHITE, name="badge"):
    return dict(kind="circle", x=x, y=y, w=d, h=d, fill=fill, label=label, size=size, color=color, name=name)


def image(path, x, y, w=None, h=None, name="figure"):
    with Image.open(path) as im:
        ar = im.size[1] / im.size[0]
    if w is not None and h is not None:  # fit inside the box, keep aspect, centre
        if w * ar > h:
            nw = h / ar; x += (w - nw) / 2; w = nw
        else:
            nh = w * ar; y += (h - nh) / 2; h = nh
    elif w is not None:
        h = w * ar
    else:
        w = h / ar
    return dict(kind="image", path=str(path), x=x, y=y, w=w, h=h, name=name)


def bar_chart(x, y, w, h, cats, series, colors, horizontal=True, fmt='0.00"%"', title=None, maxv=None, name="chart",
              label_size=12, point_colors=None):
    return dict(kind="chart", x=x, y=y, w=w, h=h, cats=cats, series=series, colors=colors, horizontal=horizontal,
                fmt=fmt, title=title, maxv=maxv, name=name, label_size=label_size, point_colors=point_colors)


def table(x, y, col_w, rows, size=13, header_fill=NAVY, row_h=0.42, name="table", bold_rows=()):
    return dict(kind="table", x=x, y=y, w=sum(col_w), h=row_h * len(rows), col_w=col_w, rows=rows, size=size,
                header_fill=header_fill, row_h=row_h, name=name, bold_rows=bold_rows)


# ----------------------------------------------------------------- text measurement (shared by pptx fit-check and preview)
INSET = 0.08  # inches, applied on all sides of text boxes


def para_runs(p):
    if isinstance(p, str):
        return [(p, {})]
    return p


def wrap_lines(el, px=100):
    """Return list of (line_runs, size, bold, is_first_line_of_para) for a text element."""
    out = []
    width = (el["w"] - 2 * INSET - (0.3 if el["bullets"] else 0)) * px
    for p in el["paras"]:
        runs = para_runs(p)
        words = []
        for t, o in runs:
            for wd in t.split(" "):
                words.append((wd, o))
        line, cur, first = [], 0.0, True
        for wd, o in words:
            f = pil_font(el["face"] if not o.get("face") else o["face"], o.get("bold", el["bold"]), o.get("size", el["size"]), px)
            wl = f.getlength(wd); sp = f.getlength(" ")
            if line and cur + sp + wl > width:
                out.append((line, first)); line, cur, first = [], 0.0, False
            if line:
                cur += sp
            line.append((wd, o)); cur += wl
        out.append((line, first))
    return out


def text_height(el, px=100):
    lines = wrap_lines(el, px)
    h = 0.0
    for i, (line, first) in enumerate(lines):
        size = max([o.get("size", el["size"]) for _, o in line] or [el["size"]])
        h += size * 1.2 / 72
        if first and i > 0:
            h += el["spacing"] / 72
    return h + 2 * INSET


# ----------------------------------------------------------------- pptx writer
def add_to_slide(slide, el):
    k = el["kind"]
    X, Y, Wd, Ht = Inches(el["x"]), Inches(el["y"]), Inches(el["w"]), Inches(el["h"])
    if k == "rect":
        shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if el["rounded"] else MSO_SHAPE.RECTANGLE, X, Y, Wd, Ht)
        if el["rounded"]:
            shp.adjustments[0] = min(0.12, 0.18 / min(el["w"], el["h"]))
        shp.fill.solid(); shp.fill.fore_color.rgb = rgb(el["fill"])
        if el["line"]:
            shp.line.color.rgb = rgb(el["line"]); shp.line.width = Pt(1)
        else:
            shp.line.fill.background()
        shp.shadow.inherit = False
        shp.name = el["name"]
    elif k == "circle":
        shp = slide.shapes.add_shape(MSO_SHAPE.OVAL, X, Y, Wd, Ht)
        shp.fill.solid(); shp.fill.fore_color.rgb = rgb(el["fill"]); shp.line.fill.background(); shp.shadow.inherit = False
        tf = shp.text_frame; tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = el["label"]; r.font.size = Pt(el["size"]); r.font.bold = True
        r.font.color.rgb = rgb(el["color"]); r.font.name = BODY
        shp.name = el["name"]
    elif k == "image":
        pic = slide.shapes.add_picture(el["path"], X, Y, Wd, Ht); pic.name = el["name"]
    elif k == "text":
        tb = slide.shapes.add_textbox(X, Y, Wd, Ht); tb.name = el["name"]
        tf = tb.text_frame; tf.word_wrap = True
        for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
            setattr(tf, m, Inches(INSET))
        tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[el["valign"]]
        for i, p_spec in enumerate(el["paras"]):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[el["align"]]
            if i > 0:
                p.space_before = Pt(el["spacing"])
            if el["bullets"]:
                _bullet(p)
            for t, o in para_runs(p_spec):
                r = p.add_run(); r.text = t
                r.font.size = Pt(o.get("size", el["size"])); r.font.bold = o.get("bold", el["bold"])
                r.font.italic = o.get("italic", el["italic"])
                r.font.color.rgb = rgb(o.get("color", el["color"])); r.font.name = o.get("face", el["face"])
    elif k == "chart":
        cd = CategoryChartData(); cd.categories = el["cats"]
        for sname, vals in el["series"].items():
            cd.add_series(sname, vals)
        ctype = XL_CHART_TYPE.BAR_CLUSTERED if el["horizontal"] else XL_CHART_TYPE.COLUMN_CLUSTERED
        gf = slide.shapes.add_chart(ctype, X, Y, Wd, Ht, cd); gf.name = el["name"]
        ch = gf.chart
        ch.font.size = Pt(12); ch.font.name = BODY; ch.font.color.rgb = rgb(INK)
        ch.has_title = bool(el["title"])
        if el["title"]:
            ch.chart_title.text_frame.text = el["title"]
            tr = ch.chart_title.text_frame.paragraphs[0].runs[0]; tr.font.size = Pt(13); tr.font.bold = True
        multi = len(el["series"]) > 1
        ch.has_legend = multi
        if multi:
            ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False; ch.legend.font.size = Pt(12)
        va = ch.value_axis
        va.has_major_gridlines = True; va.major_gridlines.format.line.color.rgb = rgb("D9DEE1")
        va.format.line.fill.background(); va.tick_labels.font.size = Pt(11); va.tick_labels.font.color.rgb = rgb(GREY)
        va.minimum_scale = 0
        if el["maxv"]:
            va.maximum_scale = el["maxv"]
        va.tick_labels.number_format = el["fmt"]; va.tick_labels.number_format_is_linked = False
        ca = ch.category_axis
        ca.tick_labels.font.size = Pt(12); ca.format.line.color.rgb = rgb("A0A9AE"); ca.has_major_gridlines = False
        if el["horizontal"]:
            ca.reverse_order = True
            from pptx.enum.chart import XL_TICK_LABEL_POSITION
            va.tick_label_position = XL_TICK_LABEL_POSITION.HIGH  # keep the value axis at the bottom after reversing
        plot = ch.plots[0]; plot.gap_width = 60 if not multi else 80; plot.overlap = 0 if not multi else -10
        plot.has_data_labels = True
        dl = plot.data_labels; dl.number_format = el["fmt"]; dl.number_format_is_linked = False
        dl.font.size = Pt(el["label_size"]); dl.font.bold = True; dl.position = XL_LABEL_POSITION.OUTSIDE_END
        for s, col in zip(plot.series, el["colors"]):
            s.format.fill.solid(); s.format.fill.fore_color.rgb = rgb(col)
            s.invert_if_negative = False
        if el["point_colors"]:
            s0 = plot.series[0]
            for i, col in enumerate(el["point_colors"]):
                pt = s0.points[i]; pt.format.fill.solid(); pt.format.fill.fore_color.rgb = rgb(col)
    elif k == "table":
        rows, cols = len(el["rows"]), len(el["col_w"])
        gt = slide.shapes.add_table(rows, cols, X, Y, Wd, Inches(el["row_h"] * rows)); gt.name = el["name"]
        t = gt.table
        for ci, cw in enumerate(el["col_w"]):
            t.columns[ci].width = Inches(cw)
        for ri, row in enumerate(el["rows"]):
            t.rows[ri].height = Inches(el["row_h"])
            for ci, val in enumerate(row):
                cell = t.cell(ri, ci); cell.margin_left = cell.margin_right = Inches(0.08)
                cell.margin_top = cell.margin_bottom = Inches(0.03); cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                cell.fill.solid()
                cell.fill.fore_color.rgb = rgb(el["header_fill"] if ri == 0 else (PALE if ri % 2 == 0 else WHITE))
                tf = cell.text_frame; tf.word_wrap = True
                p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER
                r = p.add_run(); r.text = str(val); r.font.size = Pt(el["size"]); r.font.name = BODY
                r.font.bold = (ri == 0) or (ri in el["bold_rows"])
                r.font.color.rgb = rgb(WHITE if ri == 0 else INK)


def _bullet(p):
    from pptx.oxml.ns import qn
    from lxml import etree
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(Emu(Inches(0.3)))); pPr.set("indent", str(-Emu(Inches(0.25))))
    bu = etree.SubElement(pPr, qn("a:buFont")); bu.set("typeface", "Arial")
    bc = etree.SubElement(pPr, qn("a:buChar")); bc.set("char", "•")


def build_pptx(slides, out):
    prs = Presentation(); prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    blank = prs.slide_layouts[6]
    for sp in slides:
        s = prs.slides.add_slide(blank)
        bg = s.background.fill; bg.solid(); bg.fore_color.rgb = rgb(sp.get("bg", WHITE))
        for el in sp["els"]:
            add_to_slide(s, el)
        if sp.get("notes"):
            s.notes_slide.notes_text_frame.text = sp["notes"]
    prs.core_properties.title = slides[0].get("title_text", "Presentation")
    prs.save(out)


# ----------------------------------------------------------------- preview renderer + fit check
def render_preview(sp, out_png, px=110):
    img = Image.new("RGB", (int(W * px), int(H * px)), "#" + sp.get("bg", WHITE))
    d = ImageDraw.Draw(img)
    issues = []
    for el in sp["els"]:
        k = el["kind"]
        x0, y0, x1, y1 = el["x"] * px, el["y"] * px, (el["x"] + el["w"]) * px, (el["y"] + el["h"]) * px
        if x0 < -1 or y0 < -1 or x1 > W * px + 1 or y1 > H * px + 1:
            issues.append(f"{el.get('name')} off slide")
        if k == "rect":
            if el["rounded"]:
                d.rounded_rectangle([x0, y0, x1, y1], radius=0.12 * px, fill="#" + el["fill"],
                                    outline=("#" + el["line"]) if el["line"] else None)
            else:
                d.rectangle([x0, y0, x1, y1], fill="#" + el["fill"])
        elif k == "circle":
            d.ellipse([x0, y0, x1, y1], fill="#" + el["fill"])
            f = pil_font(BODY, True, el["size"], px)
            d.text(((x0 + x1) / 2, (y0 + y1) / 2), el["label"], font=f, fill="#" + el["color"], anchor="mm")
        elif k == "image":
            with Image.open(el["path"]) as im:
                im = im.convert("RGB").resize((int(x1 - x0), int(y1 - y0)))
                img.paste(im, (int(x0), int(y0)))
        elif k == "chart":
            _preview_chart(img, el, px)
        elif k == "table":
            yy = y0
            for ri, row in enumerate(el["rows"]):
                xx = x0
                for ci, cw in enumerate(el["col_w"]):
                    fill = "#" + (el["header_fill"] if ri == 0 else (PALE if ri % 2 == 0 else WHITE))
                    d.rectangle([xx, yy, xx + cw * px, yy + el["row_h"] * px], fill=fill, outline="#C9D1D6")
                    f = pil_font(BODY, ri == 0 or ri in el["bold_rows"], el["size"], px)
                    tw = f.getlength(str(row[ci]))
                    if tw > (cw - 0.16) * px:
                        issues.append(f"table cell too narrow: '{row[ci]}'")
                    tx = xx + 0.08 * px if ci == 0 else xx + cw * px / 2 - tw / 2
                    d.text((tx, yy + el["row_h"] * px / 2), str(row[ci]), font=f, fill="#" + (WHITE if ri == 0 else INK), anchor="lm")
                    xx += cw * px
                yy += el["row_h"] * px
        elif k == "text":
            need = text_height(el, px)
            if need > el["h"] + 0.02:
                issues.append(f"text overflow in {el['name']}: needs {need:.2f} in, box {el['h']:.2f} in: '{str(para_runs(el['paras'][0])[0][0])[:40]}'")
            lines = wrap_lines(el, px)
            total = need - 2 * INSET
            yy = y0 + INSET * px + {"t": 0, "m": (el["h"] - need) / 2 * px, "b": (el["h"] - need) * px}[el["valign"]]
            for i, (line, first) in enumerate(lines):
                size = max([o.get("size", el["size"]) for _, o in line] or [el["size"]])
                if first and i > 0:
                    yy += el["spacing"] / 72 * px
                widths = []
                for wd, o in line:
                    f = pil_font(o.get("face", el["face"]), o.get("bold", el["bold"]), o.get("size", el["size"]), px)
                    widths.append(f.getlength(wd + " "))
                lw = sum(widths)
                left = x0 + INSET * px + (0.3 * px if el["bullets"] else 0)
                xx = {"l": left, "c": (x0 + x1) / 2 - lw / 2, "r": x1 - INSET * px - lw}[el["align"]]
                if el["bullets"] and first:
                    d.text((left - 0.22 * px, yy), "•", font=pil_font(BODY, False, el["size"], px), fill="#" + el["color"])
                for (wd, o), wl in zip(line, widths):
                    f = pil_font(o.get("face", el["face"]), o.get("bold", el["bold"]), o.get("size", el["size"]), px)
                    d.text((xx, yy + (size - o.get("size", el["size"])) / 72 * px * 0.9), wd, font=f, fill="#" + o.get("color", el["color"]))
                    xx += wl
                yy += size * 1.2 / 72 * px
    img.save(out_png)
    return issues


def _preview_chart(img, el, px):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io
    fig, ax = plt.subplots(figsize=(el["w"], el["h"]), dpi=px)
    cats = el["cats"]; n = len(el["series"]); import numpy as np
    idx = np.arange(len(cats)); bw = 0.8 / n
    for i, ((name, vals), col) in enumerate(zip(el["series"].items(), el["colors"])):
        cols = ["#" + c for c in el["point_colors"]] if (el["point_colors"] and i == 0) else "#" + col
        pos = idx + (i - (n - 1) / 2) * bw
        if el["horizontal"]:
            ax.barh(pos, vals, bw * 0.9, color=cols, label=name)
        else:
            ax.bar(pos, vals, bw * 0.9, color=cols, label=name)
        for p_, v in zip(pos, vals):
            dec = 2 if "0.00" in el["fmt"] else 1 if "0.0" in el["fmt"] else 0
            lab = f"{v:.{dec}f}%"
            if el["horizontal"]:
                ax.text(v, p_, " " + lab, va="center", fontsize=el["label_size"] * 0.9, fontweight="bold")
            else:
                ax.text(p_, v, lab, ha="center", va="bottom", fontsize=el["label_size"] * 0.9, fontweight="bold")
    if el["horizontal"]:
        ax.set_yticks(idx, cats); ax.invert_yaxis()
    else:
        ax.set_xticks(idx, cats)
    if el["maxv"]:
        (ax.set_xlim if el["horizontal"] else ax.set_ylim)(0, el["maxv"])
    if n > 1:
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=n, frameon=False)
    if el["title"]:
        ax.set_title(el["title"], fontsize=12, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
    buf.seek(0)
    with Image.open(buf) as im:
        img.paste(im.convert("RGB").resize((int(el["w"] * px), int(el["h"] * px))), (int(el["x"] * px), int(el["y"] * px)))
