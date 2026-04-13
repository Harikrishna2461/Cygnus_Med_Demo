"""
Shunt Assessment PDF Report Generator
White and red Cygnus Medical theme — multi-finding layout.
"""

import io
import os
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

LOGO_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "frontend", "public", "logo.png")
)

# ── Colours ───────────────────────────────────────────────────
_RED      = "#C01C1C"
_DARK_RED = "#8B0000"
_LIGHT_RED= "#FFF0F0"
_GREY_BG  = "#F7F7F7"
_BORDER   = "#E0E0E0"
_TEXT     = "#1A1A1A"
_MUTED    = "#666666"
_WHITE    = "#FFFFFF"
_GREEN    = "#166534"
_GREEN_BG = "#F0FDF4"
_AMBER    = "#92400E"
_AMBER_BG = "#FFFBEB"


def generate_shunt_report_pdf(
    classification: dict[str, Any],
    clip_list: list[dict[str, Any]],
    patient_info: dict[str, str] | None = None,
) -> bytes:
    """
    Generate a PDF report.

    classification must contain either:
      - 'findings': list of per-leg finding dicts  (new multi-finding format)
      - OR top-level shunt_type/reasoning/ligation  (legacy single-finding)
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm, mm
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            HRFlowable, Image, KeepTogether,
        )
        from reportlab.lib import colors as C
    except ImportError as e:
        raise ImportError(f"reportlab required: pip install reportlab. {e}")

    # ── Helpers ───────────────────────────────────────────────
    def hex_c(h): return C.HexColor(h)

    def ps(name, **kw):
        defaults = dict(fontName="Helvetica", fontSize=10, leading=14,
                        textColor=hex_c(_TEXT))
        defaults.update(kw)
        return ParagraphStyle(name, **defaults)

    # Style set
    S = {
        "h1":     ps("h1",  fontName="Helvetica-Bold", fontSize=20,
                     textColor=hex_c(_RED), spaceAfter=2),
        "h2":     ps("h2",  fontName="Helvetica-Bold", fontSize=12,
                     textColor=hex_c(_RED), spaceBefore=10, spaceAfter=4),
        "h3":     ps("h3",  fontName="Helvetica-Bold", fontSize=10,
                     textColor=hex_c(_DARK_RED), spaceBefore=6, spaceAfter=2),
        "body":   ps("body"),
        "small":  ps("small", fontSize=8, textColor=hex_c(_MUTED), leading=11),
        "bullet": ps("bullet", leftIndent=10, spaceAfter=2,
                     textColor=hex_c(_TEXT), fontSize=9, leading=13),
        "badge":  ps("badge", fontName="Helvetica-Bold", fontSize=13,
                     textColor=C.white, alignment=TA_CENTER),
        "conf":   ps("conf",  fontName="Helvetica", fontSize=10,
                     textColor=C.white, alignment=TA_CENTER),
        "leg":    ps("leg",   fontName="Helvetica-Bold", fontSize=11,
                     textColor=hex_c(_DARK_RED)),
        "center": ps("center", alignment=TA_CENTER, fontSize=9,
                     textColor=hex_c(_MUTED)),
        "th":     ps("th",    fontName="Helvetica-Bold", fontSize=8,
                     textColor=C.white, alignment=TA_CENTER),
        "td":     ps("td",    fontSize=8, alignment=TA_CENTER, leading=11),
        "td_l":   ps("td_l",  fontSize=8, leading=11),
    }

    def td(text, style="td", **kw):
        _s = S[style]
        if kw:
            _s = ps("_td", parent=_s, **kw)
        return Paragraph(str(text), _s)

    def hr(color=_BORDER, thickness=0.5, space=4):
        return HRFlowable(width="100%", thickness=thickness,
                          color=hex_c(color), spaceBefore=space, spaceAfter=space)

    def bullet_para(text):
        clean = str(text).lstrip("•- \t").strip()
        return Paragraph(f"• {clean}", S["bullet"])

    def flag_table(flags: list[str]):
        rows = [[Paragraph(f, ps("_f", fontSize=8, fontName="Helvetica",
                                  textColor=hex_c(_AMBER), leading=12))]
                for f in flags]
        t = Table(rows, colWidths=["100%"])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), hex_c(_AMBER_BG)),
            ("BOX", (0, 0), (-1, -1), 0.5, hex_c("#F59E0B")),
            ("LEFTPADDING",  (0, 0), (-1, -1), 8),
            ("TOPPADDING",   (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ]))
        return t

    # ── Normalise findings ───────────────────────────────────
    findings = classification.get("findings")
    if not findings:
        # Legacy single-finding
        findings = [{
            "leg": "Assessment",
            "shunt_type": classification.get("shunt_type", "Unknown"),
            "confidence": classification.get("confidence", 0),
            "reasoning":  classification.get("reasoning", []),
            "ligation":   classification.get("ligation", []),
            "needs_elim_test": classification.get("needs_elim_test", False),
            "ask_diameter":    classification.get("ask_diameter", False),
            "ask_branching":   classification.get("ask_branching", False),
            "summary":    classification.get("summary", ""),
            "num_clips":  len(clip_list),
        }]

    # Clips per leg (for the per-finding table)
    clips_by_leg: dict[str, list] = {}
    for c in clip_list:
        side = (c.get("legSide") or c.get("leg_side") or "unspecified").strip().lower().capitalize()
        clips_by_leg.setdefault(side, []).append(c)
    if not any(clips_by_leg.values()):
        clips_by_leg["Assessment"] = clip_list

    # ── Build PDF ────────────────────────────────────────────
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=1.5*cm, bottomMargin=2*cm,
        title="Shunt Assessment Report", author="Cygnus Medical",
    )

    story = []

    # ── HEADER ───────────────────────────────────────────────
    logo_cell = ""
    if os.path.exists(LOGO_PATH):
        try:
            logo_cell = Image(LOGO_PATH, width=1.6*cm, height=1.0*cm)
        except Exception:
            pass

    hdr = Table(
        [[logo_cell,
          Paragraph("<b>Cygnus Medical</b>",
                    ps("hn", fontName="Helvetica-Bold", fontSize=16,
                       textColor=hex_c(_RED), alignment=TA_RIGHT))]],
        colWidths=["50%", "50%"],
    )
    hdr.setStyle(TableStyle([
        ("VALIGN", (0,0),(-1,-1),"MIDDLE"),
        ("BOTTOMPADDING",(0,0),(-1,-1),6),
    ]))
    story.append(hdr)
    story.append(hr(_RED, thickness=2, space=2))
    story.append(Spacer(1, 3*mm))

    story.append(Paragraph("Shunt Assessment Report", S["h1"]))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}",
        S["small"]))
    story.append(Spacer(1, 4*mm))

    # ── PATIENT INFO ─────────────────────────────────────────
    if patient_info:
        rows = [
            [td(k.replace("_"," ").title(), "body", fontName="Helvetica-Bold"),
             td(v, "body")]
            for k, v in patient_info.items() if v
        ]
        if rows:
            story.append(Paragraph("Assessment Information", S["h2"]))
            info_t = Table(rows, colWidths=["35%", "65%"])
            info_t.setStyle(TableStyle([
                ("BACKGROUND", (0,0),(0,-1), hex_c(_GREY_BG)),
                ("GRID",       (0,0),(-1,-1), 0.4, hex_c(_BORDER)),
                ("TOPPADDING", (0,0),(-1,-1), 5),
                ("BOTTOMPADDING",(0,0),(-1,-1),5),
                ("LEFTPADDING",(0,0),(-1,-1),8),
                ("VALIGN",    (0,0),(-1,-1),"TOP"),
            ]))
            story.append(info_t)
            story.append(Spacer(1, 4*mm))

    # ── SUMMARY BADGE ROW  (one badge per finding) ──────────
    n = len(findings)
    if n > 1:
        story.append(Paragraph("Findings Summary", S["h2"]))
        badge_cells = []
        for f in findings:
            conf = f.get("confidence", 0)
            badge_cells.append(
                Table(
                    [[Paragraph(f['leg'], ps("_bl", fontSize=8, fontName="Helvetica-Bold",
                                             textColor=C.white, alignment=TA_CENTER))],
                     [Paragraph(f['shunt_type'], ps("_bst", fontSize=11, fontName="Helvetica-Bold",
                                                     textColor=C.white, alignment=TA_CENTER))],
                     [Paragraph(f"Confidence: {conf*100:.0f}%",
                                ps("_bc", fontSize=8, textColor=C.white,
                                   alignment=TA_CENTER))]],
                    colWidths=["100%"],
                )
            )
        summary_row = Table([badge_cells], colWidths=[f"{100/n:.1f}%"]*n)
        badge_style = [
            ("BACKGROUND", (0,0),(-1,-1), hex_c(_RED)),
            ("TOPPADDING", (0,0),(-1,-1), 8),
            ("BOTTOMPADDING",(0,0),(-1,-1),8),
            ("LEFTPADDING",(0,0),(-1,-1), 4),
            ("RIGHTPADDING",(0,0),(-1,-1),4),
            ("VALIGN",    (0,0),(-1,-1),"MIDDLE"),
            ("LINEAFTER", (0,0),(-2,-1), 1.5, C.white),
        ]
        for sub in badge_cells:
            sub.setStyle(TableStyle(badge_style))
        summary_row.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1), hex_c(_RED)),
            ("LINEAFTER", (0,0),(-2,-1), 1.5, C.white),
            ("TOPPADDING",(0,0),(-1,-1),0),
            ("BOTTOMPADDING",(0,0),(-1,-1),0),
            ("LEFTPADDING",(0,0),(-1,-1),0),
            ("RIGHTPADDING",(0,0),(-1,-1),0),
        ]))
        story.append(summary_row)
        story.append(Spacer(1, 5*mm))

    # ── PER-FINDING DETAIL ───────────────────────────────────
    for fi, finding in enumerate(findings):
        leg         = finding.get("leg", "Assessment")
        shunt_type  = finding.get("shunt_type", "Unknown")
        confidence  = finding.get("confidence", 0)
        reasoning   = finding.get("reasoning", [])
        ligation    = finding.get("ligation", [])
        summary_txt = finding.get("summary", "")
        n_clips     = finding.get("num_clips", 0)

        elems = []

        # Section divider
        if fi > 0:
            elems.append(hr(_RED, thickness=1.5, space=8))

        # Leg heading
        elems.append(Paragraph(
            f"{'─'*2}  {leg} Leg   ({n_clips} clips)",
            S["leg"]))
        elems.append(Spacer(1, 2*mm))

        # Shunt type badge
        badge_row = Table(
            [[Paragraph(shunt_type,
                        ps("_b", fontName="Helvetica-Bold", fontSize=14,
                           textColor=C.white, alignment=TA_CENTER)),
              Paragraph(f"Confidence: {confidence*100:.0f}%",
                        ps("_c", fontSize=10, textColor=C.white,
                           alignment=TA_CENTER))]],
            colWidths=["65%", "35%"],
        )
        badge_row.setStyle(TableStyle([
            ("BACKGROUND", (0,0),(-1,-1), hex_c(_RED)),
            ("TOPPADDING", (0,0),(-1,-1),10),
            ("BOTTOMPADDING",(0,0),(-1,-1),10),
            ("LEFTPADDING",(0,0),(-1,-1),14),
            ("RIGHTPADDING",(0,0),(-1,-1),14),
            ("VALIGN",    (0,0),(-1,-1),"MIDDLE"),
        ]))
        elems.append(badge_row)
        elems.append(Spacer(1, 2*mm))

        # Clinical summary
        if summary_txt:
            elems.append(Paragraph(summary_txt, S["body"]))
            elems.append(Spacer(1, 2*mm))

        # Flags
        flags = []
        if finding.get("needs_elim_test"):
            flags.append("⚠  Elimination test required before ligation decision")
        if finding.get("ask_diameter"):
            flags.append("ℹ  Specify RP diameter at N2: Small or Large")
        if finding.get("ask_branching"):
            flags.append("ℹ  Specify N3 branching pattern for precise tributary ligation")
        if finding.get("ask_reflux_perforator"):
            flags.append("ℹ  Assess reflux at Hunterian perforator at follow-up")
        if flags:
            elems.append(flag_table(flags))
            elems.append(Spacer(1, 2*mm))

        # Two-column: reasoning | ligation
        left_items = ([Paragraph("Clinical Reasoning", S["h3"])] +
                      [bullet_para(r) for r in reasoning]
                      if reasoning else
                      [Paragraph("Clinical Reasoning", S["h3"]),
                       Paragraph("No pathological pattern detected.", S["body"])])

        right_items = ([Paragraph("Proposed Ligation", S["h3"])] +
                       [bullet_para(l) for l in ligation]
                       if ligation else
                       [Paragraph("Proposed Ligation", S["h3"]),
                        Paragraph("No ligation required.", S["body"])])

        two_col = Table(
            [[left_items, right_items]],
            colWidths=["50%", "50%"],
        )
        two_col.setStyle(TableStyle([
            ("VALIGN",       (0,0),(-1,-1),"TOP"),
            ("LEFTPADDING",  (0,0),(-1,-1), 4),
            ("RIGHTPADDING", (0,0),(-1,-1), 4),
            ("LINEAFTER",    (0,0),(0,-1),  0.5, hex_c(_BORDER)),
            ("BACKGROUND",   (0,0),(0,-1),  hex_c(_GREY_BG)),
        ]))
        elems.append(two_col)
        elems.append(Spacer(1, 4*mm))

        story.append(KeepTogether(elems))

    # ── CLIP TABLE ───────────────────────────────────────────
    story.append(hr(_RED, thickness=1.5, space=6))
    story.append(Paragraph(f"Complete Assessment Data — {len(clip_list)} Clips", S["h2"]))

    col_heads = ["#", "Leg", "Flow", "From", "To", "Pos (x, y)", "Elim. Test", "Step"]
    thead = [td(h, "th") for h in col_heads]
    tdata = [thead]

    for i, c in enumerate(clip_list):
        flow = c.get("flow", "")
        side = (c.get("legSide") or c.get("leg_side") or "—").capitalize()[:1]  # L/R/—
        ft   = c.get("fromType", "")
        tt   = c.get("toType", "")
        x    = c.get("posXRatio", "")
        y    = c.get("posYRatio", "")
        elim = c.get("eliminationTest", "") or "—"
        step = c.get("step", "") or "—"
        pos  = f"({float(x):.3f}, {float(y):.3f})" if x != "" else "—"

        fc = hex_c("#CC0000") if flow == "RP" else hex_c("#166534")
        row = [
            td(str(i)),
            td(side),
            Paragraph(f"<b>{flow}</b>",
                      ps(f"_f{i}", fontName="Helvetica-Bold", fontSize=8,
                         textColor=fc, alignment=TA_CENTER)),
            td(ft), td(tt),
            td(pos, "td", fontSize=7),
            td(elim),
            td(step, "td_l", fontSize=7),
        ]
        tdata.append(row)

    clip_table = Table(
        tdata,
        colWidths=[0.7*cm, 0.7*cm, 1.1*cm, 1.1*cm, 1.1*cm, 3.0*cm, 2.2*cm, None],
        repeatRows=1,
    )
    ts = [
        ("BACKGROUND", (0,0),(-1,0), hex_c(_RED)),
        ("GRID",       (0,0),(-1,-1), 0.4, hex_c(_BORDER)),
        ("VALIGN",     (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0),(-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1),3),
        ("LEFTPADDING",(0,0),(-1,-1),3),
        ("RIGHTPADDING",(0,0),(-1,-1),3),
    ]
    for i, c in enumerate(clip_list, 1):
        if c.get("flow") == "RP":
            ts.append(("BACKGROUND",(0,i),(-1,i), hex_c(_LIGHT_RED)))
        elif i % 2 == 0:
            ts.append(("BACKGROUND",(0,i),(-1,i), hex_c(_GREY_BG)))
    clip_table.setStyle(TableStyle(ts))
    story.append(clip_table)
    story.append(Spacer(1, 3*mm))

    # Legend
    leg_data = [
        [Paragraph("<b>EP</b>", ps("_ep", fontSize=8, fontName="Helvetica-Bold",
                                    textColor=hex_c(_GREEN))),
         td("Physiological (forward) flow — normal", "small")],
        [Paragraph("<b>RP</b>", ps("_rp", fontSize=8, fontName="Helvetica-Bold",
                                    textColor=hex_c("#CC0000"))),
         td("Retrograde (pathological) reflux — highlighted in pink", "small")],
    ]
    leg_t = Table(leg_data, colWidths=[1.2*cm, None])
    leg_t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1), hex_c(_GREY_BG)),
        ("GRID",(0,0),(-1,-1),0.3, hex_c(_BORDER)),
        ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
        ("LEFTPADDING",(0,0),(-1,-1),5),
    ]))
    story.append(leg_t)
    story.append(Spacer(1, 5*mm))

    # ── FOOTER ───────────────────────────────────────────────
    story.append(hr(_BORDER, space=4))
    story.append(Paragraph(
        "This report is produced by the Cygnus Medical AI-assisted CHIVA assessment system. "
        "All findings must be reviewed and validated by a qualified vascular surgeon before "
        "any clinical decision-making.",
        ps("_disc", fontSize=7.5, fontName="Helvetica-Oblique",
           textColor=hex_c(_MUTED), leading=11),
    ))

    doc.build(story)
    data = buf.getvalue()
    buf.close()
    return data
