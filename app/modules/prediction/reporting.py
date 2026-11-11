"""PDF report generation (FE-3) built on reportlab.

``PdfReportBuilder`` is a small, reusable document builder (headings,
paragraphs, key/value blocks, tables, line charts, bullet lists) used by the
token forecast report here and by Module 8's market research reports.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from io import BytesIO
from typing import Any

from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.core.timeutils import utcnow

ACCENT = colors.HexColor("#F0B90B")
INK = colors.HexColor("#1E2329")
MUTED = colors.HexColor("#707A8A")
SERIES_COLOURS = [
    colors.HexColor("#F0B90B"),
    colors.HexColor("#0ECB81"),
    colors.HexColor("#F6465D"),
    colors.HexColor("#5B8DEF"),
]


class PdfReportBuilder:
    def __init__(self, title: str, *, subtitle: str | None = None, author: str = "Quantachain Intelligence") -> None:
        self.title = title
        self.subtitle = subtitle
        self.author = author
        styles = getSampleStyleSheet()
        self._styles = {
            "title": ParagraphStyle("qc-title", parent=styles["Title"], textColor=INK, fontSize=20, spaceAfter=4),
            "subtitle": ParagraphStyle(
                "qc-subtitle", parent=styles["Normal"], textColor=MUTED, fontSize=10, spaceAfter=12
            ),
            "h1": ParagraphStyle("qc-h1", parent=styles["Heading2"], textColor=INK, spaceBefore=12, spaceAfter=6),
            "body": ParagraphStyle("qc-body", parent=styles["Normal"], fontSize=9.5, leading=13),
            "small": ParagraphStyle("qc-small", parent=styles["Normal"], fontSize=8, leading=10, textColor=MUTED),
        }
        self._story: list[Any] = [
            Paragraph(title, self._styles["title"]),
            Paragraph(
                subtitle or f"Generated {utcnow().strftime('%Y-%m-%d %H:%M UTC')} by {author}", self._styles["subtitle"]
            ),
        ]

    # ------------------------------------------------------------- blocks
    def heading(self, text: str) -> PdfReportBuilder:
        self._story.append(Paragraph(text, self._styles["h1"]))
        return self

    def paragraph(self, text: str) -> PdfReportBuilder:
        self._story.append(Paragraph(text, self._styles["body"]))
        self._story.append(Spacer(1, 4))
        return self

    def key_values(self, items: Sequence[tuple[str, Any]], *, columns: int = 2) -> PdfReportBuilder:
        cells = [
            [Paragraph(f"<b>{k}</b>", self._styles["body"]), Paragraph(_fmt(v), self._styles["body"])] for k, v in items
        ]
        rows: list[list[Any]] = []
        for i in range(0, len(cells), columns):
            row: list[Any] = []
            for cell in cells[i : i + columns]:
                row.extend(cell)
            while len(row) < columns * 2:
                row.extend(["", ""])
            rows.append(row)
        width = (A4[0] - 4 * cm) / (columns * 2)
        table = Table(rows, colWidths=[width] * (columns * 2))
        table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        self._story.append(table)
        self._story.append(Spacer(1, 6))
        return self

    def table(self, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> PdfReportBuilder:
        data = [[Paragraph(f"<b>{h}</b>", self._styles["small"]) for h in headers]]
        data += [[Paragraph(_fmt(c), self._styles["body"]) for c in row] for row in rows]
        table = Table(data, repeatRows=1, hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F5F5F5")),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAFAFA")]),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        self._story.append(table)
        self._story.append(Spacer(1, 8))
        return self

    def line_chart(
        self, series: dict[str, Sequence[float]], *, x_labels: Sequence[str] | None = None, height: float = 6 * cm
    ) -> PdfReportBuilder:
        width = A4[0] - 4 * cm
        drawing = Drawing(width, height + 1.2 * cm)
        plot = LinePlot()
        plot.x, plot.y, plot.width, plot.height = 1.2 * cm, 0.9 * cm, width - 1.6 * cm, height - 0.4 * cm
        plot.data = [
            [(float(i), float(v)) for i, v in enumerate(values) if v is not None] for values in series.values()
        ] or [[(0.0, 0.0)]]
        for index in range(len(plot.data)):  # plot.lines is an unbounded property collection; never iterate it
            plot.lines[index].strokeColor = SERIES_COLOURS[index % len(SERIES_COLOURS)]
            plot.lines[index].strokeWidth = 1.2
        plot.xValueAxis.visibleLabels = False
        plot.yValueAxis.labelTextFormat = _axis_label
        plot.yValueAxis.labels.fontSize = 7
        drawing.add(plot)
        y_text = 0.2 * cm
        if x_labels:
            drawing.add(String(1.2 * cm, y_text, str(x_labels[0]), fontSize=7, fillColor=MUTED))
            drawing.add(String(width - 3.5 * cm, y_text, str(x_labels[-1]), fontSize=7, fillColor=MUTED))
        legend_x = 1.2 * cm
        for index, name in enumerate(series):
            drawing.add(
                String(
                    legend_x,
                    height + 0.5 * cm,
                    f"■ {name}",
                    fontSize=7,
                    fillColor=SERIES_COLOURS[index % len(SERIES_COLOURS)],
                )
            )
            legend_x += 3.2 * cm
        self._story.append(drawing)
        self._story.append(Spacer(1, 6))
        return self

    def bullet_list(self, items: Sequence[str]) -> PdfReportBuilder:
        self._story.append(
            ListFlowable(
                [ListItem(Paragraph(i, self._styles["body"])) for i in items], bulletType="bullet", leftIndent=12
            )
        )
        self._story.append(Spacer(1, 6))
        return self

    def disclaimer(self, text: str | None = None) -> PdfReportBuilder:
        self._story.append(Spacer(1, 10))
        self._story.append(
            Paragraph(
                text
                or "This report is generated automatically from market, on-chain and sentiment data and statistical / machine-learning "
                "models. Forecasts are probabilistic estimates, not guarantees, and nothing here constitutes financial advice.",
                self._styles["small"],
            )
        )
        return self

    # -------------------------------------------------------------- output
    def build(self) -> bytes:
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            leftMargin=2 * cm,
            rightMargin=2 * cm,
            topMargin=1.8 * cm,
            bottomMargin=1.8 * cm,
            title=self.title,
            author=self.author,
        )
        doc.build(list(self._story))
        return buffer.getvalue()


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.2f}"
        return f"{value:.4g}" if abs(value) < 1 else f"{value:,.2f}"
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M UTC")
    return str(value)


def _axis_label(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.2f}"


# ------------------------------------------------------------ token report
def build_token_report(symbol: str, data: dict[str, Any]) -> bytes:
    """Confidence-scored forecast report for one token.

    ``data`` keys: ``price``, ``price_history`` ([(time, close)]), ``horizons`` (list of dicts),
    ``volatility`` (dict|None), ``sentiment`` (dict|None), ``whales`` (dict|None),
    ``accuracy`` (dict|None), ``model`` (str), ``weights`` (dict).
    """
    builder = PdfReportBuilder(
        f"{symbol} Forecast Report", subtitle=f"QuantaAI Predict · generated {utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
    )
    builder.heading("Snapshot")
    accuracy = data.get("accuracy") or {}
    builder.key_values(
        [
            ("Current price", data.get("price")),
            ("Model", data.get("model")),
            (
                "Directional accuracy (evaluated)",
                f"{accuracy['directional_accuracy_pct']:.1f}% over {accuracy['count']} forecasts"
                if accuracy.get("count")
                else "not enough evaluated forecasts yet",
            ),
            ("Signal weights", ", ".join(f"{k} {v:.0%}" for k, v in (data.get("weights") or {}).items())),
        ]
    )

    history = data.get("price_history") or []
    if history:
        builder.heading("Price history (1h closes)")
        builder.line_chart(
            {"close": [c for _, c in history]},
            x_labels=[t.strftime("%d %b %H:%M") if isinstance(t, datetime) else str(t) for t, _ in history],
        )

    horizons = data.get("horizons") or []
    if horizons:
        builder.heading("Forecast")
        builder.table(
            ["Horizon", "Target", "Change", "Band (90%)", "Direction", "Confidence", "Risk/Reward"],
            [
                [
                    h.get("horizon"),
                    h.get("target_price"),
                    f"{h.get('change_pct', 0):+.2f}%",
                    f"{_fmt(h.get('lower'))} – {_fmt(h.get('upper'))}",
                    h.get("direction"),
                    f"{(h.get('confidence') or 0) * 100:.0f}%",
                    f"{h['risk_reward']:.2f}:1" if h.get("risk_reward") is not None else "—",
                ]
                for h in horizons
            ],
        )
        path = next((h for h in horizons if h.get("path_points")), None)
        if path:
            builder.line_chart(
                {"forecast": path["path_points"], "lower": path["path_lower"], "upper": path["path_upper"]},
                x_labels=["now", f"+{len(path['path_points'])}h"],
            )

    vol = data.get("volatility")
    if vol:
        builder.heading("Volatility")
        builder.key_values(
            [
                ("Realised 24h (annualised)", f"{vol.get('realized_24h_pct', 0):.1f}%"),
                ("EWMA", f"{vol.get('ewma_pct', 0):.1f}%"),
                ("GARCH(1,1)", f"{vol['garch_pct']:.1f}%" if vol.get("garch_pct") is not None else "n/a"),
                ("Next 24h expected move (1σ)", f"±{vol.get('forecast_24h_horizon_pct', 0):.2f}%"),
                ("Regime", f"{vol.get('regime')} (percentile {vol.get('percentile')})"),
            ]
        )

    sentiment = data.get("sentiment")
    if sentiment and sentiment.get("index") is not None:
        builder.heading("Sentiment")
        builder.paragraph(
            f"Latest sentiment index <b>{sentiment['index']}</b> ({str(sentiment.get('label', '')).replace('_', ' ')}) from {sentiment.get('sample_size', 0)} analysed items."
        )

    whales = data.get("whales")
    if whales and whales.get("transfers"):
        builder.heading("On-chain activity (24h)")
        builder.key_values(
            [
                ("Whale transfers", whales.get("transfers")),
                ("Volume", f"${whales.get('volume_usd', 0):,.0f}"),
                ("Exchange netflow", f"${whales.get('exchange_netflow_usd', 0):,.0f}"),
            ]
        )

    builder.heading("How to read this report")
    builder.bullet_list(
        [
            "Targets come from an ensemble of a damped-trend model and an autoregressive return model (or a trained LSTM when available), nudged by sentiment and exchange-flow factors.",
            "The band is a 90% prediction interval from a residual bootstrap; wider bands mean lower confidence.",
            "Confidence blends band tightness, the model's recent directional accuracy and data sufficiency.",
            "Risk/reward compares the distance to target with an ATR-based stop distance.",
        ]
    )
    builder.disclaimer()
    return builder.build()


__all__ = ["PdfReportBuilder", "build_token_report"]
