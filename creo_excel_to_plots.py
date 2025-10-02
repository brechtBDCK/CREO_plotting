from pathlib import Path
import sys
import math
import yaml
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ------------------------------- Helpers -------------------------------- #

def read_config(path: str = "config.yaml") -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        print("ERROR: config.yaml not found (expected in current working directory).", file=sys.stderr)
        sys.exit(2)
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

def load_excel(path: Path, usecols=None) -> pd.DataFrame:
    if not path.exists():
        print(f"ERROR: Excel file not found: {path}", file=sys.stderr)
        sys.exit(3)
    return pd.read_excel(path, usecols=usecols)

def coerce_x_col(df: pd.DataFrame, x_sel):
    """Accepts an integer index or a column name; falls back to first column."""
    cols = list(df.columns)
    if isinstance(x_sel, int):
        return cols[x_sel] if 0 <= x_sel < len(cols) else cols[0]
    if x_sel is None or str(x_sel) not in cols:
        return cols[0]
    return str(x_sel)

def apply_row_slicing(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Slice rows by absolute indices (start_row/end_row) and/or by fraction.
    Fraction modes:
      - which + value: first / last / middle   (value in (0,1])
      - window: keep rows in [start, end] where 0 <= start < end <= 1
    """
    filt = (cfg.get("filter") or {})

    # 1) Absolute row slicing 
    start_row = filt.get("start_row", None)
    end_row   = filt.get("end_row", None)
    if start_row is not None or end_row is not None:
        s = 0 if start_row is None else int(start_row)
        e = len(df) if end_row is None else int(end_row)
        s = max(0, s)
        e = min(len(df), e)
        df = df.iloc[s:e]

    n = len(df)
    if n == 0:
        return df

    # 2) Fractional keep
    kf = (filt.get("keep_fraction") or {})
    which = kf.get("which", None)
    value = kf.get("value", None)

    # Window mode (does NOT require 'value')
    if which == "window":
        f0 = kf.get("start", 0.0)
        f1 = kf.get("end", 1.0)
        try:
            f0 = float(f0); f1 = float(f1)
        except Exception:
            return df
        f0 = max(0.0, min(1.0, f0))
        f1 = max(0.0, min(1.0, f1))
        if f0 >= f1:  # invalid window -> no-op
            return df
        s = int(math.floor(f0 * n))
        e = int(math.ceil(f1 * n))
        return df.iloc[s:e]

    # First/last/middle modes (require 'value')
    if which in {"first", "last", "middle"} and value is not None:
        try:
            frac = float(value)
        except Exception:
            return df
        if not (0.0 < frac <= 1.0):
            return df
        k = max(1, int(round(frac * n)))
        if which == "first":
            return df.iloc[:k]
        if which == "last":
            return df.iloc[-k:]
        if which == "middle":
            start = max(0, (n - k) // 2)
            return df.iloc[start:start + k]

    return df

def choose_y_columns(df: pd.DataFrame, variables, x_col: str | None):
    """Return list of Y columns for fixed (non-interactive) mode."""
    if isinstance(variables, str) and variables.lower() == "all":
        # all numeric, excluding x
        y_cols = [c for c in df.select_dtypes(include="number").columns if c != x_col]
    else:
        y_cols = [c for c in (variables or []) if c in df.columns and c != x_col]
    if not y_cols:
        print("No Y columns to plot (check 'plot.variables' in config).", file=sys.stderr)
        print("Available columns:", list(df.columns))
        sys.exit(4)
    return y_cols

def build_interactive_figure(df: pd.DataFrame, title: str, template: str,
                             line_width: float, line_shape: str, connect_gaps: bool):
    """
    Interactive mode:
      - X candidates = all columns (assumes clean inputs)
      - Y candidates = all numeric columns
      - Dropdowns to choose X and Y; 'Y: ALL' overlays all series
    """
    x_candidates = list(df.columns)
    y_candidates = list(df.select_dtypes(include="number").columns)
    if not x_candidates:
        print("No usable X candidates found.", file=sys.stderr)
        sys.exit(6)
    if not y_candidates:
        print("No numeric Y candidates found.", file=sys.stderr)
        sys.exit(7)

    default_x = x_candidates[0]
    default_y = next((c for c in y_candidates if c != default_x), y_candidates[0])

    fig = go.Figure()
    for col in y_candidates:
        fig.add_trace(go.Scatter(
            x=df[default_x], y=df[col],
            mode="lines", name=col,
            visible=(col == default_y),
            line=dict(width=line_width),
            line_shape=line_shape,
            connectgaps=connect_gaps,
        ))
    fig.update_layout(title=title, template=template,
                      xaxis_title=default_x, yaxis_title=default_y)

    # X dropdown: update all traces' x + x-axis title
    n_traces = len(fig.data) #type: ignore
    x_buttons = []
    for name in x_candidates:
        xs = [df[name]] * n_traces
        x_buttons.append(dict(
            label=f"X: {name}",
            method="update",
            args=[{"x": xs}, {"xaxis": {"title": name}}]
        ))

    # Y dropdown: toggle single trace visibility; plus "ALL"
    y_buttons = []
    for idx, name in enumerate(y_candidates):
        visible = [i == idx for i in range(n_traces)]
        y_buttons.append(dict(
            label=f"Y: {name}",
            method="update",
            args=[{"visible": visible}, {"yaxis": {"title": name}}]
        ))
    y_buttons.append(dict(  # overlay all
        label="Y: ALL",
        method="update",
        args=[{"visible": [True] * n_traces}, {"yaxis": {"title": "value"}}]
    ))

    # Nice box arrangement (as requested)
    fig.update_layout(
        updatemenus=[
            dict(type="dropdown", x=0.0, xanchor="left", y=1.15, yanchor="top", showactive=True, buttons=x_buttons),
            dict(type="dropdown", x=0.2, xanchor="left", y=1.15, yanchor="top", showactive=True, buttons=y_buttons),
        ]
    )
    return fig

def build_fixed_figure(df: pd.DataFrame, x_col: str, y_cols: list[str], title: str,
                       template: str, line_width: float, line_shape: str, connect_gaps: bool):
    fig = px.line(df, x=x_col, y=y_cols, title=title, template=template, line_shape=line_shape)
    fig.update_traces(line=dict(width=line_width), connectgaps=connect_gaps)
    return fig

def save_figure(fig: go.Figure, out_dir: Path, base: str, fmt: str, save: bool, show: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{base}.{ 'html' if fmt not in ('png','pdf','svg') else fmt }"
    if save:
        # Keep it simple: default to HTML; static export requires kaleido (not enforced here)
        fig.write_html(out, include_plotlyjs="cdn")
        print(f"Saved {out}")
    if show:
        fig.show()


# ------------------------------- Main ----------------------------------- #

def main():
    cfg = read_config("config.yaml")

    # ---- Input
    in_cfg     = cfg.get("input", {}) or {}
    excel_path = Path(in_cfg.get("excel_path", ""))
    usecols    = in_cfg.get("usecols", None)
    x_col_cfg  = in_cfg.get("x_col", None)   # int | str | "all"

    # ---- Plot
    plot_cfg     = cfg.get("plot", {}) or {}
    variables    = plot_cfg.get("variables", "all")    # "all" or list
    title        = plot_cfg.get("title", "Excel Plots")
    template     = plot_cfg.get("plotly_style", plot_cfg.get("template", "plotly_white"))
    line_width   = float(plot_cfg.get("line_width", 1.5))
    line_shape   = str(plot_cfg.get("line_shape", "linear"))
    connect_gaps = bool(plot_cfg.get("connect_gaps", False))

    # ---- Output
    out_cfg    = cfg.get("output", {}) or {}
    out_dir    = Path(out_cfg.get("dir", "plots_out"))
    save_files = bool(out_cfg.get("save", True))
    fmt        = str(out_cfg.get("format", "html")).lower()
    show_plot  = bool(out_cfg.get("show", False))

    # ---- Data
    df = load_excel(excel_path, usecols=usecols)
    df = apply_row_slicing(df, cfg)

    # Interactive mode only when BOTH x_col=="all" and variables=="all"
    interactive_xy = (
        isinstance(x_col_cfg, str) and x_col_cfg.lower() == "all"
        and isinstance(variables, str) and variables.lower() == "all"
    )

    print(f"Loaded: {excel_path}")
    print(f"Rows:   {len(df)}")
    print(f"Mode:   {'interactive together' if interactive_xy else 'fixed together'}")

    # Build figure
    if interactive_xy:
        fig = build_interactive_figure(
            df, title, template, line_width, line_shape, connect_gaps
        )
    else:
        x_col = coerce_x_col(df, x_col_cfg)
        y_cols = choose_y_columns(df, variables, x_col)
        fig = build_fixed_figure(
            df, x_col, y_cols, title, template, line_width, line_shape, connect_gaps
        )

    # Save/show
    save_figure(fig, out_dir, base="plot_together", fmt=fmt, save=save_files, show=show_plot)


if __name__ == "__main__":
    main()
