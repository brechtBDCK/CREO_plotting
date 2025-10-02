
from pathlib import Path
import sys
import math
import yaml
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ------------------------------- Helpers -------------------------------- #

def coerce_x_col(df: pd.DataFrame, x_col_setting):
    """Accepts an integer index or a column name; falls back to first column."""
    cols = list(df.columns)
    if isinstance(x_col_setting, int):
        return cols[x_col_setting] if 0 <= x_col_setting < len(cols) else cols[0]
    if x_col_setting is None or str(x_col_setting) not in cols:
        return cols[0]
    return str(x_col_setting)


def apply_row_slicing(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Slice rows by absolute indices (start_row/end_row) and/or by fraction.
    Fraction modes:
      - which + value: first / last / middle   (value in (0,1])
      - window: keep rows in [start, end] where 0 <= start < end <= 1
    """
    filt = (cfg.get("filter") or {})

    # 1) Absolute row slicing (optional)
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
        if f0 >= f1:                           # invalid window -> no-op
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


def resolve_excel_path(p: Path) -> Path:
    """Small convenience: try current dir and /mnt/data if the provided path doesn't exist."""
    if p.exists():
        return p
    # try by basename in cwd
    alt = Path.cwd() / p.name
    if alt.exists():
        return alt
    # try /mnt/data
    alt2 = Path("/mnt/data") / p.name
    if alt2.exists():
        return alt2
    return p  # return as-is; caller will error out


# ------------------------------- Main ----------------------------------- #

def main():
    # Load YAML config next to this script
    cfg_path = Path("config.yaml")
    if not cfg_path.exists():
        print("ERROR: config.yaml not found (expected in current working directory).", file=sys.stderr)
        sys.exit(2)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    # ---- Input config
    in_cfg     = cfg.get("input", {}) or {}
    excel_path = Path(in_cfg.get("excel_path", ""))
    usecols    = in_cfg.get("usecols", None)  # list or None
    x_col_cfg  = in_cfg.get("x_col", None)    # int or str
    order      = (in_cfg.get("order") or "index").lower()   # 'index' | 'x'
    dropna_x   = bool(in_cfg.get("dropna_x", True))
    dedupe_x   = in_cfg.get("dedupe_x", None)               # None|'first'|'last'|'mean'|'median'

    # ---- Plot config
    plot_cfg     = cfg.get("plot", {}) or {}
    variables    = plot_cfg.get("variables", "all")          # "all" or list
    combine      = str(plot_cfg.get("combine", "together")).lower()  # together|separate|grid
    grid_cols    = int(plot_cfg.get("grid_cols", 2))
    title        = plot_cfg.get("title", "Excel Plots")
    template     = plot_cfg.get("plotly_style", plot_cfg.get("template", "plotly_white"))
    line_width   = float(plot_cfg.get("line_width", 1.5))
    line_shape   = str(plot_cfg.get("line_shape", "linear"))
    connect_gaps = bool(plot_cfg.get("connect_gaps", False))

    # ---- Output config
    out_cfg    = cfg.get("output", {}) or {}
    out_dir    = Path(out_cfg.get("dir", "plots_out"))
    save_files = bool(out_cfg.get("save", True))
    fmt        = str(out_cfg.get("format", "html")).lower()      # "html"|"png"|"pdf"|"svg"
    show_plot  = bool(out_cfg.get("show", False))

    # ---- Load Excel
    excel_path = resolve_excel_path(excel_path)
    if not excel_path.exists():
        print(f"ERROR: Excel file not found: {excel_path}", file=sys.stderr)
        sys.exit(3)

    df = pd.read_excel(excel_path, usecols=usecols)

    # Determine X column name
    x_col = coerce_x_col(df, x_col_cfg)

    # Coerce X: try datetime first; if it fails, numeric; else leave as-is
    if not is_numeric_dtype(df[x_col]):
        try:
            df[x_col] = pd.to_datetime(df[x_col])
        except Exception:
            df[x_col] = pd.to_numeric(df[x_col], errors="coerce")

    # Basic X cleanup
    if dropna_x:
        df = df.dropna(subset=[x_col])

    # Order: keep natural row order (index) by default; or sort by X if requested
    if order == "x":
        df = df.sort_values(by=x_col, kind="mergesort")  # stable sort

    # Optional row slicing
    df = apply_row_slicing(df, cfg)

    # Choose Y columns
    if isinstance(variables, str) and variables.lower() == "all":
        y_cols = [c for c in df.select_dtypes(include="number").columns if c != x_col]
    else:
        y_cols = [c for c in (variables or []) if c in df.columns and c != x_col]

    if not y_cols:
        print("No Y columns to plot (check 'plot.variables' in config).", file=sys.stderr)
        print("Available columns:", list(df.columns))
        sys.exit(4)

    # Optionally deduplicate duplicate X values
    if dedupe_x in ("first", "last"):
        df = df.sort_values(by=x_col, kind="mergesort").drop_duplicates(subset=[x_col], keep=dedupe_x)
    elif dedupe_x in ("mean", "median"):
        agg_fn = np.mean if dedupe_x == "mean" else np.median
        agg_map = {c: agg_fn for c in y_cols}
        df = df.groupby(x_col, as_index=False).agg(agg_map)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded: {excel_path}")
    print(f"x_col:  {x_col}")
    print(f"y_cols: {y_cols}")
    print(f"Rows:   {len(df)}")
    print(f"Combine: {combine}")

    # ---------------------------- Plotly only ---------------------------- #
    if combine == "together":
        # One figure with all Y series
        fig = px.line(df, x=x_col, y=y_cols, title=title, template=template, line_shape=line_shape)
        fig.update_traces(line=dict(width=line_width), connectgaps=connect_gaps)
        out = out_dir / f"plot_together.{ 'html' if fmt not in ('png','pdf','svg') else fmt }"

    elif combine == "separate":
        # One figure per Y column
        last_out = None
        for col in y_cols:
            fig = px.line(df, x=x_col, y=col, title=f"{title} – {col}", template=template, line_shape=line_shape)
            fig.update_traces(line=dict(width=line_width), connectgaps=connect_gaps)
            last_out = out_dir / f"plot_{col}.{ 'html' if fmt not in ('png','pdf','svg') else fmt }"
            if save_files:
                if last_out.suffix.lstrip('.').lower() in ("png", "pdf", "svg"):
                    # requires kaleido for static export
                    try:
                        fig.write_image(last_out)
                        print(f"Saved {last_out}")
                    except Exception:
                        last_out_html = last_out.with_suffix(".html")
                        fig.write_html(last_out_html, include_plotlyjs="cdn")
                        print(f"Saved {last_out_html} (install 'kaleido' for static export).")
                else:
                    fig.write_html(last_out, include_plotlyjs="cdn")
                    print(f"Saved {last_out}")
            if show_plot:
                fig.show()
        return  # finished separate

    elif combine == "grid":
        # Subplot grid: one trace per Y column
        rows = max(1, (len(y_cols) + grid_cols - 1) // grid_cols)
        fig = make_subplots(rows=rows, cols=grid_cols, subplot_titles=y_cols, shared_xaxes=False)
        r, c = 1, 1
        for col in y_cols:
            fig.add_trace(go.Scatter(x=df[x_col], y=df[col], mode="lines", name=col,
                                     line=dict(width=line_width), connectgaps=connect_gaps),
                          row=r, col=c)
            c += 1
            if c > grid_cols:
                c = 1
                r += 1
        fig.update_layout(title_text=title, template=template, showlegend=False)
        out = out_dir / f"plot_grid.{ 'html' if fmt not in ('png','pdf','svg') else fmt }"

    else:
        print(f"Unknown combine mode: {combine}", file=sys.stderr)
        sys.exit(5)

    # ---- Save output
    if save_files:
        if out.suffix.lstrip('.').lower() in ("png", "pdf", "svg"):
            # requires kaleido for static images
            try:
                fig.write_image(out)
                print(f"Saved {out}")
            except Exception:
                out_html = out.with_suffix(".html")
                fig.write_html(out_html, include_plotlyjs="cdn")
                print(f"Saved {out_html} (install 'kaleido' for static export).")
        else:
            fig.write_html(out, include_plotlyjs="cdn")
            print(f"Saved {out}")
    if show_plot:
        fig.show()


if __name__ == "__main__":
    main()
