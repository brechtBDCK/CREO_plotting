"""
Plot from Excel with a minimal, config-driven workflow.
- Loads Excel file
- Choose what to plot, ranges, and layout via a YAML config
- Uses existing libraries (pandas, plotly, seaborn/matplotlib) with minimal custom code
"""

import argparse
from pathlib import Path
import sys
import math

import pandas as pd
import numpy as np
import yaml

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt


def main():
# -------------------------------- Load config ------------------------------- #
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    input_cfg   = cfg.get("input", {})
    filter_cfg  = cfg.get("filter", {})
    plot_cfg    = cfg.get("plot", {})
    axes_cfg    = cfg.get("axes", {})
    out_cfg     = cfg.get("output", {})

    excel_path  = Path(input_cfg.get("excel_path", ""))
    sheet_name  = input_cfg.get("sheet_name", 0)
    usecols     = input_cfg.get("usecols", None)
    x_col       = input_cfg.get("x_col", None)

    range_by        = filter_cfg.get("range_by", None)      # "rows" | "x" | None
    start_row       = filter_cfg.get("start_row", None)
    end_row         = filter_cfg.get("end_row", None)
    x_min           = filter_cfg.get("x_min", None)
    x_max           = filter_cfg.get("x_max", None)
    sample_every_n  = filter_cfg.get("sample_every_n", 1)
    sample_fraction = filter_cfg.get("sample_fraction", None)
    random_state    = filter_cfg.get("random_state", 42)

    engine      = str(plot_cfg.get("engine", "plotly")).lower()   # "plotly" | "seaborn" | "matplotlib"
    variables   = plot_cfg.get("variables", "auto")                # "auto" or list
    combine     = str(plot_cfg.get("combine", "together")).lower() # "together" | "separate" | "grid"
    grid_cols   = int(plot_cfg.get("grid_cols", 2))
    title       = plot_cfg.get("title", "Excel Plots")
    template    = plot_cfg.get("template", "plotly_white")
    seaborn_style = plot_cfg.get("seaborn_style", "whitegrid")
    line_width  = float(plot_cfg.get("line_width", 1.5))

    x_range     = axes_cfg.get("x_range", None)        # [min, max] or None
    y_range_all = axes_cfg.get("y_range_all", None)    # [min, max] or None
    y_ranges    = axes_cfg.get("y_ranges", {})         # {var: [min,max]}

    out_dir     = Path(out_cfg.get("dir", "plots_out"))
    save_files  = bool(out_cfg.get("save", True))
    fmt         = str(out_cfg.get("format", "html" if engine == "plotly" else "png")).lower()  # "html"|"png"|"pdf"|"svg"
    show_plots  = bool(out_cfg.get("show", False))


    # --------------------------------
    # Load Excel
    # --------------------------------
    if not excel_path.exists():
        print(f"ERROR: Excel file not found: {excel_path}", file=sys.stderr)
        sys.exit(2)

    df = pd.read_excel(excel_path, sheet_name=sheet_name, usecols=usecols)

    if x_col is None or x_col not in df.columns:
        # Default: first column as x-axis
        x_col = df.columns[0]

    # Try convert x to numeric or datetime where possible (non-destructive if fails)
    # This helps easy range filtering & axis formatting.
    if not np.issubdtype(df[x_col].dtype, np.number):
        try:
            df[x_col] = pd.to_datetime(df[x_col])
        except Exception:
            try:
                df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
            except Exception:
                pass

    # --------------------------------
    # Slice/filter rows
    # --------------------------------
    df = df.sort_values(by=x_col)

    if range_by == "rows":
        s = 0 if start_row is None else int(start_row)
        e = len(df) if end_row is None else int(end_row)
        df = df.iloc[s:e]
    elif range_by == "x":
        if x_min is not None:
            df = df[df[x_col] >= x_min]
        if x_max is not None:
            df = df[df[x_col] <= x_max]

    # Subsample
    if sample_every_n and int(sample_every_n) > 1:
        df = df.iloc[::int(sample_every_n)]
    if sample_fraction is not None:
        df = (df.sample(frac=float(sample_fraction), random_state=random_state)
                .sort_index())

    # --------------------------------
    # Determine Y columns
    # --------------------------------
    if variables == "auto" or variables is None:
        y_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if x_col in y_cols:
            y_cols.remove(x_col)
    else:
        y_cols = [c for c in variables if c in df.columns]

    if not y_cols:
        print("No Y columns to plot (check 'variables' in config).", file=sys.stderr)
        print("Available columns:", list(df.columns))
        sys.exit(3)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded: {excel_path}")
    print(f"Sheet:  {sheet_name}")
    print(f"x_col:  {x_col}")
    print(f"y_cols: {y_cols}")
    print(f"Rows:   {len(df)}")
    print(f"Engine: {engine}, Combine: {combine}")

    # --------------------------------
    # Plot with Plotly
    # --------------------------------
    if engine == "plotly":
        if combine == "together":
            fig = px.line(df, x=x_col, y=y_cols, title=title, template=template)
            fig.update_traces(line=dict(width=line_width))
            if x_range:
                fig.update_xaxes(range=x_range)
            if y_range_all:
                fig.update_yaxes(range=y_range_all)

            if save_files:
                out = out_dir / f"plot_together.{ 'html' if fmt not in ('png','pdf','svg') else fmt }"
                if fmt in ("png", "pdf", "svg"):
                    try:
                        fig.write_image(out)
                        print(f"Saved {out}")
                    except Exception as e:
                        # Fallback to HTML if kaleido missing
                        out_html = out.with_suffix(".html")
                        fig.write_html(out_html)
                        print(f"Saved {out_html} (install 'kaleido' for static export).")
                else:
                    fig.write_html(out)
                    print(f"Saved {out}")

            if show_plots:
                fig.show()

        elif combine == "separate":
            for col in y_cols:
                fig = px.line(df, x=x_col, y=col, title=f"{col}", template=template)
                fig.update_traces(line=dict(width=line_width))
                if x_range:
                    fig.update_xaxes(range=x_range)
                if y_ranges and col in y_ranges:
                    fig.update_yaxes(range=y_ranges[col])
                elif y_range_all:
                    fig.update_yaxes(range=y_range_all)

                if save_files:
                    out = out_dir / f"plot_{col}.{ 'html' if fmt not in ('png','pdf','svg') else fmt }"
                    if fmt in ("png", "pdf", "svg"):
                        try:
                            fig.write_image(out)
                            print(f"Saved {out}")
                        except Exception:
                            out_html = out.with_suffix(".html")
                            fig.write_html(out_html)
                            print(f"Saved {out_html} (install 'kaleido' for static export).")
                    else:
                        fig.write_html(out)
                        print(f"Saved {out}")

                if show_plots:
                    fig.show()

        elif combine == "grid":
            rows = math.ceil(len(y_cols) / grid_cols)
            fig = make_subplots(rows=rows, cols=grid_cols, subplot_titles=y_cols, shared_xaxes=False)
            r, c = 1, 1
            for col in y_cols:
                fig.add_trace(go.Scatter(x=df[x_col], y=df[col], mode="lines", name=col,
                                         line=dict(width=line_width)),
                              row=r, col=c)
                # Set per-panel ranges
                if x_range:
                    fig.update_xaxes(range=x_range, row=r, col=c)
                if y_ranges and col in y_ranges:
                    fig.update_yaxes(range=y_ranges[col], row=r, col=c)
                elif y_range_all:
                    fig.update_yaxes(range=y_range_all, row=r, col=c)

                c += 1
                if c > grid_cols:
                    c = 1
                    r += 1

            fig.update_layout(title_text=title, template=template, showlegend=False)

            if save_files:
                out = out_dir / f"plot_grid.{ 'html' if fmt not in ('png','pdf','svg') else fmt }"
                if fmt in ("png", "pdf", "svg"):
                    try:
                        fig.write_image(out)
                        print(f"Saved {out}")
                    except Exception:
                        out_html = out.with_suffix(".html")
                        fig.write_html(out_html)
                        print(f"Saved {out_html} (install 'kaleido' for static export).")
                else:
                    fig.write_html(out)
                    print(f"Saved {out}")

            if show_plots:
                fig.show()

        else:
            print(f"Unknown combine mode for Plotly: {combine}", file=sys.stderr)
            sys.exit(4)

    # --------------------------------
    # Plot with Seaborn/Matplotlib
    # --------------------------------
    else:
        sns.set_theme(style=seaborn_style, context="talk")

        if combine == "together":
            # Long-form for hue overlay
            long_df = df[[x_col] + y_cols].melt(id_vars=[x_col], var_name="variable", value_name="value")
            ax = sns.lineplot(data=long_df, x=x_col, y="value", hue="variable", linewidth=line_width)
            if x_range:
                ax.set_xlim(x_range)
            if y_range_all:
                ax.set_ylim(y_range_all)
            ax.set_title(title)
            ax.figure.tight_layout()

            if save_files:
                out = out_dir / f"plot_together.{fmt}"
                ax.figure.savefig(out, dpi=150)
                print(f"Saved {out}")
            if show_plots:
                plt.show()
            plt.close(ax.figure)

        elif combine == "separate":
            for col in y_cols:
                ax = sns.lineplot(data=df, x=x_col, y=col, linewidth=line_width)
                if x_range:
                    ax.set_xlim(x_range)
                if y_ranges and col in y_ranges:
                    ax.set_ylim(y_ranges[col])
                elif y_range_all:
                    ax.set_ylim(y_range_all)
                ax.set_title(col)
                ax.figure.tight_layout()

                if save_files:
                    out = out_dir / f"plot_{col}.{fmt}"
                    ax.figure.savefig(out, dpi=150)
                    print(f"Saved {out}")
                if show_plots:
                    plt.show()
                plt.close(ax.figure)

        elif combine == "grid":
            rows = math.ceil(len(y_cols) / grid_cols)
            fig, axes = plt.subplots(rows, grid_cols, figsize=(6*grid_cols, 4*rows), squeeze=False)
            idx = 0
            for r in range(rows):
                for c in range(grid_cols):
                    if idx < len(y_cols):
                        col = y_cols[idx]
                        ax = axes[r][c]
                        sns.lineplot(data=df, x=x_col, y=col, ax=ax, linewidth=line_width)
                        if x_range:
                            ax.set_xlim(x_range)
                        if y_ranges and col in y_ranges:
                            ax.set_ylim(y_ranges[col])
                        elif y_range_all:
                            ax.set_ylim(y_range_all)
                        ax.set_title(col)
                        idx += 1
                    else:
                        axes[r][c].axis("off")

            fig.suptitle(title)
            fig.tight_layout(rect=[0, 0, 1, 0.97])

            if save_files:
                out = out_dir / f"plot_grid.{fmt}"
                fig.savefig(out, dpi=150)
                print(f"Saved {out}")
            if show_plots:
                plt.show()
            plt.close(fig)

        else:
            print(f"Unknown combine mode for Seaborn/Matplotlib: {combine}", file=sys.stderr)
            sys.exit(5)


if __name__ == "__main__":
    main()
