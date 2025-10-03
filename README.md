
## CREO Data Tools

This repository provides a minimal, config-driven workflow for converting CREO `.grt` data files to Excel and generating interactive, publication-quality plots from Excel using Python and Plotly.

### Features
- **Convert** CREO `.grt` files to Excel (`.xlsx`) for easy data inspection and further analysis.
- **Plot** data from Excel with flexible, YAML-configurable options (columns, row ranges, plot style, output format, etc.).
- **Interactive Plotly HTML**: Choose X and Y columns dynamically in the browser, or overlay all Y series, if enabled in config.
- **No code changes required** for most workflows—just edit the `config.yaml`.

---

## Quick Start

1. **Install dependencies** (ideally in a virtual environment):
	 ```bash
	 pip install -r requirements.txt
	 ```

2. **Convert a `.grt` file to Excel:**
	 ```bash
	 python creo_GRT_to_excel.py
	 ```
	 - By default, this will convert the `.grt` to an Excel file in the current directory.
	 - You can modify the script or call `grt_to_excel(input_path, output_dir)` for other files.

3. **Configure your plot in `config.yaml`:**
	 - Set the path to your Excel file, columns to plot, filters, and plot style.
	 - **Interactive mode:**
		 - Set both `x_col: "all"` and `variables: "all"` to enable dropdowns for X and Y selection in the HTML plot.
	 - **Row filtering:**
		 - Use `start_row`, `end_row` for absolute slicing, or `keep_fraction` for fractional selection (window, first, last, middle).
	 - Example config is provided and well-commented.

4. **Generate plots from Excel:**
	 ```bash
	 python creo_excel_to_plots.py
	 ```
	 - Plots will be saved to the `plots_out/` directory (configurable).
	 - Output format: interactive HTML (Plotly). Static export (PNG/PDF/SVG) requires `kaleido`.

---

## File Overview

- `creo_GRT_to_excel.py` — Converts CREO `.grt` files to Excel. Run directly or import the function.
- `creo_excel_to_plots.py` — Plots data from Excel using a YAML config. Supports interactive and fixed plotting modes.
- `config.yaml` — Main configuration file for plotting (input, filters, plot style, output).
- `requirements.txt` — Python dependencies.

---

## Example: Interactive Plotting

To enable interactive X/Y selection in the HTML plot, set in `config.yaml`:

```yaml
input:
	x_col: "all"
plot:
	variables: "all"
```

This will generate a Plotly HTML file with dropdowns for X and Y columns, or overlay all Y series.

---

## Requirements

- Python 3.8+
- See `requirements.txt` for required packages (pandas, numpy, plotly, PyYAML, etc.)

