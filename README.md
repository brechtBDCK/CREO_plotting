## CREO Data Tools

This repository provides a minimal, config-driven workflow for converting CREO `.grt` data files to Excel and generating publication-quality plots from Excel using Python. 

### Features
- **Convert** CREO `.grt` files to Excel (`.xlsx`) for easy data inspection and further analysis.
- **Plot** data from Excel with flexible, YAML-configurable options (columns, ranges, plot style, output format, etc.).
- **Modern plotting** with Plotly (interactive HTML) or Seaborn/Matplotlib (static images).

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
	- Example config is provided and well-commented.

4. **Generate plots from Excel:**
	```bash
	python creo_excel_to_plots.py
	```
	- Plots will be saved to the `plots_out/` directory (configurable).
	- Output format: interactive HTML (Plotly) or PNG/PDF/SVG (Seaborn/Matplotlib).

---

## File Overview

- `creo_GRT_to_excel.py` — Converts CREO `.grt` files to Excel. Run directly or import the function.
- `creo_excel_to_plots.py` — Plots data from Excel using a YAML config. Highly flexible.
- `config.yaml` — Main configuration file for plotting (input, filters, plot style, output).
- `requirements.txt` — Python dependencies.
---


## Requirements

- Python 3.8+
- See `requirements.txt` for required packages (pandas, numpy, plotly, seaborn, matplotlib, PyYAML, etc.)

