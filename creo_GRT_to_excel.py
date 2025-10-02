import os
import re
import pandas as pd

axis0_re = re.compile(r"axis\s*0\s+([^\r\n#]+)", re.IGNORECASE)
axis1_re = re.compile(r"axis\s*1\s+([^\r\n#]+)", re.IGNORECASE)
plot_re  = re.compile(r"plot\s+(\d+)\s+([^\r\n#]+)", re.IGNORECASE)
num_re   = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

def grt_to_excel(input_path, output_dir=None):
    # read file
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # title = first line or [Title]
    first_line = text.splitlines()[0].strip()
    m = re.search(r"\[([^\]]+)\]", first_line)
    title = m.group(1).strip() if m else first_line.split("axis 0")[0].strip()
    if not title:
        title = "output"

    # axis labels
    x_match = axis0_re.search(text)
    x_name = x_match.group(1).strip() if x_match else "X"

    # find plots
    headers = list(plot_re.finditer(text))
    series = []
    for i, h in enumerate(headers):
        idx = int(h.group(1))
        name = h.group(2).strip()
        start = h.end()
        end = headers[i+1].start() if i+1 < len(headers) else len(text)
        seg = text[start:end]
        vals = [float(v) for v in num_re.findall(seg)]
        if len(vals) % 2: vals = vals[:-1]
        xs, ys = vals[0::2], vals[1::2]
        series.append((idx, name, xs, ys))
    series.sort(key=lambda s: s[0])

    if not series:
        raise ValueError("No plots found in .grt file")

    # use X from first non-empty
    x = series[0][2] if series[0][2] else []
    if not x:
        for _,_,xs,_ in series:
            if xs: x = xs; break

    # build table
    data = {x_name: x}
    for _, name, xs, ys in series:
        y = ys
        if x:
            n = max(len(x), len(y))
            if len(x) < n: x = x + [None]*(n-len(x)); data[x_name] = x
            if len(y) < n: y = y + [None]*(n-len(y))
        data[name] = y

    df = pd.DataFrame(data)

    # output path
    out_dir = output_dir or os.path.dirname(input_path)
    os.makedirs(out_dir, exist_ok=True)
    out_xlsx = os.path.join(out_dir, f"{title}.xlsx")

    with pd.ExcelWriter(out_xlsx) as writer:
        df.to_excel(writer, index=False, sheet_name="Data")

    print("Excel written:", out_xlsx)

if __name__ == "__main__":
    grt_to_excel(
        "/home/bdck/PROJECTS_WSL/CREO/testgrt2.grt",
        "/home/bdck/PROJECTS_WSL/CREO/"
    )
