import typer
import matplotlib.pyplot as plt
from pathlib import Path

app = typer.Typer(help="Build charts from text files with multiple datasets.")

def parse_data_file(file_path):
    data_dict = {}
    with open(file_path) as f:
        lines = f.read().splitlines()
    key = None
    values = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.endswith(":"):
            if key:
                data_dict[key] = [float(v.strip()) for v in ",".join(values).split(",")]
            key = line[:-1].strip()
            values = []
        else:
            values.append(line)
    if key:
        data_dict[key] = [float(v.strip()) for v in ",".join(values).split(",")]
    return data_dict

@app.command("build")
def build(
    file: str = typer.Argument(..., help="Path to data text file."),
    bar: bool = typer.Option(False, "--bar", help="Build a bar chart."),
    line: bool = typer.Option(False, "--line", help="Build a line chart."),
    scatter: bool = typer.Option(False, "--scatter", help="Build a scatter chart."),
):
    file_path = Path(file)
    if not file_path.exists():
        typer.echo(f"File '{file}' not found.")
        raise typer.Exit(code=1)

    data = parse_data_file(file_path)

    if "x_data" not in data or "y_data" not in data:
        typer.echo("File must contain at least 'x_data' and 'y_data'.")
        raise typer.Exit(code=1)
    x = data["x_data"]

    y_series_keys = [k for k in data.keys() if k != "x_data"]

    for key in y_series_keys:
        if len(data[key]) != len(x):
            typer.echo(f"Length mismatch: '{key}' has {len(data[key])} points, x_data has {len(x)} points.")
            raise typer.Exit(code=1)

    title = typer.prompt("Chart title", default="My Chart")
    xlabel = typer.prompt("X-axis label", default="X-axis")
    ylabel = typer.prompt("Y-axis label", default="Y-axis")

    colors = {}
    legends = {}
    for key in y_series_keys:
        color = typer.prompt(f"Color for {key}", default="blue")
        colors[key] = color
        if len(y_series_keys) > 1:
            legend = typer.prompt(f"Legend label for {key}", default=key)
            legends[key] = legend
        else:
            legends[key] = key

    chart_type = None
    if bar:
        chart_type = "bar"
    elif line:
        chart_type = "line"
    elif scatter:
        chart_type = "scatter"
    else:
        chart_type = "line"

    plt.figure(figsize=(8,6))
    for key in y_series_keys:
        y = data[key]
        if chart_type == "bar":
            plt.bar(x, y, label=legends[key], color=colors[key], alpha=0.7)
        elif chart_type == "line":
            plt.plot(x, y, label=legends[key], color=colors[key], marker='o')
        elif chart_type == "scatter":
            plt.scatter(x, y, label=legends[key], color=colors[key])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if len(y_series_keys) > 1:
        plt.legend()
    plt.tight_layout()

    try:
        plt.show()
    except:
        typer.echo("Unable to display chart; will save automatically.")

    save_chart = typer.prompt("Do you want to save the chart as PNG? (y/n)").strip().lower()
    if save_chart == "y":
        filename = file_path.stem + f"_{chart_type}.png"
        plt.savefig(filename, dpi=300)
        typer.echo(f"Chart saved as: {filename}")
    plt.close()

if __name__ == "__main__":
    app()