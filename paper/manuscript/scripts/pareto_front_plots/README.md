# Pareto Front Plots

Script to generate publication-quality plots comparing NSGA-II variants on ZDT4 (2-objective) and DTLZ2 (3-objective).

## Data files

Located in the local `data/` subfolder (same directory as this README):

| File | Description |
|------|-------------|
| `FUN.NSGAII.ZDT4.csv` | Standard NSGA-II on ZDT4 |
| `FUN.NSGAII.ZDT4.steady_state.csv` | Steady-state NSGA-II on ZDT4 |
| `FUN.NSGAII.DTLZ2.csv` | Standard NSGA-II on DTLZ2 |
| `FUN.NSGAII.DTLZ2.archive.csv` | Unbounded-archive NSGA-II on DTLZ2 |
| `ZDT4.csv` | True Pareto front for ZDT4 |
| `DTLZ2.3D.csv` | True Pareto front for DTLZ2 |

All CSV files are headerless and comma-separated.

## Generated plots

Saved to the ignored `paper/generated/figures/` directory:

| Output file | Problem | Algorithm variant |
|-------------|---------|-------------------|
| `front_zdt4_standard.png` | ZDT4 (2D) | Standard NSGA-II |
| `front_zdt4_steady_state.png` | ZDT4 (2D) | Steady-state NSGA-II |
| `front_dtlz2_standard.png` | DTLZ2 (3D) | Standard NSGA-II |
| `front_dtlz2_archive.png` | DTLZ2 (3D) | Unbounded-archive NSGA-II |

## Plot configuration

### Style
- **Library**: Matplotlib + Seaborn (`whitegrid` theme, `paper` context, `font_scale=1.15`)
- **Font**: Serif (`font.family: serif`, `mathtext.fontset: cm`)
- **Resolution**: 600 DPI
- **Reference front colour**: `#4A4A4A` (dark grey)
- **Obtained front colour**: `#E74C3C` (red)

### 2D plots (ZDT4)
- **Figure size**: 3.2 × 2.6 inches
- **Reference front**: sorted line, linewidth 1.2
- **Obtained front**: scatter, marker size `s=8`, white edge (linewidth 0.2), alpha 0.85
- **Legend**: upper right, fontsize 7

### 3D plots (DTLZ2)
- **Figure size**: 3.6 × 3.0 inches
- **View angle**: `elev=35`, `azim=45` (equivalent to Plotly Express default camera)
- **Reference front**: scatter, marker size `s=1`, alpha 0.12 (semi-transparent point cloud)
- **Obtained front**: scatter, marker size `s=10`, white edge (linewidth 0.2), alpha 0.9
- **Legend**: upper left (`bbox_to_anchor=(0.0, 0.95)`), fontsize 7
- **Tick label size**: 7
- **Axis labels**: $f_1$, $f_2$, $f_3$ with `labelpad=4`

## Usage

```bash
python paper/manuscript/scripts/pareto_front_plots/plot_fronts.py
```

## LaTeX inclusion

The four-plot block in `main.tex` is currently commented out. These retained data
and the plotting script remain available as illustrative material; they are not
required to compile the current article or supplement. Start with the
[reviewer guide](../../../README.md) for the active paper reproduction path.
