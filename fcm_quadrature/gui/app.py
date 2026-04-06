"""
FCM Cut Cell Quadrature Weight Explorer

GUI for computing quadrature weights for cut cells using neural network
inference and moment fitting reference computation.

Input: two points A and B on the cell boundary [-1,1]².
Cut line goes from A to B. Polygon is on the left side of A→B.
Polygon vertices are auto-computed by walking the boundary CCW from B to A.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Polygon as MplPolygon

from fcm_quadrature.data_generation.moment_fitting import MomentFittingQuadrature, fit_quadrature_weights

import tensorflow as tf

# ============================================================
# Constants
# ============================================================

MODEL_WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..',
    'experiments', 'small_model_runs', 'best_models',
    'model_052_w256_d5_lr1e-03_bs16384_relu', 'checkpoints', 'best_model.weights.h5'
)

# Model architecture parameters (model_052: w256, d5, relu)
MODEL_INPUT_DIM = 12
MODEL_NUM_OUTPUTS = 4
MODEL_WIDTH = 256
MODEL_DEPTH = 5
MODEL_ACTIVATION = 'relu'

XI = 1.0 / np.sqrt(3.0)

# Gauss points ordered as in moment_fitting.py:get_standard_gauss_points_2d
GAUSS_POINTS = np.array([
    [-XI, -XI],  # G0
    [-XI,  XI],  # G1
    [ XI, -XI],  # G2
    [ XI,  XI],  # G3
])

# 12 target points matching training data (from Job_moment_fitting.py:getTargets)
TARGET_POINTS = np.array([
    [ 1.0,  1.0],   # T0: cell vertex
    [-1.0,  1.0],   # T1: cell vertex
    [-1.0, -1.0],   # T2: cell vertex
    [ 1.0, -1.0],   # T3: cell vertex
    [ 0.998,  1.0],  # T4: edge 0 near start
    [-0.998,  1.0],  # T5: edge 0 near end
    [-1.0,  0.998],  # T6: edge 1 near start
    [-1.0, -0.998],  # T7: edge 1 near end
    [-0.998, -1.0],  # T8: edge 2 near start
    [ 0.998, -1.0],  # T9: edge 2 near end
    [ 1.0, -0.998],  # T10: edge 3 near start
    [ 1.0,  0.998],  # T11: edge 3 near end
])

TARGET_LABELS = [
    "Cell vertex (1,1)", "Cell vertex (-1,1)",
    "Cell vertex (-1,-1)", "Cell vertex (1,-1)",
    "Edge 0 near (1,1)", "Edge 0 near (-1,1)",
    "Edge 1 near (-1,1)", "Edge 1 near (-1,-1)",
    "Edge 2 near (-1,-1)", "Edge 2 near (1,-1)",
    "Edge 3 near (1,-1)", "Edge 3 near (1,1)",
]

# Vandermonde matrix: V[i,j] = phi_j(gauss_point_i) for basis {1, x, y, xy}
VANDERMONDE = np.array([
    [1.0, -XI, -XI,  XI * XI],   # G0: (-xi, -xi)
    [1.0, -XI,  XI, -XI * XI],   # G1: (-xi, +xi)
    [1.0,  XI, -XI, -XI * XI],   # G2: (+xi, -xi)
    [1.0,  XI,  XI,  XI * XI],   # G3: (+xi, +xi)
])

# Cell corners in CCW order (matching Mesh2D.py Rectangle vertices)
# Edge 0: corner0 → corner1 (top,    right to left)
# Edge 1: corner1 → corner2 (left,   top to bottom)
# Edge 2: corner2 → corner3 (bottom, left to right)
# Edge 3: corner3 → corner0 (right,  bottom to top)
CELL_CORNERS = np.array([
    [ 1.0,  1.0],   # Corner 0 (start of edge 0)
    [-1.0,  1.0],   # Corner 1 (start of edge 1)
    [-1.0, -1.0],   # Corner 2 (start of edge 2)
    [ 1.0, -1.0],   # Corner 3 (start of edge 3)
])

# Presets: cut line endpoints (A, B) where cut goes A→B, polygon on the left.
# The polygon is auto-computed by walking the boundary CCW from B to A.
PRESETS = {
    "Half cell (vertical)":     {"A": np.array([0.0, -1.0]),  "B": np.array([0.0,  1.0])},
    "Half cell (horizontal)":   {"A": np.array([1.0,  0.0]),  "B": np.array([-1.0, 0.0])},
    "Triangle (top-right)":     {"A": np.array([0.5,  1.0]),  "B": np.array([1.0, -0.3])},
    "Triangle (bottom-left)":   {"A": np.array([-0.3, -1.0]), "B": np.array([-1.0, 0.7])},
    "Quadrilateral (diagonal)": {"A": np.array([0.0,  1.0]),  "B": np.array([0.5, -1.0])},
    "Pentagon":                 {"A": np.array([1.0,  0.3]),  "B": np.array([-0.5, 1.0])},
}


# ============================================================
# Main GUI Class
# ============================================================

class CutCellGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("FCM Cut Cell Quadrature Weight Explorer")
        self.root.geometry("1600x950")

        self.model = None
        self.status_var = tk.StringVar(value="Initializing...")

        self._nn_weights = None
        self._mf_weights = None
        self._polygon = None      # auto-computed polygon vertices
        self._cut_start = None    # point A
        self._cut_end = None      # point B
        self._drag_idx = None     # 0 = dragging A, 1 = dragging B

        self._setup_styles()
        self._create_main_layout()
        self._create_left_panel()
        self._create_right_panel()
        self._create_status_bar()

        self._load_model()
        self._apply_preset("Half cell (vertical)")

    # ----------------------------------------------------------------
    # Styling
    # ----------------------------------------------------------------

    def _setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Colors
        self.C_BG = '#f5f6fa'
        self.C_CARD = '#ffffff'
        self.C_ACCENT = '#2563eb'
        self.C_ACCENT_HOVER = '#1d4ed8'
        self.C_TEXT = '#1e293b'
        self.C_TEXT_DIM = '#64748b'
        self.C_BORDER = '#e2e8f0'
        self.C_GREEN = '#16a34a'
        self.C_RED = '#dc2626'
        self.C_ORANGE = '#ea580c'
        self.C_ROW_ALT = '#f8fafc'

        self.root.configure(bg=self.C_BG)

        self.style.configure('TFrame', background=self.C_BG)
        self.style.configure('Card.TFrame', background=self.C_CARD, relief='solid', borderwidth=1)
        self.style.configure('TLabel', background=self.C_BG, foreground=self.C_TEXT,
                             font=('Segoe UI', 10))
        self.style.configure('Header.TLabel', background=self.C_BG, foreground=self.C_TEXT,
                             font=('Segoe UI', 10, 'bold'))
        self.style.configure('Section.TLabel', background=self.C_CARD, foreground=self.C_TEXT_DIM,
                             font=('Segoe UI', 9, 'bold'))
        self.style.configure('Dim.TLabel', background=self.C_CARD, foreground=self.C_TEXT_DIM,
                             font=('Segoe UI', 9))
        self.style.configure('CardBg.TFrame', background=self.C_CARD)
        self.style.configure('CardBg.TLabel', background=self.C_CARD, foreground=self.C_TEXT,
                             font=('Segoe UI', 10))

        self.style.configure('TLabelframe', background=self.C_CARD, foreground=self.C_TEXT,
                             borderwidth=1, relief='solid', padding=8)
        self.style.configure('TLabelframe.Label', background=self.C_CARD, foreground=self.C_ACCENT,
                             font=('Segoe UI', 10, 'bold'))

        self.style.configure('Accent.TButton', font=('Segoe UI', 11, 'bold'), padding=(12, 6))
        self.style.map('Accent.TButton',
                       background=[('active', self.C_ACCENT_HOVER), ('!active', self.C_ACCENT)],
                       foreground=[('active', 'white'), ('!active', 'white')])

        self.style.configure('TButton', font=('Segoe UI', 9), padding=(6, 3))

        self.style.configure('TEntry', padding=3)

        # Treeview
        self.style.configure('Results.Treeview', font=('Consolas', 9), rowheight=22,
                             background=self.C_CARD, fieldbackground=self.C_CARD,
                             foreground=self.C_TEXT, borderwidth=0)
        self.style.configure('Results.Treeview.Heading', font=('Segoe UI', 9, 'bold'),
                             background=self.C_BORDER, foreground=self.C_TEXT,
                             borderwidth=0, relief='flat')
        self.style.map('Results.Treeview', background=[('selected', '#dbeafe')],
                       foreground=[('selected', self.C_TEXT)])

        self.style.configure('TPanedwindow', background=self.C_BG)

    # ----------------------------------------------------------------
    # UI Construction
    # ----------------------------------------------------------------

    def _create_main_layout(self):
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        self.left_frame = ttk.Frame(self.main_paned, width=580)
        self.right_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.left_frame, weight=0)
        self.main_paned.add(self.right_frame, weight=1)

    def _create_left_panel(self):
        # Scrollable canvas
        canvas = tk.Canvas(self.left_frame, width=560, highlightthickness=0, bg=self.C_BG)
        scrollbar = ttk.Scrollbar(self.left_frame, orient="vertical", command=canvas.yview)
        self.scroll_frame = ttk.Frame(canvas)
        self.scroll_frame.bind(
            "<Configure>",
            lambda _: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        def _on_mousewheel_linux(event):
            if event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", _on_mousewheel_linux)
        canvas.bind_all("<Button-5>", _on_mousewheel_linux)

        pad = dict(fill="x", padx=8, pady=4)

        # ---- Cut Line Section ----
        cut_frame = ttk.LabelFrame(self.scroll_frame, text="  Cut Line  ", padding=8)
        cut_frame.pack(**pad)

        hint = ttk.Label(cut_frame,
                         text="Cut goes A \u2192 B. Polygon is on the left side. Drag A/B on plot.",
                         style='Dim.TLabel')
        hint.pack(anchor="w", pady=(0, 6))

        # Point A (start)
        a_frame = ttk.Frame(cut_frame, style='CardBg.TFrame')
        a_frame.pack(fill="x", pady=2)
        ttk.Label(a_frame, text="A (start)", width=9, style='CardBg.TLabel',
                  font=('Consolas', 10, 'bold'), foreground='#dc2626').pack(side="left", padx=(0, 4))
        ttk.Label(a_frame, text="x", style='Dim.TLabel').pack(side="left")
        self.ax_var = tk.StringVar(value="0.0000")
        ttk.Entry(a_frame, textvariable=self.ax_var, width=10, font=('Consolas', 10)).pack(side="left", padx=(2, 8))
        ttk.Label(a_frame, text="y", style='Dim.TLabel').pack(side="left")
        self.ay_var = tk.StringVar(value="-1.0000")
        ttk.Entry(a_frame, textvariable=self.ay_var, width=10, font=('Consolas', 10)).pack(side="left", padx=(2, 8))

        # Point B (end)
        b_frame = ttk.Frame(cut_frame, style='CardBg.TFrame')
        b_frame.pack(fill="x", pady=2)
        ttk.Label(b_frame, text="B (end)  ", width=9, style='CardBg.TLabel',
                  font=('Consolas', 10, 'bold'), foreground='#2563eb').pack(side="left", padx=(0, 4))
        ttk.Label(b_frame, text="x", style='Dim.TLabel').pack(side="left")
        self.bx_var = tk.StringVar(value="0.0000")
        ttk.Entry(b_frame, textvariable=self.bx_var, width=10, font=('Consolas', 10)).pack(side="left", padx=(2, 8))
        ttk.Label(b_frame, text="y", style='Dim.TLabel').pack(side="left")
        self.by_var = tk.StringVar(value="1.0000")
        ttk.Entry(b_frame, textvariable=self.by_var, width=10, font=('Consolas', 10)).pack(side="left", padx=(2, 8))

        # Swap button
        ttk.Button(cut_frame, text="\u2194 Swap A \u2194 B (flip polygon side)",
                   command=self._swap_ab).pack(fill="x", pady=(4, 2))

        ttk.Separator(cut_frame, orient='horizontal').pack(fill='x', pady=6)

        # Presets
        preset_row = ttk.Frame(cut_frame, style='CardBg.TFrame')
        preset_row.pack(fill="x")
        ttk.Label(preset_row, text="Preset:", style='CardBg.TLabel').pack(side="left")
        self.preset_var = tk.StringVar(value="Half cell (vertical)")
        preset_combo = ttk.Combobox(preset_row, textvariable=self.preset_var,
                                    values=list(PRESETS.keys()), state="readonly", width=24)
        preset_combo.pack(side="left", padx=6)
        ttk.Button(preset_row, text="Apply",
                   command=lambda: self._apply_preset(self.preset_var.get())).pack(side="left")

        ttk.Separator(cut_frame, orient='horizontal').pack(fill='x', pady=6)

        # Computed polygon vertices (read-only)
        ttk.Label(cut_frame, text="Computed polygon vertices:", style='Dim.TLabel').pack(anchor="w")
        self.poly_info_var = tk.StringVar(value="--")
        ttk.Label(cut_frame, textvariable=self.poly_info_var, style='CardBg.TLabel',
                  font=('Consolas', 9), wraplength=520).pack(anchor="w", pady=(2, 0))

        # ---- Compute Button ----
        compute_frame = ttk.Frame(self.scroll_frame)
        compute_frame.pack(fill="x", padx=8, pady=(6, 4))
        self.compute_btn = tk.Button(
            compute_frame, text="Compute All", command=self._compute_all,
            bg=self.C_ACCENT, fg='white', activebackground=self.C_ACCENT_HOVER,
            activeforeground='white', font=('Segoe UI', 11, 'bold'),
            relief='flat', cursor='hand2', bd=0, pady=6
        )
        self.compute_btn.pack(fill="x")

        # ---- Distances Section ----
        dist_frame = ttk.LabelFrame(self.scroll_frame, text="  Signed Distances  ", padding=4)
        dist_frame.pack(**pad)

        cols_d = ('idx', 'label', 'dist')
        self.dist_tree = ttk.Treeview(dist_frame, columns=cols_d, show='headings', height=8,
                                      style='Results.Treeview', selectmode='none')
        self.dist_tree.heading('idx', text='#', anchor='w')
        self.dist_tree.heading('label', text='Target Point', anchor='w')
        self.dist_tree.heading('dist', text='Distance', anchor='e')
        self.dist_tree.column('idx', width=44, minwidth=36, stretch=False)
        self.dist_tree.column('label', width=280, minwidth=160)
        self.dist_tree.column('dist', width=150, minwidth=100, anchor='e')
        self.dist_tree.tag_configure('odd', background=self.C_ROW_ALT)
        self.dist_tree.pack(fill="x")

        # ---- Weights Section ----
        wt_frame = ttk.LabelFrame(self.scroll_frame, text="  Quadrature Weights  ", padding=4)
        wt_frame.pack(**pad)

        cols_w = ('pt', 'nn', 'mf', 'diff', 'rel')
        self.wt_tree = ttk.Treeview(wt_frame, columns=cols_w, show='headings', height=6,
                                    style='Results.Treeview', selectmode='none')
        self.wt_tree.heading('pt', text='Point', anchor='w')
        self.wt_tree.heading('nn', text='NN Weight', anchor='e')
        self.wt_tree.heading('mf', text='MF Weight', anchor='e')
        self.wt_tree.heading('diff', text='Diff', anchor='e')
        self.wt_tree.heading('rel', text='Rel %', anchor='e')
        self.wt_tree.column('pt', width=140, minwidth=100, stretch=False)
        self.wt_tree.column('nn', width=120, minwidth=90, anchor='e')
        self.wt_tree.column('mf', width=120, minwidth=90, anchor='e')
        self.wt_tree.column('diff', width=100, minwidth=70, anchor='e')
        self.wt_tree.column('rel', width=80, minwidth=60, anchor='e')
        self.wt_tree.tag_configure('odd', background=self.C_ROW_ALT)
        self.wt_tree.tag_configure('sum_row', background=self.C_BORDER, font=('Consolas', 9, 'bold'))
        self.wt_tree.tag_configure('good', foreground=self.C_GREEN)
        self.wt_tree.tag_configure('warn', foreground=self.C_ORANGE)
        self.wt_tree.tag_configure('bad', foreground=self.C_RED)
        self.wt_tree.pack(fill="x")

        self.wt_status_var = tk.StringVar(value="")
        self.wt_status_label = ttk.Label(wt_frame, textvariable=self.wt_status_var,
                                         style='Dim.TLabel')
        self.wt_status_label.pack(anchor="w", pady=(3, 0))

        # ---- Moments Section ----
        mom_frame = ttk.LabelFrame(self.scroll_frame, text="  Moment Values  {1, x, y, xy}  ", padding=4)
        mom_frame.pack(**pad)

        cols_m = ('basis', 'nn', 'mf', 'exact', 'err', 'rel')
        self.mom_tree = ttk.Treeview(mom_frame, columns=cols_m, show='headings', height=4,
                                     style='Results.Treeview', selectmode='none')
        self.mom_tree.heading('basis', text='Basis', anchor='w')
        self.mom_tree.heading('nn', text='NN (V\u1d40w)', anchor='e')
        self.mom_tree.heading('mf', text='MF (V\u1d40w)', anchor='e')
        self.mom_tree.heading('exact', text='Exact', anchor='e')
        self.mom_tree.heading('err', text='NN Err', anchor='e')
        self.mom_tree.heading('rel', text='Rel %', anchor='e')
        self.mom_tree.column('basis', width=60, minwidth=50, stretch=False)
        self.mom_tree.column('nn', width=110, minwidth=80, anchor='e')
        self.mom_tree.column('mf', width=110, minwidth=80, anchor='e')
        self.mom_tree.column('exact', width=110, minwidth=80, anchor='e')
        self.mom_tree.column('err', width=90, minwidth=60, anchor='e')
        self.mom_tree.column('rel', width=80, minwidth=60, anchor='e')
        self.mom_tree.tag_configure('odd', background=self.C_ROW_ALT)
        self.mom_tree.tag_configure('good', foreground=self.C_GREEN)
        self.mom_tree.tag_configure('warn', foreground=self.C_ORANGE)
        self.mom_tree.tag_configure('bad', foreground=self.C_RED)
        self.mom_tree.pack(fill="x")

        # ---- Bilinear Function Section ----
        bilin_frame = ttk.LabelFrame(self.scroll_frame,
                                     text="  Bilinear Function  f(x,y) = a + bx + cy + dxy  ", padding=8)
        bilin_frame.pack(fill="x", padx=8, pady=(4, 10))

        coeff_frame = ttk.Frame(bilin_frame, style='CardBg.TFrame')
        coeff_frame.pack(fill="x", pady=(0, 6))

        self.coeff_vars = {}
        for i, (name, default) in enumerate([("a", "1.0"), ("b", "0.0"), ("c", "0.0"), ("d", "0.0")]):
            ttk.Label(coeff_frame, text=f"{name}:", style='CardBg.TLabel',
                      font=('Segoe UI', 10, 'bold')).grid(row=0, column=i * 2, padx=(6, 2))
            var = tk.StringVar(value=default)
            entry = ttk.Entry(coeff_frame, textvariable=var, width=8, font=('Consolas', 10))
            entry.grid(row=0, column=i * 2 + 1, padx=(0, 6))
            self.coeff_vars[name] = var

        ttk.Button(bilin_frame, text="Evaluate Integral",
                   command=self._evaluate_bilinear).pack(fill="x", pady=(2, 6))

        self.bilin_result_frame = ttk.Frame(bilin_frame, style='CardBg.TFrame')
        self.bilin_result_frame.pack(fill="x")
        self.bilin_labels = {}
        for key, label_text in [('fn', 'f(x,y)'), ('nn', 'NN integral'), ('mf', 'MF integral'), ('exact', 'Exact')]:
            row = ttk.Frame(self.bilin_result_frame, style='CardBg.TFrame')
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=f"{label_text}:", width=14, anchor='w',
                      style='CardBg.TLabel').pack(side="left")
            val_label = ttk.Label(row, text="--", anchor='e',
                                  style='CardBg.TLabel', font=('Consolas', 10))
            val_label.pack(side="right")
            self.bilin_labels[key] = val_label

    def _create_right_panel(self):
        self.fig = Figure(figsize=(7, 7), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.right_frame)
        self.toolbar.update()

        # Connect mouse events for dragging A and B
        self.canvas.mpl_connect('button_press_event', self._on_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('button_release_event', self._on_release)

    def _create_status_bar(self):
        status_frame = ttk.Frame(self.root, relief="sunken")
        status_frame.pack(fill="x", side="bottom", padx=5, pady=2)
        ttk.Label(status_frame, textvariable=self.status_var).pack(side="left")

    # ----------------------------------------------------------------
    # Cut Line Management
    # ----------------------------------------------------------------

    def _get_cut_points(self):
        """Parse A and B from entry fields."""
        ax = float(self.ax_var.get())
        ay = float(self.ay_var.get())
        bx = float(self.bx_var.get())
        by = float(self.by_var.get())
        return np.array([ax, ay]), np.array([bx, by])

    def _set_cut_points(self, a, b):
        """Set A and B entry fields."""
        self.ax_var.set(f"{a[0]:.4f}")
        self.ay_var.set(f"{a[1]:.4f}")
        self.bx_var.set(f"{b[0]:.4f}")
        self.by_var.set(f"{b[1]:.4f}")

    def _swap_ab(self):
        """Swap A and B to flip which side of the cut is the polygon."""
        try:
            a, b = self._get_cut_points()
        except ValueError:
            return
        self._set_cut_points(b, a)
        self._recompute(silent=True)

    def _apply_preset(self, name):
        if name not in PRESETS:
            return
        p = PRESETS[name]
        self._set_cut_points(p["A"], p["B"])
        self._recompute(silent=True)

    # ----------------------------------------------------------------
    # Geometry Helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _snap_to_boundary(x, y):
        """Snap a point to the nearest position on the [-1,1]^2 boundary."""
        candidates = []
        # Edge 0: y=1, x in [-1,1]
        cx = np.clip(x, -1, 1)
        candidates.append((cx, 1.0, (cx - x)**2 + (1.0 - y)**2))
        # Edge 1: x=-1, y in [-1,1]
        cy = np.clip(y, -1, 1)
        candidates.append((-1.0, cy, (-1.0 - x)**2 + (cy - y)**2))
        # Edge 2: y=-1, x in [-1,1]
        cx2 = np.clip(x, -1, 1)
        candidates.append((cx2, -1.0, (cx2 - x)**2 + (-1.0 - y)**2))
        # Edge 3: x=1, y in [-1,1]
        cy2 = np.clip(y, -1, 1)
        candidates.append((1.0, cy2, (1.0 - x)**2 + (cy2 - y)**2))
        best = min(candidates, key=lambda c: c[2])
        return best[0], best[1]

    @staticmethod
    def _get_edge_index(x, y):
        """Determine which edge (0-3) a boundary point is on.

        Edge 0: top (y=1), Edge 1: left (x=-1),
        Edge 2: bottom (y=-1), Edge 3: right (x=1).
        Priority order handles corner ambiguity consistently.
        """
        tol = 1e-9
        if abs(y - 1.0) < tol:
            return 0
        if abs(x + 1.0) < tol:
            return 1
        if abs(y + 1.0) < tol:
            return 2
        if abs(x - 1.0) < tol:
            return 3
        return 0  # fallback

    @staticmethod
    def _build_polygon(cut_start, cut_end):
        """Build cut cell polygon from two boundary points.

        Cut goes from cut_start (A) to cut_end (B).
        Polygon is on the left of A->B.
        Vertices: [B, ...CCW boundary corners..., A]

        Mirrors getCutVertices from Mesh2D.py:
        Walk boundary CCW from B's edge to A's edge, collecting cell corners.
        """
        edge_a = CutCellGUI._get_edge_index(cut_start[0], cut_start[1])
        edge_b = CutCellGUI._get_edge_index(cut_end[0], cut_end[1])

        vertices = [np.array(cut_end, dtype=float)]
        current_edge = edge_b
        for _ in range(4):  # at most 4 edges to traverse
            if current_edge == edge_a:
                break
            current_edge = (current_edge + 1) % 4
            vertices.append(CELL_CORNERS[current_edge].copy())
        vertices.append(np.array(cut_start, dtype=float))

        # Remove consecutive duplicate vertices (happens when A or B is at a corner)
        cleaned = [vertices[0]]
        for v in vertices[1:]:
            if not np.allclose(v, cleaned[-1], atol=1e-10):
                cleaned.append(v)
        if len(cleaned) > 1 and np.allclose(cleaned[0], cleaned[-1], atol=1e-10):
            cleaned.pop()

        return np.array(cleaned)

    # ----------------------------------------------------------------
    # Interactive Dragging (A and B along boundary)
    # ----------------------------------------------------------------

    def _on_press(self, event):
        """On mouse press, check if near A or B and start dragging."""
        if event.inaxes != self.ax or event.button != 1:
            return
        if self.toolbar.mode:
            return
        try:
            a, b = self._get_cut_points()
        except ValueError:
            return

        click = np.array([event.xdata, event.ydata])
        xlim = self.ax.get_xlim()
        pick_radius = (xlim[1] - xlim[0]) * 0.04

        dist_a = np.linalg.norm(a - click)
        dist_b = np.linalg.norm(b - click)

        if dist_a < pick_radius and dist_a <= dist_b:
            self._drag_idx = 0  # dragging A
        elif dist_b < pick_radius:
            self._drag_idx = 1  # dragging B

    def _on_motion(self, event):
        """During drag, snap to boundary and recompute everything live."""
        if self._drag_idx is None or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        sx, sy = self._snap_to_boundary(event.xdata, event.ydata)

        if self._drag_idx == 0:
            self.ax_var.set(f"{sx:.4f}")
            self.ay_var.set(f"{sy:.4f}")
        else:
            self.bx_var.set(f"{sx:.4f}")
            self.by_var.set(f"{sy:.4f}")

        self._recompute(silent=True)

    def _on_release(self, _event):
        """End drag."""
        self._drag_idx = None

    # ----------------------------------------------------------------
    # Model Loading
    # ----------------------------------------------------------------

    def _load_model(self):
        try:
            model = tf.keras.Sequential(name="fnn_model")
            model.add(tf.keras.Input(shape=(MODEL_INPUT_DIM,), dtype='float32'))
            for i in range(MODEL_DEPTH):
                model.add(tf.keras.layers.Dense(
                    MODEL_WIDTH, activation=MODEL_ACTIVATION,
                    kernel_initializer='he_normal', bias_initializer='zeros',
                    name=f'dense{i}'
                ))
            model.add(tf.keras.layers.Dense(
                MODEL_NUM_OUTPUTS, activation=None,
                kernel_initializer='he_normal', bias_initializer='zeros',
                name='outputLayer'
            ))
            model.load_weights(MODEL_WEIGHTS_PATH)
            self.model = model
            self.status_var.set(f"Model loaded: {os.path.basename(os.path.dirname(os.path.dirname(MODEL_WEIGHTS_PATH)))}")
        except Exception as e:
            self.model = None
            self.status_var.set(f"Model load failed: {e}")

    # ----------------------------------------------------------------
    # Computation
    # ----------------------------------------------------------------

    def _compute_signed_distances(self, cut_start, cut_end):
        """Compute signed distances from cut line A->B to all target points."""
        start_h = np.array([cut_start[0], cut_start[1], 1.0])
        end_h = np.array([cut_end[0], cut_end[1], 1.0])
        line_vec = np.cross(start_h, end_h)
        a, b, c = line_vec
        norm = np.sqrt(a * a + b * b)
        if norm < 1e-15:
            raise ValueError("Cut line has zero length (A and B overlap).")
        distances = (a * TARGET_POINTS[:, 0] + b * TARGET_POINTS[:, 1] + c) / norm
        return distances

    def _run_nn_inference(self, signed_distances):
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        input_data = signed_distances.reshape(1, -1).astype(np.float32)
        prediction = self.model(input_data, training=False)
        return prediction.numpy()[0]

    def _compute_reference(self, vertices):
        mfq = MomentFittingQuadrature()
        weights, verification = fit_quadrature_weights(GAUSS_POINTS, vertices, verify=True)
        exact_moments = mfq.compute_moments(vertices, method='green')
        return weights, exact_moments, verification

    def _compute_moments_from_weights(self, weights):
        return VANDERMONDE.T @ weights

    def _recompute(self, silent=False):
        """Run the full computation pipeline.

        If silent=True, errors are swallowed (used during interactive drag).
        """
        try:
            a, b = self._get_cut_points()
        except ValueError as e:
            if not silent:
                messagebox.showerror("Input Error", f"Invalid coordinates: {e}")
            return False

        # Snap to boundary
        a = np.array(self._snap_to_boundary(a[0], a[1]))
        b = np.array(self._snap_to_boundary(b[0], b[1]))

        if np.allclose(a, b, atol=1e-10):
            if not silent:
                messagebox.showerror("Input Error", "A and B must be different points.")
            return False

        # Build polygon from cut line
        try:
            vertices = self._build_polygon(a, b)
        except Exception as e:
            if not silent:
                messagebox.showerror("Geometry Error", str(e))
            return False

        if len(vertices) < 3:
            if not silent:
                messagebox.showerror("Geometry Error",
                                     "Could not form a valid polygon. Try placing A and B on different edges.")
            return False

        self._polygon = vertices
        self._cut_start = a
        self._cut_end = b

        # Update polygon info display
        vert_strs = [f"({v[0]:.3f}, {v[1]:.3f})" for v in vertices]
        self.poly_info_var.set("  \u2192  ".join(vert_strs))

        # Compute signed distances
        try:
            distances = self._compute_signed_distances(a, b)
        except ValueError as e:
            if not silent:
                messagebox.showerror("Geometry Error", str(e))
            return False

        # NN inference
        nn_weights = None
        if self.model is not None:
            try:
                nn_weights = self._run_nn_inference(distances)
            except Exception as e:
                if not silent:
                    messagebox.showwarning("NN Error", f"Inference failed: {e}")

        # Moment fitting reference
        try:
            mf_weights, exact_moments, verification = self._compute_reference(vertices)
        except Exception as e:
            if not silent:
                messagebox.showerror("Moment Fitting Error", str(e))
            return False

        self._nn_weights = nn_weights
        self._mf_weights = mf_weights

        # Moment approximations
        nn_moments = self._compute_moments_from_weights(nn_weights) if nn_weights is not None else None
        mf_moments = self._compute_moments_from_weights(mf_weights)

        # Update displays
        self._update_distances_display(distances)
        self._update_weights_display(nn_weights, mf_weights, verification)
        self._update_moments_display(nn_moments, mf_moments, exact_moments)
        self._update_plot(vertices, a, b, nn_weights, mf_weights)

        self.status_var.set("Computation complete.")
        return True

    def _compute_all(self):
        self._recompute(silent=False)

    # ----------------------------------------------------------------
    # Bilinear function evaluation
    # ----------------------------------------------------------------

    def _evaluate_bilinear(self):
        if self._nn_weights is None and self._mf_weights is None:
            messagebox.showinfo("Info", "Run 'Compute All' first.")
            return

        try:
            a_coeff = float(self.coeff_vars["a"].get())
            b_coeff = float(self.coeff_vars["b"].get())
            c_coeff = float(self.coeff_vars["c"].get())
            d_coeff = float(self.coeff_vars["d"].get())
        except ValueError:
            messagebox.showerror("Input Error", "Coefficients must be valid numbers.")
            return

        coeffs = np.array([a_coeff, b_coeff, c_coeff, d_coeff])

        # Function display
        terms = []
        if a_coeff != 0: terms.append(f'{a_coeff:g}')
        if b_coeff != 0: terms.append(f'{b_coeff:+g}x')
        if c_coeff != 0: terms.append(f'{c_coeff:+g}y')
        if d_coeff != 0: terms.append(f'{d_coeff:+g}xy')
        self.bilin_labels['fn'].configure(text=' '.join(terms) if terms else '0')

        if self._nn_weights is not None:
            nn_moments = self._compute_moments_from_weights(self._nn_weights)
            nn_result = np.dot(coeffs, nn_moments)
            self.bilin_labels['nn'].configure(text=f'{nn_result: .10f}')
        else:
            self.bilin_labels['nn'].configure(text='N/A')

        mf_moments = self._compute_moments_from_weights(self._mf_weights)
        mf_result = np.dot(coeffs, mf_moments)
        self.bilin_labels['mf'].configure(text=f'{mf_result: .10f}')

        try:
            if self._polygon is not None:
                mfq = MomentFittingQuadrature()
                exact = mfq.compute_moments(self._polygon, method='green')
                exact_result = np.dot(coeffs, exact)
                self.bilin_labels['exact'].configure(text=f'{exact_result: .10f}')
            else:
                self.bilin_labels['exact'].configure(text='--')
        except Exception:
            self.bilin_labels['exact'].configure(text='--')

    # ----------------------------------------------------------------
    # Display Updates
    # ----------------------------------------------------------------

    def _error_tag(self, err, ref):
        """Return a color tag based on relative error magnitude."""
        if ref != 0:
            rel = abs(err / ref)
        else:
            rel = abs(err)
        if rel < 1e-3:
            return 'good'
        if rel < 1e-1:
            return 'warn'
        return 'bad'

    def _update_distances_display(self, distances):
        self.dist_tree.delete(*self.dist_tree.get_children())
        for i in range(12):
            tag = ('odd',) if i % 2 else ()
            self.dist_tree.insert('', 'end', values=(
                f'T{i}',
                TARGET_LABELS[i],
                f'{distances[i]: .6f}'
            ), tags=tag)

    def _update_weights_display(self, nn_weights, mf_weights, verification):
        self.wt_tree.delete(*self.wt_tree.get_children())

        for i in range(4):
            gx, gy = GAUSS_POINTS[i]
            nn_val = f'{nn_weights[i]: .8f}' if nn_weights is not None else 'N/A'
            mf_val = f'{mf_weights[i]: .8f}'
            if nn_weights is not None:
                diff_val = nn_weights[i] - mf_weights[i]
                diff_str = f'{diff_val: .2e}'
                if mf_weights[i] != 0:
                    rel_pct = abs(diff_val / mf_weights[i]) * 100
                    rel_str = f'{rel_pct:.4f}%'
                else:
                    rel_str = f'{abs(diff_val):.2e}'
                tag = self._error_tag(diff_val, mf_weights[i])
            else:
                diff_str = 'N/A'
                rel_str = 'N/A'
                tag = ''
            row_tag = ('odd', tag) if i % 2 else (tag,)
            self.wt_tree.insert('', 'end', values=(
                f'G{i} ({gx:+.3f},{gy:+.3f})',
                nn_val, mf_val, diff_str, rel_str
            ), tags=row_tag)

        # Sum row
        nn_sum = f'{np.sum(nn_weights): .8f}' if nn_weights is not None else 'N/A'
        mf_sum = f'{np.sum(mf_weights): .8f}'
        self.wt_tree.insert('', 'end', values=('\u03a3  Sum', nn_sum, mf_sum, '', ''), tags=('sum_row',))

        # Verification status
        if verification:
            valid = verification['is_valid']
            icon = '\u2713' if valid else '\u2717'
            self.wt_status_var.set(f'{icon} MF verification: {"PASS" if valid else "FAIL"}  '
                                   f'(max err: {verification["max_error"]:.2e})')
            self.wt_status_label.configure(
                foreground=self.C_GREEN if valid else self.C_RED)

    def _update_moments_display(self, nn_moments, mf_moments, exact_moments):
        self.mom_tree.delete(*self.mom_tree.get_children())
        basis_names = ['\u03c6=1', '\u03c6=x', '\u03c6=y', '\u03c6=xy']

        for j in range(4):
            nn_val = f'{nn_moments[j]: .8f}' if nn_moments is not None else 'N/A'
            mf_val = f'{mf_moments[j]: .8f}'
            ex_val = f'{exact_moments[j]: .8f}'
            if nn_moments is not None:
                err = abs(nn_moments[j] - exact_moments[j])
                err_str = f'{err: .2e}'
                if exact_moments[j] != 0:
                    rel_pct = abs(err / exact_moments[j]) * 100
                    rel_str = f'{rel_pct:.4f}%'
                else:
                    rel_str = f'{err:.2e}'
                tag = self._error_tag(err, abs(exact_moments[j]))
            else:
                err_str = 'N/A'
                rel_str = 'N/A'
                tag = ''
            row_tag = ('odd', tag) if j % 2 else (tag,)
            self.mom_tree.insert('', 'end', values=(
                basis_names[j], nn_val, mf_val, ex_val, err_str, rel_str
            ), tags=row_tag)

    # ----------------------------------------------------------------
    # Visualization
    # ----------------------------------------------------------------

    def _update_plot(self, vertices, cut_start, cut_end, nn_weights, mf_weights):
        self.ax.clear()

        # Unit square
        sq = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1], [1, 1]])
        self.ax.plot(sq[:, 0], sq[:, 1], 'k--', linewidth=1.5, label='Unit Cell [-1,1]\u00b2')

        # Cut cell polygon
        poly_patch = MplPolygon(vertices, closed=True,
                                facecolor='lightblue', edgecolor='blue',
                                alpha=0.4, linewidth=2, label='Cut Cell')
        self.ax.add_patch(poly_patch)

        # Cut line with arrow (A -> B)
        self.ax.annotate('', xy=cut_end, xytext=cut_start,
                         arrowprops=dict(arrowstyle='->', color='green', lw=2.5))
        mid_x = (cut_start[0] + cut_end[0]) / 2
        mid_y = (cut_start[1] + cut_end[1]) / 2
        self.ax.annotate('Cut Line', (mid_x, mid_y),
                         textcoords="offset points", xytext=(8, 8),
                         fontsize=8, color='green', fontweight='bold')

        # Draggable cut points A and B (larger, prominent markers)
        self.ax.plot(cut_start[0], cut_start[1], 'o', color='#dc2626', markersize=12,
                     zorder=10, markeredgecolor='white', markeredgewidth=2)
        self.ax.annotate(f'A ({cut_start[0]:.2f},{cut_start[1]:.2f})', cut_start,
                         textcoords="offset points", xytext=(8, -14),
                         fontsize=7, color='#dc2626', fontweight='bold')

        self.ax.plot(cut_end[0], cut_end[1], 'o', color='#2563eb', markersize=12,
                     zorder=10, markeredgecolor='white', markeredgewidth=2)
        self.ax.annotate(f'B ({cut_end[0]:.2f},{cut_end[1]:.2f})', cut_end,
                         textcoords="offset points", xytext=(8, -14),
                         fontsize=7, color='#2563eb', fontweight='bold')

        # Polygon vertices (computed boundary corners, smaller markers)
        for i, (vx, vy) in enumerate(vertices):
            self.ax.plot(vx, vy, 'bs', markersize=6, zorder=5)
            self.ax.annotate(f'V{i}', (vx, vy),
                             textcoords="offset points", xytext=(4, 4),
                             fontsize=6, color='blue')

        # Gauss points
        for i, (gx, gy) in enumerate(GAUSS_POINTS):
            self.ax.plot(gx, gy, 'm^', markersize=12, zorder=5,
                         label='Gauss Points' if i == 0 else '')
            ann_parts = [f'G{i}']
            if nn_weights is not None:
                ann_parts.append(f'NN: {nn_weights[i]:.4f}')
            if mf_weights is not None:
                ann_parts.append(f'MF: {mf_weights[i]:.4f}')
            self.ax.annotate('\n'.join(ann_parts), (gx, gy),
                             textcoords="offset points", xytext=(10, 8),
                             fontsize=7,
                             bbox=dict(boxstyle='round,pad=0.3',
                                       facecolor='lightyellow', alpha=0.85, edgecolor='gray'))

        # Target points
        for i, (tx, ty) in enumerate(TARGET_POINTS):
            self.ax.plot(tx, ty, 'ro', markersize=4, zorder=4,
                         label='Target Points' if i == 0 else '')
            self.ax.annotate(f'T{i}', (tx, ty),
                             textcoords="offset points", xytext=(3, 3),
                             fontsize=6, color='red')

        self.ax.set_xlim(-1.4, 1.4)
        self.ax.set_ylim(-1.4, 1.4)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper right', fontsize=8)
        self.ax.set_title('Cut Cell Visualization', fontsize=12)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')

        self.fig.tight_layout()
        self.canvas.draw()


# ============================================================
# Entry Point
# ============================================================

def main():
    root = tk.Tk()
    app = CutCellGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
