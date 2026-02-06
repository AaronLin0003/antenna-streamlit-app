import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.patches as patches
from numba import njit
import time

# ==========================================
# 1. Numba 加速核心 (JIT Compiled)
# ==========================================
@njit(fastmath=True, cache=True)
def _fast_calculate_pattern(phases_rad, sv_xz, sv_yz, ep_xz, ep_yz, total_elems, elem_gain):
    weights = np.exp(1j * phases_rad)
    af_xz = np.abs(weights @ sv_xz)
    af_yz = np.abs(weights @ sv_yz)
    af_xz = np.maximum(af_xz, 1e-9)
    af_yz = np.maximum(af_yz, 1e-9)
    const_val = -10 * np.log10(total_elems) + elem_gain
    pat_xz = 20 * np.log10(af_xz * ep_xz) + const_val
    pat_yz = 20 * np.log10(af_yz * ep_yz) + const_val
    return pat_xz, pat_yz

# ==========================================
# 2. 物理物件
# ==========================================
class AntennaSystem:
    def __init__(self, Nx, Ny, freq, spacing, elem_gain, q, is_jio, is_sym):
        self.update_params(Nx, Ny, freq, spacing, elem_gain, q, is_jio, is_sym)

    def update_params(self, Nx, Ny, freq, spacing, elem_gain, q, is_jio, is_sym):
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        self.elem_gain = elem_gain
        self.q = q
        self.is_jio = is_jio
        self.is_sym = is_sym
        
        lambda_val = 299.792 / freq
        d_lambda = spacing / lambda_val
        k = 2 * np.pi
        
        self.map_idx, self.num_genes = self._create_mapping(self.Nx, self.Ny, is_jio, is_sym)
        self.total_elements = float(self.Nx * self.Ny)
        
        self.angles = np.arange(-90, 90.5, 0.5)
        self.rads = np.deg2rad(self.angles) # 儲存起來供後續使用
        
        x = np.arange(self.Nx)
        y = np.arange(self.Ny)
        X, Y = np.meshgrid(x, y, indexing='ij') 
        self.pos_x = X.flatten()
        self.pos_y = Y.flatten()
        
        u_xz = np.sin(self.rads)
        phase_xz = k * d_lambda * np.outer(self.pos_x, u_xz) 
        self.sv_xz = np.exp(1j * phase_xz).astype(np.complex128)
        self.ep_xz = (np.maximum(1e-6, np.cos(self.rads)) ** q).astype(np.float64)
        
        v_yz = np.sin(self.rads)
        phase_yz = k * d_lambda * np.outer(self.pos_y, v_yz)
        self.sv_yz = np.exp(1j * phase_yz).astype(np.complex128)
        self.ep_yz = (np.maximum(1e-6, np.cos(self.rads)) ** q).astype(np.float64)

    def _create_mapping(self, Nx, Ny, is_jio, is_sym):
        dim_nx = int(np.ceil(Nx / 2)) if is_sym else Nx
        dim_ny = int(np.ceil(Ny / 2)) if is_sym else Ny
        if is_jio: dim_ny = int(np.ceil(dim_ny / 2))
        
        counter = 0
        gene_indices = np.zeros((dim_nx, dim_ny), dtype=int)
        for i in range(dim_nx):
            for j in range(dim_ny):
                gene_indices[i, j] = counter
                counter += 1
                
        mapping = np.zeros((Nx, Ny), dtype=int)
        for x in range(Nx):
            for y in range(Ny):
                ix = x
                if is_sym and x >= dim_nx: ix = Nx - 1 - x
                iy = y
                limit_y = int(np.ceil(Ny/2)) if is_sym else Ny
                if is_sym and y >= limit_y: iy = Ny - 1 - y
                jio_iy = iy
                if is_jio: jio_iy = int(np.floor(iy / 2))
                ix = min(ix, dim_nx-1)
                jio_iy = min(jio_iy, dim_ny-1)
                mapping[x, y] = gene_indices[ix, jio_iy]
        return mapping.flatten(), counter

    def calculate_pattern(self, population_genes):
        phases = population_genes[:, self.map_idx]
        phases_rad = np.deg2rad(phases)
        return _fast_calculate_pattern(
            phases_rad, self.sv_xz, self.sv_yz, 
            self.ep_xz, self.ep_yz, self.total_elements, self.elem_gain
        )

# ==========================================
# 3. 優化器 (Optimizer)
# ==========================================
class Optimizer:
    def __init__(self, sys_model: AntennaSystem, pop_size=50):
        self.sys = sys_model
        self.pop_size = pop_size
        self.dim = sys_model.num_genes
        self.reset() 

    def reset(self):
        self.pop = np.random.uniform(-180, 180, (self.pop_size, self.dim))
        self.pop[0] = np.zeros(self.dim)
        self.scores = np.full(self.pop_size, np.inf)
        self.best_idx = 0
        self.iteration = 0

    def evaluate(self, pat_xz, pat_yz, goals):
        total_cost = np.zeros(self.pop_size)
        
        def calc_cut_cost(pat, enable, angle, width, mask_val, bw_def, min_gain, mask_w):
            if not enable: return 0
            peak_vals = np.max(pat, axis=1)
            peak_indices = np.argmax(pat, axis=1)
            peak_angles = self.sys.angles[peak_indices]
            
            err_peak = (peak_angles - angle)**2 * 50
            
            err_gain = np.zeros(self.pop_size)
            low_gain = peak_vals < min_gain
            err_gain[low_gain] = (min_gain - peak_vals[low_gain])**2 * 500
            
            # Symmetry / Width Cost
            half_w = width / 2
            # Find indices closest to target angles
            idx_l = np.searchsorted(self.sys.angles, angle - half_w)
            idx_r = np.searchsorted(self.sys.angles, angle + half_w)
            # Clip to bounds
            idx_l = np.clip(idx_l, 0, len(self.sys.angles) - 1)
            idx_r = np.clip(idx_r, 0, len(self.sys.angles) - 1)

            row_idx = np.arange(self.pop_size)
            target_level = peak_vals + bw_def
            val_l = pat[row_idx, idx_l]
            val_r = pat[row_idx, idx_r]
            err_sym = ((val_l - target_level)**2 + (val_r - target_level)**2) * 10
            
            # Mask
            mask_start = angle - (width * 0.6)
            mask_end = angle + (width * 0.6)
            mask_indices = (self.sys.angles < mask_start) | (self.sys.angles > mask_end)
            
            pat_masked = pat[:, mask_indices]
            limits = peak_vals[:, None] - mask_val
            violations = np.maximum(0, pat_masked - limits)
            mask_penalties = np.sum(violations**2, axis=1) * mask_w
            
            return err_peak + err_gain + err_sym + mask_penalties

        total_cost += calc_cut_cost(pat_xz, goals['xz_en'], goals['xz_ang'], goals['xz_bw'], 
                                    goals['xz_sll'], goals['bw_def'], goals['min_gain'], goals['mask_w'])
        total_cost += calc_cut_cost(pat_yz, goals['yz_en'], goals['yz_ang'], goals['yz_bw'], 
                                    goals['yz_sll'], goals['bw_def'], goals['min_gain'], goals['mask_w'])
        return total_cost

    def step(self, algo_type, goals):
        self.iteration += 1
        pat_xz, pat_yz = self.sys.calculate_pattern(self.pop)
        scores = self.evaluate(pat_xz, pat_yz, goals)
        
        min_idx = np.argmin(scores)
        if scores[min_idx] < np.min(self.scores): 
            self.best_idx = min_idx
        self.scores = scores
        
        if algo_type == "DE":
            a = np.random.randint(0, self.pop_size, self.pop_size)
            b = np.random.randint(0, self.pop_size, self.pop_size)
            c = np.random.randint(0, self.pop_size, self.pop_size)
            F = 0.5 + 0.3 * np.random.rand(self.pop_size, 1)
            mutant = self.pop[a] + F * (self.pop[b] - self.pop[c])
            mutant = (mutant + 180) % 360 - 180 
            CR = 0.8
            cross = np.random.rand(self.pop_size, self.dim) < CR
            trial = np.where(cross, mutant, self.pop)
            t_xz, t_yz = self.sys.calculate_pattern(trial)
            t_scores = self.evaluate(t_xz, t_yz, goals)
            better = t_scores < scores
            self.pop[better] = trial[better]
            self.scores[better] = t_scores[better]
            
        elif algo_type == "GA":
            sorted_idx = np.argsort(scores)
            elite_count = max(1, int(self.pop_size * 0.1))
            elites = self.pop[sorted_idx[:elite_count]]
            p1 = np.random.randint(0, self.pop_size, self.pop_size)
            p2 = np.random.randint(0, self.pop_size, self.pop_size)
            winners = np.where((scores[p1] < scores[p2])[:, None], self.pop[p1], self.pop[p2])
            mask = np.random.rand(self.pop_size, self.dim) < 0.5
            children = np.where(mask, winners, np.roll(winners, 1, axis=0))
            mut_mask = np.random.rand(self.pop_size, self.dim) < 0.1
            noise = np.random.normal(0, 30, (self.pop_size, self.dim))
            children[mut_mask] += noise[mut_mask]
            children = (children + 180) % 360 - 180
            self.pop = children
            self.pop[:elite_count] = elites
            
        elif algo_type == "Random":
            best_dna = self.pop[np.argmin(scores)]
            self.pop = np.random.uniform(-180, 180, (self.pop_size, self.dim))
            self.pop[0] = best_dna

        return np.min(self.scores)

# ==========================================
# 4. GUI 介面 (Tkinter)
# ==========================================
class AntennaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Python Antenna Synthesizer (Numba + Interactive Plot)")
        self.root.geometry("1400x900")
        
        self.is_running = False
        
        # --- Variables ---
        self.nx_var = tk.IntVar(value=4)
        self.ny_var = tk.IntVar(value=8)
        self.freq_var = tk.DoubleVar(value=28.0)
        self.dist_var = tk.DoubleVar(value=5.35)
        self.gain_var = tk.DoubleVar(value=5.0)
        self.q_var = tk.DoubleVar(value=1.5)
        self.jio_var = tk.BooleanVar(value=False)
        self.sym_var = tk.BooleanVar(value=False)
        
        self.algo_var = tk.StringVar(value="DE")
        self.min_gain_var = tk.DoubleVar(value=10.0)
        self.bw_def_var = tk.DoubleVar(value=-3.0)
        self.mask_w_var = tk.DoubleVar(value=20.0)
        
        self.xz_en = tk.BooleanVar(value=True)
        self.xz_ang = tk.DoubleVar(value=30)
        self.xz_bw = tk.DoubleVar(value=10)
        self.xz_sll = tk.DoubleVar(value=15)
        
        self.yz_en = tk.BooleanVar(value=False)
        self.yz_ang = tk.DoubleVar(value=0)
        self.yz_bw = tk.DoubleVar(value=15)
        self.yz_sll = tk.DoubleVar(value=15)

        self.sys = AntennaSystem(4, 8, 28.0, 5.35, 5.0, 1.5, False, False)
        self.opt = Optimizer(self.sys)
        
        self._setup_ui()
        self._init_plots()
        
        # --- 連結滑鼠事件 ---
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_hover)
        
        self.root.after(20, self.update_loop)

    def _setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10, width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # Config
        lf_arr = ttk.LabelFrame(control_frame, text="Array Config")
        lf_arr.pack(fill=tk.X, pady=5)
        self._add_entry(lf_arr, "Nx:", self.nx_var, self.on_structure_change)
        self._add_entry(lf_arr, "Ny:", self.ny_var, self.on_structure_change)
        self._add_entry(lf_arr, "Freq (GHz):", self.freq_var, self.on_structure_change)
        self._add_entry(lf_arr, "Spacing (mm):", self.dist_var, self.on_structure_change)
        self._add_entry(lf_arr, "Elem Gain:", self.gain_var)
        self._add_entry(lf_arr, "Q Factor:", self.q_var)
        ttk.Checkbutton(lf_arr, text="Project JIO", variable=self.jio_var, command=self.on_structure_change).pack(anchor='w')
        ttk.Checkbutton(lf_arr, text="Symmetry", variable=self.sym_var, command=self.on_structure_change).pack(anchor='w')
        
        # Specs
        self._create_spec_frame(control_frame, "XZ Cut (Phi=0)", self.xz_en, self.xz_ang, self.xz_bw, self.xz_sll)
        self._create_spec_frame(control_frame, "YZ Cut (Phi=90)", self.yz_en, self.yz_ang, self.yz_bw, self.yz_sll)
        
        # Algo
        lf_algo = ttk.LabelFrame(control_frame, text="Algorithm")
        lf_algo.pack(fill=tk.X, pady=5)
        ttk.Combobox(lf_algo, textvariable=self.algo_var, values=["DE", "GA", "Random"]).pack(fill=tk.X)
        self._add_entry(lf_algo, "Min Gain:", self.min_gain_var)
        self._add_entry(lf_algo, "BW Def (dB):", self.bw_def_var)
        self._add_entry(lf_algo, "Mask Weight:", self.mask_w_var)
        
        # Buttons
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        ttk.Button(btn_frame, text="▶ Start", command=self.start).pack(side=tk.LEFT, expand=True)
        ttk.Button(btn_frame, text="⏸ Pause", command=self.stop).pack(side=tk.LEFT, expand=True)
        ttk.Button(btn_frame, text="🔄 Reset", command=self.reset).pack(side=tk.LEFT, expand=True)
        
        self.status_lbl = ttk.Label(control_frame, text="Ready", foreground="blue")
        self.status_lbl.pack(pady=5)
        
        self.phase_text = tk.Text(control_frame, height=10, width=40, font=("Consolas", 8))
        self.phase_text.pack(fill=tk.BOTH, expand=True, pady=5)

        # Plots
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        self.fig.subplots_adjust(hspace=0.3)
        
        # Add Navigation Toolbar (Optional, for zooming/panning)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _create_spec_frame(self, parent, title, en_var, ang, bw, sll):
        lf = ttk.LabelFrame(parent, text=title)
        lf.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(lf, text="Enable", variable=en_var).pack()
        self._add_entry(lf, "Angle:", ang)
        self._add_entry(lf, "Width:", bw)
        self._add_entry(lf, "SLL (dBc):", sll)

    def _add_entry(self, parent, label, var, cmd=None):
        f = ttk.Frame(parent)
        f.pack(fill=tk.X, pady=1)
        ttk.Label(f, text=label, width=12).pack(side=tk.LEFT)
        e = ttk.Entry(f, textvariable=var)
        e.pack(side=tk.RIGHT, expand=True, fill=tk.X)
        if cmd:
            e.bind('<Return>', lambda e: cmd())
            e.bind('<FocusOut>', lambda e: cmd())

    def _init_plots(self):
        # 主線條
        self.line_xz, = self.ax1.plot([], [], 'g-', linewidth=2, label='XZ Pattern')
        self.line_yz, = self.ax2.plot([], [], 'b-', linewidth=2, label='YZ Pattern')
        
        # SLL 紅色虛線
        self.mask_line_xz, = self.ax1.plot([], [], 'r--', linewidth=1, label='SLL Mask')
        self.mask_line_yz, = self.ax2.plot([], [], 'r--', linewidth=1, label='SLL Mask')
        
        # --- 新增：紫色虛線 (Beamwidth Target) ---
        # 使用 'm--' (magenta dashed)
        self.bw_line_xz, = self.ax1.plot([], [], 'm--', linewidth=1.5, label='Target BW Level')
        self.bw_line_yz, = self.ax2.plot([], [], 'm--', linewidth=1.5, label='Target BW Level')
        
        # Target Box (Rectangles)
        self.rect_xz = patches.Rectangle((0, -60), 1, 100, linewidth=0, facecolor='green', alpha=0.1)
        self.rect_yz = patches.Rectangle((0, -60), 1, 100, linewidth=0, facecolor='blue', alpha=0.1)
        self.ax1.add_patch(self.rect_xz)
        self.ax2.add_patch(self.rect_yz)
        
        # --- 新增：滑鼠懸停註釋 (Annotations) ---
        # 預設不可見 (visible=False)
        bbox_args = dict(boxstyle="round", fc="0.9", alpha=0.8)
        self.annot_xz = self.ax1.annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points", bbox=bbox_args, visible=False)
        self.annot_yz = self.ax2.annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points", bbox=bbox_args, visible=False)

        for ax, title in zip([self.ax1, self.ax2], ["XZ Cut (Phi=0)", "YZ Cut (Phi=90)"]):
            ax.set_title(title)
            ax.set_xlim(-90, 90)
            ax.set_ylim(-60, 30) # 稍微加大 Y 軸範圍
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.set_xlabel("Angle (deg)")
            ax.set_ylabel("Gain (dBi)")
            # 顯示圖例
            ax.legend(loc='upper right', fontsize='small')

    # --- 新增：滑鼠懸停事件處理 ---
    def on_mouse_hover(self, event):
        # 如果滑鼠不在任何軸內，隱藏所有註釋並重繪
        if event.inaxes not in [self.ax1, self.ax2]:
            if self.annot_xz.get_visible() or self.annot_yz.get_visible():
                self.annot_xz.set_visible(False)
                self.annot_yz.set_visible(False)
                self.canvas.draw_idle() # 使用 draw_idle 避免卡頓
            return

        # 判斷在哪個軸
        if event.inaxes == self.ax1:
            ax = self.ax1
            line = self.line_xz
            annot = self.annot_xz
            other_annot = self.annot_yz
        else:
            ax = self.ax2
            line = self.line_yz
            annot = self.annot_yz
            other_annot = self.annot_xz
            
        # 隱藏另一個軸的註釋
        other_annot.set_visible(False)

        # 獲取滑鼠位置
        x_mouse, y_mouse = event.xdata, event.ydata
        
        # 找到最接近滑鼠 X 的數據點索引
        # self.sys.angles 是已排序的陣列，使用 searchsorted 最快
        idx = np.searchsorted(self.sys.angles, x_mouse)
        # 邊界檢查
        if idx >= len(self.sys.angles): idx = len(self.sys.angles) - 1
        if idx > 0 and (np.abs(x_mouse - self.sys.angles[idx-1]) < np.abs(x_mouse - self.sys.angles[idx])):
             idx -= 1

        # 獲取該點的精確數據
        x_data = self.sys.angles[idx]
        y_data = line.get_ydata()[idx]

        # 更新註釋位置和文字
        annot.xy = (x_data, y_data)
        text = f"Ang: {x_data:.1f}°\nGain: {y_data:.2f} dBi"
        annot.set_text(text)
        annot.set_visible(True)
        
        # 請求閒置時重繪 (比 draw() 更高效)
        self.canvas.draw_idle()

    def on_structure_change(self, *args):
        try:
            self.sys = AntennaSystem(
                self.nx_var.get(), self.ny_var.get(),
                self.freq_var.get(), self.dist_var.get(),
                self.gain_var.get(), self.q_var.get(),
                self.jio_var.get(), self.sym_var.get()
            )
            self.opt = Optimizer(self.sys)
            self.stop()
            self.update_plots_once()
            self.status_lbl.config(text="Structure Changed -> Reset", foreground="red")
        except: pass

    def start(self): self.is_running = True
    def stop(self): self.is_running = False
    def reset(self): 
        self.opt.reset()
        self.stop()
        self.update_plots_once()

    def update_loop(self):
        if self.is_running:
            try:
                self.sys.elem_gain = self.gain_var.get()
                self.sys.q = self.q_var.get()
                rads = self.sys.rads
                self.sys.ep_xz = (np.maximum(1e-6, np.cos(rads)) ** self.sys.q).astype(np.float64)
                self.sys.ep_yz = (np.maximum(1e-6, np.cos(rads)) ** self.sys.q).astype(np.float64)

                goals = {
                    'xz_en': self.xz_en.get(), 'xz_ang': self.xz_ang.get(), 'xz_bw': self.xz_bw.get(), 'xz_sll': self.xz_sll.get(),
                    'yz_en': self.yz_en.get(), 'yz_ang': self.yz_ang.get(), 'yz_bw': self.yz_bw.get(), 'yz_sll': self.yz_sll.get(),
                    'bw_def': self.bw_def_var.get(), 'min_gain': self.min_gain_var.get(), 'mask_w': self.mask_w_var.get()
                }

                best_cost = 0
                for _ in range(20): 
                    best_cost = self.opt.step(self.algo_var.get(), goals)

                self.update_plots_once(best_cost)
            except Exception as e:
                print(e)
                self.is_running = False

        self.root.after(30, self.update_loop)

    def update_plots_once(self, cost=0.0):
        best_idx = np.argmin(self.opt.scores)
        pat_xz, pat_yz = self.sys.calculate_pattern(self.opt.pop[best_idx:best_idx+1])
        pat_xz, pat_yz = pat_xz.flatten(), pat_yz.flatten()
        
        peak_xz, peak_yz = np.max(pat_xz), np.max(pat_yz)
        floor = max(peak_xz, peak_yz) - 60
        
        # Visual Clamp
        pat_xz_vis = np.maximum(pat_xz, floor)
        pat_yz_vis = np.maximum(pat_yz, floor)
        
        self.line_xz.set_data(self.sys.angles, pat_xz_vis)
        self.line_yz.set_data(self.sys.angles, pat_yz_vis)
        
        bw_def = self.bw_def_var.get()
        
        # Update Visuals XZ
        if self.xz_en.get():
            g, w, sll = self.xz_ang.get(), self.xz_bw.get(), self.xz_sll.get()
            self.rect_xz.set_bounds(g - w/2, floor, w, 100)
            mask_val = peak_xz - sll
            self.mask_line_xz.set_data([-90, 90], [mask_val, mask_val])
            
            # --- 更新 XZ 紫色虛線 (BW Level) ---
            bw_level = peak_xz + bw_def
            # 只在目標寬度範圍內顯示
            self.bw_line_xz.set_data([g - w/2, g + w/2], [bw_level, bw_level])
            
            self.ax1.set_ylim(floor, peak_xz + 10)
        else:
            # 如果沒啟用，把線移到看不見的地方
            self.bw_line_xz.set_data([], [])

        # Update Visuals YZ
        if self.yz_en.get():
            g, w, sll = self.yz_ang.get(), self.yz_bw.get(), self.yz_sll.get()
            self.rect_yz.set_bounds(g - w/2, floor, w, 100)
            mask_val = peak_yz - sll
            self.mask_line_yz.set_data([-90, 90], [mask_val, mask_val])

            # --- 更新 YZ 紫色虛線 (BW Level) ---
            bw_level = peak_yz + bw_def
            self.bw_line_yz.set_data([g - w/2, g + w/2], [bw_level, bw_level])
            
            self.ax2.set_ylim(floor, peak_yz + 10)
        else:
            self.bw_line_yz.set_data([], [])

        # Redraw Canvas
        self.canvas.draw()
        self.status_lbl.config(text=f"Iter: {self.opt.iteration} | Cost: {cost:.2f}")
        
        # Update Matrix Text
        if self.opt.iteration % 10 == 0 or not self.is_running:
            best_genes = self.opt.pop[best_idx]
            phases = best_genes[self.sys.map_idx].reshape(self.sys.Nx, self.sys.Ny)
            txt = "   " + " ".join([f"Y{i+1:02d}" for i in range(self.sys.Ny)]) + "\n"
            for r in range(self.sys.Nx):
                txt += f"X{r+1:02d} " + " ".join([f"{val:4.0f}" for val in phases[r]]) + "\n"
            self.phase_text.delete(1.0, tk.END)
            self.phase_text.insert(tk.END, txt)

if __name__ == "__main__":
    root = tk.Tk()
    app = AntennaApp(root)
    root.mainloop()