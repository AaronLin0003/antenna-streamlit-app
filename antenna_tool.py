import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
from scipy.optimize import minimize # 引入 SciPy 強力優化庫

# --- 1. 頁面設定 ---
st.set_page_config(page_title="PyAntenna Pro", layout="wide", page_icon="📡")

st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 0rem;}
        div[data-testid="metric-container"] {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            padding: 10px;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# --- 2. 物理核心 ---
class AntennaSystem:
    def __init__(self, Nx, Ny, freq, spacing, elem_gain, q, is_jio, is_sym):
        self.update_params(Nx, Ny, freq, spacing, elem_gain, q, is_jio, is_sym)

    def update_params(self, Nx, Ny, freq, spacing, elem_gain, q, is_jio, is_sym):
        self.Nx = Nx
        self.Ny = Ny
        self.elem_gain = elem_gain
        self.q = q
        
        lambda_val = 299.792 / freq
        d_lambda = spacing / lambda_val
        k = 2 * np.pi
        
        self.map_idx, self.num_genes = self._create_mapping(Nx, Ny, is_jio, is_sym)
        self.total_elements = Nx * Ny
        
        self.angles = np.arange(-90, 90.5, 0.5)
        rads = np.deg2rad(self.angles)
        
        x = np.arange(Nx)
        y = np.arange(Ny)
        X, Y = np.meshgrid(x, y, indexing='ij') 
        self.pos_x = X.flatten()
        self.pos_y = Y.flatten()
        
        u_xz = np.sin(rads)
        phase_xz = k * d_lambda * np.outer(self.pos_x, u_xz) 
        self.sv_xz = np.exp(1j * phase_xz)
        self.ep_xz = np.maximum(1e-6, np.cos(rads)) ** q 
        
        v_yz = np.sin(rads)
        phase_yz = k * d_lambda * np.outer(self.pos_y, v_yz)
        self.sv_yz = np.exp(1j * phase_yz)
        self.ep_yz = np.maximum(1e-6, np.cos(rads)) ** q

    def _create_mapping(self, Nx, Ny, is_jio, is_sym):
        mapping = np.zeros((Nx, Ny), dtype=int)
        dim_nx = int(np.ceil(Nx / 2)) if is_sym else Nx
        dim_ny = int(np.ceil(Ny / 2)) if is_sym else Ny
        if is_jio: dim_ny = int(np.ceil(dim_ny / 2))
            
        counter = 0
        gene_indices = np.zeros((dim_nx, dim_ny), dtype=int)
        for i in range(dim_nx):
            for j in range(dim_ny):
                gene_indices[i, j] = counter
                counter += 1
                
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
        # 支援單個體或群體
        if population_genes.ndim == 1:
            population_genes = population_genes.reshape(1, -1)
            
        phases = population_genes[:, self.map_idx]
        weights = np.exp(1j * np.deg2rad(phases))
        af_xz = np.abs(weights @ self.sv_xz)
        af_yz = np.abs(weights @ self.sv_yz)
        
        def to_dbi(af, ep):
            af = np.maximum(af, 1e-9)
            val = af * ep 
            db = 20 * np.log10(val) - 10 * np.log10(self.total_elements) + self.elem_gain
            return db

        pat_xz = to_dbi(af_xz, self.ep_xz)
        pat_yz = to_dbi(af_yz, self.ep_yz)
        return pat_xz, pat_yz

# --- 3. 優化引擎 (含 PSO 與 Scipy Polish) ---
class Optimizer:
    def __init__(self, sys_model: AntennaSystem, pop_size=50):
        self.sys = sys_model
        self.pop_size = pop_size
        self.dim = sys_model.num_genes
        self.reset() 

    def reset(self):
        # 初始化 population
        self.pop = np.random.uniform(-180, 180, (self.pop_size, self.dim))
        self.pop[0] = np.zeros(self.dim)
        self.scores = np.full(self.pop_size, np.inf)
        self.best_idx = 0
        self.iteration = 0
        
        # PSO 專用變數
        self.velocities = np.zeros_like(self.pop)
        self.pbest_pos = np.copy(self.pop)
        self.pbest_score = np.full(self.pop_size, np.inf)
        self.gbest_pos = np.zeros(self.dim)
        self.gbest_score = np.inf

    def evaluate(self, pat_xz, pat_yz, goals):
        total_cost = np.zeros(pat_xz.shape[0])
        
        def calc_cut_cost(pat, enable, angle, width, mask_val, bw_def, min_gain, mask_w):
            if not enable: return 0
            
            peak_vals = np.max(pat, axis=1)
            peak_indices = np.argmax(pat, axis=1)
            peak_angles = self.sys.angles[peak_indices]
            
            err_peak = (peak_angles - angle)**2 * 50
            
            err_gain = np.zeros(len(pat))
            low_gain = peak_vals < min_gain
            err_gain[low_gain] = (min_gain - peak_vals[low_gain])**2 * 500
            
            target_level = peak_vals + bw_def
            half_w = width / 2
            
            idx_l = np.searchsorted(self.sys.angles, angle - half_w)
            idx_r = np.searchsorted(self.sys.angles, angle + half_w)
            idx_l = np.clip(idx_l, 0, len(self.sys.angles)-1)
            idx_r = np.clip(idx_r, 0, len(self.sys.angles)-1)
            
            row_idx = np.arange(len(pat))
            val_l = pat[row_idx, idx_l]
            val_r = pat[row_idx, idx_r]
            
            err_sym = ((val_l - target_level)**2 + (val_r - target_level)**2) * 10
            
            mask_start = angle - (width * 0.6)
            mask_end = angle + (width * 0.6)
            mask_zone_indices = (self.sys.angles < mask_start) | (self.sys.angles > mask_end)
            
            limits = peak_vals[:, None] - mask_val
            pat_masked = pat[:, mask_zone_indices]
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
        
        # 1. 評估當前 Cost
        pat_xz, pat_yz = self.sys.calculate_pattern(self.pop)
        scores = self.evaluate(pat_xz, pat_yz, goals)
        
        # 更新全域最佳
        min_idx = np.argmin(scores)
        if scores[min_idx] < self.gbest_score:
            self.gbest_score = scores[min_idx]
            self.gbest_pos = self.pop[min_idx].copy()
            
        self.scores = scores
        self.best_idx = min_idx # 用於顯示目前最佳
        
        # 2. 演算法分支
        if algo_type == "PSO (粒子群)":
            # 更新 PBest
            improved = scores < self.pbest_score
            self.pbest_pos[improved] = self.pop[improved]
            self.pbest_score[improved] = scores[improved]
            
            # PSO 參數
            w = 0.7  # 慣性
            c1 = 1.5 # 認知 (自我)
            c2 = 1.5 # 社會 (群體)
            
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            
            # 速度更新
            self.velocities = (w * self.velocities + 
                               c1 * r1 * (self.pbest_pos - self.pop) + 
                               c2 * r2 * (self.gbest_pos - self.pop))
            
            # 位置更新
            self.pop += self.velocities
            # 邊界處理 (Wrap -180 ~ 180)
            self.pop = (self.pop + 180) % 360 - 180
            
        elif algo_type == "DE (差分進化)":
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
            
        elif algo_type == "GA (基因算法)":
            # 略微簡化以提升速度
            p1 = np.random.randint(0, self.pop_size, self.pop_size)
            p2 = np.random.randint(0, self.pop_size, self.pop_size)
            winners = np.where((scores[p1] < scores[p2])[:, None], self.pop[p1], self.pop[p2])
            mask = np.random.rand(self.pop_size, self.dim) < 0.5
            children = np.where(mask, winners, np.roll(winners, 1, axis=0))
            noise = np.random.normal(0, 10, (self.pop_size, self.dim))
            children += noise # 輕微變異
            children = (children + 180) % 360 - 180
            self.pop = children
            self.pop[0] = self.gbest_pos # Elitism

        return self.gbest_score

    def polish(self, goals):
        """使用梯度下降法 (Gradient Descent) 進行最終微調"""
        # 定義給 scipy 用的單一目標函數
        def objective(x):
            # x is shape (dim,)
            pop_wrapper = x.reshape(1, -1)
            p_xz, p_yz = self.sys.calculate_pattern(pop_wrapper)
            cost = self.evaluate(p_xz, p_yz, goals)
            return cost[0]

        # 從目前全域最佳解開始
        start_pos = self.gbest_pos
        
        # 執行 L-BFGS-B 優化 (Python 的強力數學優化器)
        res = minimize(objective, start_pos, method='L-BFGS-B', bounds=[(-180, 180)]*self.dim, options={'maxiter': 50})
        
        # 更新結果
        self.pop[0] = res.x
        self.gbest_pos = res.x
        self.gbest_score = res.fun
        return res.fun

# --- 4. UI Layout & Session State ---

if 'opt' not in st.session_state:
    st.session_state.opt = None
if 'running' not in st.session_state:
    st.session_state.running = False

# Sidebar
with st.sidebar:
    st.header("🛠️ 陣列配置")
    c1, c2 = st.columns(2)
    Nx = c1.number_input("Nx (X軸)", 1, 64, 4)
    Ny = c2.number_input("Ny (Y軸)", 1, 64, 8)
    freq = c1.number_input("Freq (GHz)", 1.0, 100.0, 28.0)
    space = c2.number_input("Spacing (mm)", 1.0, 100.0, 5.35)
    gain = c1.number_input("Elem Gain", -50.0, 50.0, 5.0)
    q = c2.number_input("Q Factor", 0.0, 5.0, 1.5)
    jio = st.checkbox("Project JIO", False)
    sym = st.checkbox("Symmetry", False)
    
    config_id = f"{Nx}_{Ny}_{jio}_{sym}"
    if 'last_cfg' not in st.session_state: st.session_state.last_cfg = config_id
    if st.session_state.last_cfg != config_id:
        st.session_state.opt = None 
        st.session_state.running = False
        st.session_state.last_cfg = config_id
        st.toast("陣列結構變更，已重置系統", icon="🔄")

    st.divider()
    st.header("🎯 目標設定")
    xz_en = st.checkbox("XZ Cut (Phi=0)", True)
    c3, c4 = st.columns(2)
    xz_ang = c3.number_input("XZ Angle", -90, 90, 30)
    xz_bw = c4.number_input("XZ Width", 1, 90, 10)
    xz_sll = st.number_input("XZ SLL (dBc)", 0, 60, 15)
    
    yz_en = st.checkbox("YZ Cut (Phi=90)", False)
    c5, c6 = st.columns(2)
    yz_ang = c5.number_input("YZ Angle", -90, 90, 0)
    yz_bw = c6.number_input("YZ Width", 1, 90, 15)
    yz_sll = st.number_input("YZ SLL (dBc)", 0, 60, 15)
    
    st.divider()
    st.header("⚙️ 演算法控制")
    # 新增 PSO 選項
    algo = st.selectbox("Algorithm", ["PSO (粒子群)", "DE (差分進化)", "GA (基因算法)"])
    min_gain_val = st.number_input("Min Gain (dBi)", -20.0, 50.0, 10.0)
    bw_def = st.number_input("BW Def (dB)", -20.0, -0.1, -3.0)
    mask_w = st.slider("Mask Penalty", 1, 500, 50)
    
    col_a, col_b = st.columns(2)
    if col_a.button("▶ 開始", type="primary"):
        st.session_state.running = True
        st.rerun()
    if col_b.button("⏸ 暫停"):
        st.session_state.running = False
        st.rerun()
        
    col_c, col_d = st.columns(2)
    # 新增 Polish 按鈕
    polish_btn = col_c.button("✨ 精修 (Polish)")
    if col_d.button("🔄 重置"):
        if st.session_state.opt: st.session_state.opt.reset() 
        st.session_state.running = False
        st.rerun()

# --- 5. 初始化 ---
if st.session_state.opt is None:
    sys = AntennaSystem(Nx, Ny, freq, space, gain, q, jio, sym)
    st.session_state.opt = Optimizer(sys, pop_size=50)
else:
    st.session_state.opt.sys.update_params(Nx, Ny, freq, space, gain, q, jio, sym)

opt = st.session_state.opt
goals = {
    'xz_en': xz_en, 'xz_ang': xz_ang, 'xz_bw': xz_bw, 'xz_sll': xz_sll,
    'yz_en': yz_en, 'yz_ang': yz_ang, 'yz_bw': yz_bw, 'yz_sll': yz_sll,
    'bw_def': bw_def, 'min_gain': min_gain_val, 'mask_w': mask_w
}

# --- 6. Polish 邏輯 ---
if polish_btn:
    st.session_state.running = False # 暫停自動跑
    with st.spinner("正在進行梯度下降微調 (Gradient Descent Polishing)..."):
        final_cost = opt.polish(goals)
    st.success(f"精修完成！Cost 降至: {final_cost:.6f}")

# --- 7. 靜態容器 ---
metric_spot = st.empty()
col_g1, col_g2 = st.columns(2)
with col_g1: xz_spot = st.empty()
with col_g2: yz_spot = st.empty()
table_spot = st.empty()
status_spot = st.empty()

# --- 8. 更新邏輯 ---
def update_view():
    # 這裡顯示的是 Global Best (gbest)，因為在 PSO 中 gbest 才是最重要的
    best_genes = opt.gbest_pos
    
    # 這裡需要 reshpae 成 (1, dim) 才能餵給 calculate_pattern
    pat_xz, pat_yz = opt.sys.calculate_pattern(best_genes.reshape(1, -1))
    pat_xz = pat_xz.flatten()
    pat_yz = pat_yz.flatten()
    angles = opt.sys.angles
    
    pk_xz = np.max(pat_xz)
    pk_yz = np.max(pat_yz)
    cost = opt.gbest_score
    
    # Metrics
    with metric_spot.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Iterations", opt.iteration)
        c2.metric("Best Cost", f"{cost:.4f}" if not np.isinf(cost) else "0.0")
        c3.metric("XZ Peak", f"{pk_xz:.2f} dBi")
        c4.metric("YZ Peak", f"{pk_yz:.2f} dBi")
        
    # Charts
    def plot_cut(spot, pat, peak, ang, bw, sll, title, key_id):
        floor = peak - 60
        pat_vis = np.maximum(pat, floor)
        fig = go.Figure()
        
        # Mask & SLL
        fig.add_shape(type="rect", x0=ang-bw/2, x1=ang+bw/2, y0=floor, y1=peak+50, 
                      fillcolor="rgba(46, 204, 113, 0.1)", line_width=0)
        mask_val = peak - sll
        fig.add_shape(type="line", x0=-90, x1=90, y0=mask_val, y1=mask_val, 
                      line=dict(color="red", width=2, dash="dot"))
        
        # Trace
        fig.add_trace(go.Scatter(x=angles, y=pat_vis, mode='lines', line=dict(width=3)))
        
        max_possible_gain = gain + 10*np.log10(Nx*Ny) + 5
        fig.update_layout(
            title=title, xaxis_title="Angle", yaxis_title="dBi",
            yaxis=dict(range=[floor, max_possible_gain]), 
            xaxis=dict(range=[-90, 90]),
            height=350, margin=dict(l=20, r=20, t=40, b=20)
        )
        spot.plotly_chart(fig, use_container_width=True, key=key_id)

    if xz_en: plot_cut(xz_spot, pat_xz, pk_xz, xz_ang, xz_bw, xz_sll, "XZ Cut", "xz_chart")
    if yz_en: plot_cut(yz_spot, pat_yz, pk_yz, yz_ang, yz_bw, yz_sll, "YZ Cut", "yz_chart")
    
    # Table
    phases = best_genes[opt.sys.map_idx].reshape(Nx, Ny)
    df = pd.DataFrame(phases, index=[f"X{i+1}" for i in range(Nx)], columns=[f"Y{i+1}" for i in range(Ny)])
    table_spot.dataframe(df.style.background_gradient(cmap="Oranges", vmin=-180, vmax=180).format("{:.1f}"), use_container_width=True)
    
    status_msg = "Running... 🔥" if st.session_state.running else "Paused ⏸️"
    status_spot.info(f"Status: {status_msg} | Algorithm: {algo}")

# --- 9. 迴圈控制 ---
if st.session_state.running:
    frames_per_run = 10 
    steps_per_frame = 2 
    for f in range(frames_per_run):
        for _ in range(steps_per_frame):
            opt.step(algo, goals)
        update_view()
        time.sleep(0.02)
    st.rerun()
else:
    if opt.iteration == 0:
        # 初始計算一次
        p_xz, p_yz = opt.sys.calculate_pattern(opt.pop[0].reshape(1,-1))
        opt.gbest_score = opt.evaluate(p_xz, p_yz, goals)[0]
        opt.gbest_pos = opt.pop[0]
    update_view()
