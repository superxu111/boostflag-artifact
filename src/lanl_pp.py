#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lanl_pp.py — 图上标记自激励点过程（LANL cyber1 红队流）核心模块 v3
事件流: 红队事件 (t_i, v_i), v=dst_comp; 图: NTLM 计算机诱导图（无向, 14,964 节点）
条件强度: λ(t,v) = μ0·m_h(t)·q(v) + κ·Σ_{i: 0<t−t_i≤W} g(t−t_i)·K(d(v,v_i))·mk_i
  g: Omori (Δt+c)^{-p} | exp(−βΔt)     K: hop0(仅同主机)|hop≤1|hop≤2 指示 | 逐跳衰减 w1,w2 | exp(−γ·hop)
  m_h: 24-bin hour-of-day（经验,固定）| const | fourier2(自由4参数)
  q: uniform | pop | degree（经验,固定）| custom（外部传入, exp12 EB flag）  mk: 无 | src(4类,3自由) | succ(1自由)
自由参数如实计数: 默认 omori+hop1+hod24+pop 无标记 = 4 (μ0,κ,c,p); 变体 3–8 个
索引: V* = 全部事件目标去重(~295); HopPair[|V*|²] 事件目标间 hop(≤3,4=远);
      HopC[|V*|×|C|] 目标到候选集 hop; hopcnt[|V*|×5] 目标到 C 各 hop 计数(Λ/积分用)
IGPE(CSEP): 逐事件 log2(λ_model/ξ_ref), 参考=规范强度(∫=R0·T); Rhoades 2011 T 检验
"""
import numpy as np
from scipy.optimize import minimize
from scipy import stats
from collections import deque

# ---------------- 数据与图 ----------------
def load_graph(uniq_path='/tmp/induced_uniq.txt'):
    adj, nodes = {}, set()
    for l in open(uniq_path):
        p = l.rstrip('\n').split(',')
        s, d = p[3], p[4]
        nodes.add(s); nodes.add(d)
        if s != d:
            adj.setdefault(s, set()).add(d)
            adj.setdefault(d, set()).add(s)
    return adj, sorted(nodes)

def load_events(path='/tmp/redteam.txt'):
    rows = []
    for l in open(path):
        p = l.strip().split(',')
        if len(p) >= 4:
            rows.append((int(p[0]), p[1], p[2], p[3]))  # ts,user,src,dst
    seen = {}
    for ts, u, s, d in sorted(rows):
        seen.setdefault((ts, d), (ts, u, s, d))
    ev = sorted(seen.values(), key=lambda x: x[0])
    return dict(t=np.array([e[0] for e in ev], dtype=np.int64),
                dst=[e[3] for e in ev], src=[e[2] for e in ev], usr=[e[1] for e in ev],
                n=len(ev), raw_rows=len(rows))

def hour_profile(t_train, smooth=0.5):
    h = ((np.asarray(t_train, np.int64) % 86400) // 3600).astype(int)
    cnt = np.bincount(h, minlength=24).astype(float) + smooth
    m = cnt / cnt.mean()
    assert abs(m.mean() - 1.0) < 1e-9
    return m

def hour_coverage_seconds(t0, t1):
    cov = np.zeros(24)
    t = float(t0)
    while t < t1:
        h = int((t % 86400) // 3600)
        nxt = np.floor(t / 3600.0 + 1.0) * 3600.0
        cov[h] += min(nxt, t1) - t
        t = nxt
    return cov

def bfs_depth(adj, source, maxd=3):
    dist = {source: 0}
    q = deque([(source, 0)])
    while q:
        u, dep = q.popleft()
        if dep >= maxd:
            continue
        for v in adj.get(u, ()):
            if v not in dist:
                dist[v] = dep + 1
                q.append((v, dep + 1))
    return dist

# ---------------- 主模型 ----------------
class GraphHawkes:
    def __init__(self, t, dst, src, adj, nodes_all, dst_extra=(),
                 kernel='omori', gk='hop1', bg='hod24', cand='all', W=86400.0,
                 qbg='pop', marks=None, succ=None, cand_list=None, q_custom=None,
                 tau_stage=None):
        # qbg='custom' + q_custom: 外部给定的背景 mark 分布（如 exp12 EB flag 收缩乘子, 仅 train 拟合）
        # kernel='twostage' + tau_stage=(tau1, tau2): 双时间尺度指数和核 g=w·e^{−t/τ1}+(1−w)·e^{−t/τ2}
        #   （exp16 frailty 升级用; τ 档位固定只估权重 w——可辨识性惯例, exp11 A2/V3）
        self.kernel, self.gk, self.bg, self.cand, self.W = kernel, gk, bg, cand, W
        self.qbg, self.marks, self.succ = qbg, marks, succ
        if kernel == 'twostage':
            assert tau_stage is not None and len(tau_stage) == 2 and 0 < tau_stage[0] < tau_stage[1]
            self.tau1, self.tau2 = float(tau_stage[0]), float(tau_stage[1])
        self.adj, self.nodes_all = adj, list(nodes_all)
        self.t = np.asarray(t, dtype=np.float64)
        self.dst, self.src = list(dst), list(src)
        self.n = len(self.t)
        self.t0, self.T = float(self.t.min()), float(self.t.max()) + 1.0
        self.V = sorted(set(self.dst) | set(dst_extra))
        self.vidx = {v: i for i, v in enumerate(self.V)}
        self.nV = len(self.V)
        if cand_list is not None:
            self.C = list(cand_list)
        elif cand == 'all':
            self.C = self.nodes_all
        elif cand == 'active':
            Cset = set(self.V)
            for v in self.V:
                Cset |= self.adj.get(v, set())
            self.C = sorted(Cset)
        else:
            raise ValueError(cand)
        self.cidx = {c: i for i, c in enumerate(self.C)}
        self.nC = len(self.C)
        self.m_h = hour_profile(self.t) if bg == 'hod24' else np.ones(24)
        cnt = np.zeros(self.nC)
        for d in self.dst:
            cnt[self.cidx[d]] += 1
        if qbg == 'uniform':
            q = np.ones(self.nC)
        elif qbg == 'custom':
            q = np.asarray(q_custom, dtype=float).copy()
            assert len(q) == self.nC and np.all(q > 0), "custom q 须与 C 等长且为正"
        elif qbg == 'pop':
            q = cnt + 0.5
        elif qbg == 'degree':
            q = np.array([len(self.adj.get(c, ())) + 1.0 for c in self.C])
        else:
            raise ValueError(qbg)
        self.q = q / q.sum()
        self._build_distances()
        self._layout()
        # 拟合事件矩阵（向量化似然）
        self._prep_fit()

    def _build_distances(self):
        Dm = {v: bfs_depth(self.adj, v, 3) for v in self.V}
        H = np.full((self.nV, self.nV), 4, dtype=np.int8)
        for i, v in enumerate(self.V):
            H[i, i] = 0
            for u, h in Dm[v].items():
                if u in self.vidx:
                    H[i, self.vidx[u]] = h
        self.HopPair = np.minimum(H, H.T)
        self.hopcnt = np.zeros((self.nV, 5))
        self.HopC = np.full((self.nV, self.nC), 4, dtype=np.int8)
        for i, v in enumerate(self.V):
            for u, h in Dm[v].items():
                if u in self.cidx:
                    self.hopcnt[i, min(h, 4)] += 1
                    self.HopC[i, self.cidx[u]] = h
        # Qh[i,h] = sum_{u in C, hop=v_i->u =h} q(u) —— Ktilde 归一化质量（参数无关）
        self.Qh = np.zeros((self.nV, 5))
        for i in range(self.nV):
            for h in range(4):
                self.Qh[i, h] = float(self.q[self.HopC[i] == h].sum())
        assert np.all(self.Qh[:, 0] > 0), "Qh hop0 必须为正"
        assert self.HopC.dtype == np.int8

    def _layout(self):
        names, x0, lb, ub = [], [], [], []
        names.append('log_mu0'); x0.append(np.log(self.n / max(self.T - self.t0, 1.0))); lb.append(-16); ub.append(10)
        if self.kernel == 'omori':
            names += ['log_kappa', 'log_c', 'log_p']
            x0 += [np.log(0.5), np.log(60.0), np.log(1.5)]
            # 地震学标准界: p<=2 (Omori), c in [10,3600]s —— 防 δ 尖峰退化 (Exp2 实证)
            lb += [-16, np.log(10.0), 0.0]; ub += [10, np.log(3600.0), np.log(2.0)]
        elif self.kernel == 'twostage':
            names += ['log_kappa', 'logit_w1']
            x0 += [np.log(0.5), 0.0]                      # w0=sigmoid(0)=0.5 两尺度等权起步
            lb += [-16, -8.0]; ub += [10, 8.0]
        else:
            names += ['log_kappa', 'log_beta']
            x0 += [np.log(0.5), np.log(1 / 300.0)]
            lb += [-16, -10]; ub += [10, 7]
        if self.gk == 'decay':
            names += ['log_w1', 'log_w2']; x0 += [np.log(0.3), np.log(0.1)]; lb += [-16, -16]; ub += [5, 5]
        elif self.gk == 'exp_hop':
            names += ['log_gamma']; x0 += [np.log(1.0)]; lb += [-7]; ub += [7]
        if self.bg == 'fourier2':
            names += ['a1', 'b1', 'a2', 'b2']; x0 += [0.3, 0.0, 0.1, 0.0]; lb += [-5] * 4; ub += [5] * 4
        if self.marks == 'src':
            names += ['mk1', 'mk2', 'mk3']; x0 += [0.0, 0.0, 0.0]; lb += [-8] * 3; ub += [8] * 3
        elif self.marks == 'succ':
            names += ['mk_succ']; x0 += [0.0]; lb += [-8]; ub += [8]
        self.pnames = names
        self.x0 = np.array(x0); self.lb = np.array(lb); self.ub = np.array(ub)
        self.n_par = len(names)
        self.idx = {nm: i for i, nm in enumerate(names)}

    def _prep_fit(self):
        t = self.t
        n = self.n
        self.f_dt = t[None, :] - t[:, None]          # dt[i,j] = t_j − t_i
        self.f_mk = (self.f_dt > 0) & (self.f_dt <= self.W)   # 可触发掩码
        self.f_si = np.array([self.vidx[d] for d in self.dst], dtype=int)
        qv = np.array([self.q[self.cidx[d]] for d in self.dst])
        self.f_bgq = qv                               # 逐事件背景 q 值

    # ---- 核 ----
    def g(self, th, dt):
        dt = np.maximum(np.asarray(dt, float), 1e-12)
        if self.kernel == 'omori':
            c = np.exp(th[self.idx['log_c']]); p = np.exp(th[self.idx['log_p']])
            return (dt + c) ** (-p)
        if self.kernel == 'twostage':
            w = 1.0 / (1.0 + np.exp(-th[self.idx['logit_w1']]))
            return w * np.exp(-dt / self.tau1) + (1.0 - w) * np.exp(-dt / self.tau2)
        beta = np.exp(th[self.idx['log_beta']])
        return np.exp(-beta * dt)

    def G_W(self, th, x):
        x = np.minimum(np.maximum(np.asarray(x, float), 0.0), self.W)
        if self.kernel == 'omori':
            c = np.exp(th[self.idx['log_c']]); p = np.exp(th[self.idx['log_p']])
            if abs(p - 1.0) < 1e-10:
                return np.log((x + c) / c)
            return ((x + c) ** (1 - p) - c ** (1 - p)) / (1 - p)
        if self.kernel == 'twostage':
            w = 1.0 / (1.0 + np.exp(-th[self.idx['logit_w1']]))
            return w * self.tau1 * (1.0 - np.exp(-x / self.tau1)) \
                + (1.0 - w) * self.tau2 * (1.0 - np.exp(-x / self.tau2))
        beta = np.exp(th[self.idx['log_beta']])
        return (1.0 - np.exp(-beta * x)) / max(beta, 1e-300)

    def K_at_hop(self, th, h):
        h = int(h)
        if self.gk == 'hop0':
            # hop0-only: 仅同主机自触发（B4 无图核消融 / boost 机制模型; 双审 C1 修复）
            return 1.0 if h == 0 else 0.0
        if self.gk == 'hop1':
            return 1.0 if h <= 1 else 0.0
        if self.gk == 'hop2':
            return 1.0 if h <= 2 else 0.0
        if self.gk == 'decay':
            w1 = np.exp(th[self.idx['log_w1']]); w2 = np.exp(th[self.idx['log_w2']])
            return {0: 1.0, 1: float(w1), 2: float(w2)}.get(h, 0.0)
        if self.gk == 'exp_hop':
            gamma = np.exp(th[self.idx['log_gamma']])
            return float(np.exp(-gamma * h)) if h < 4 else 0.0
        raise ValueError(self.gk)

    def cbar(self, th, i):
        vals = [self.K_at_hop(th, h) * self.Qh[i, h] for h in range(4)]
        return max(float(sum(vals)), 1e-300)

    def K_from_hop(self, th, H):
        """H: hop 整数数组(任意形状) → K 值"""
        H = np.asarray(H)
        if self.gk == 'hop0':
            return (H == 0).astype(float)
        if self.gk == 'hop1':
            return (H <= 1).astype(float)
        if self.gk == 'hop2':
            return (H <= 2).astype(float)
        if self.gk == 'decay':
            w1 = np.exp(th[self.idx['log_w1']]); w2 = np.exp(th[self.idx['log_w2']])
            out = np.zeros(H.shape)
            out[H == 0] = 1.0; out[H == 1] = w1; out[H == 2] = w2
            return out
        if self.gk == 'exp_hop':
            gamma = np.exp(th[self.idx['log_gamma']])
            out = np.exp(-gamma * H.astype(float))
            out[H >= 4] = 0.0
            return out
        raise ValueError(self.gk)

    def S_row(self, th, i):
        hc = self.hopcnt[i]
        if self.gk == 'hop0':
            return hc[0]
        if self.gk == 'hop1':
            return hc[0] + hc[1]
        if self.gk == 'hop2':
            return hc[0] + hc[1] + hc[2]
        if self.gk == 'decay':
            w1 = np.exp(th[self.idx['log_w1']]); w2 = np.exp(th[self.idx['log_w2']])
            return hc[0] + w1 * hc[1] + w2 * hc[2]
        if self.gk == 'exp_hop':
            gamma = np.exp(th[self.idx['log_gamma']])
            return float(np.sum(hc[:4] * np.exp(-gamma * np.arange(4))))
        raise ValueError(self.gk)

    def mark_mult(self, th, i):
        if self.marks == 'src':
            k = sorted(set(self.src)).index(self.src[i])
            mult = 1.0
            for j in range(3):
                if k == j + 1:
                    mult *= np.exp(th[self.idx['mk%d' % (j + 1)]])
            return mult
        if self.marks == 'succ':
            if self.succ is None:
                return 1.0
            return float(np.exp(th[self.idx['mk_succ']])) if self.succ[i] else 1.0
        return 1.0

    def mark_vec(self, th, idx_list):
        if not self.marks:
            return None
        return np.array([self.mark_mult(th, i) for i in idx_list])

    def mu_base(self, th, t):
        if self.bg == 'fourier2':
            ph = 2 * np.pi * (t % 86400) / 86400.0
            base = (1.0 + th[self.idx['a1']] * np.cos(ph) + th[self.idx['b1']] * np.sin(ph)
                        + th[self.idx['a2']] * np.cos(2 * ph) + th[self.idx['b2']] * np.sin(2 * ph))
            return max(float(base), 1e-8)
        return float(self.m_h[int((t % 86400) // 3600)])

    def mu_bg_integral(self, th, a, b):
        """∫_a^b m_base(t) dt（hod24: 解析; fourier2: 600s 网格数值）"""
        cov = hour_coverage_seconds(a, b)
        if self.bg == 'fourier2':
            grid = np.arange(a, b, 600.0)
            base = np.array([self.mu_base(th, x) for x in grid])
            return float(np.trapz(base, grid))
        return float(np.sum(self.m_h * cov))

    # ---- 似然（向量化） ----
    def loglik(self, th, t=None, dst=None, si=None, T_end=None, pre=None):
        t = self.t if t is None else np.asarray(t, float)
        si = self.f_si if si is None else np.asarray(si, int)
        n = len(t)
        T_end = self.T if T_end is None else float(T_end)
        if pre is None:
            dtm = t[None, :] - t[:, None]
            msk = (dtm > 0) & (dtm <= self.W)
        else:
            dtm, msk = pre
        qv = np.array([self.q[self.cidx[d]] for d in (self.dst if dst is None else dst)])
        kap = np.exp(th[self.idx['log_kappa']])
        mv = self.mark_vec(th, range(n))
        # 逐事件 λ(t_j, v_j): bg + κ Σ_i g·K·mk
        Kp = self.K_from_hop(th, self.HopPair[si][:, si])   # (n,n)
        G = self.g(th, np.maximum(dtm, 1e-12))
        cbar_i = np.array([self.cbar(th, i) for i in si])
        A = kap * G * Kp * (qv[None, :] / cbar_i[:, None])
        if mv is not None:
            A = A * mv[:, None]
        A[~msk] = 0.0
        mu_ev = np.exp(th[self.idx['log_mu0']]) * np.array([self.mu_base(th, x) for x in t])
        lam = mu_ev * qv + A.sum(axis=0)
        if not np.all(np.isfinite(lam)) or np.any(lam <= 0):
            return -1e18
        ll = float(np.sum(np.log(lam)))
        int_bg = np.exp(th[self.idx['log_mu0']]) * self.mu_bg_integral(th, t[0], T_end)
        GWv = np.asarray(self.G_W(th, T_end - t))
        int_tr = kap * float(np.sum(GWv * (mv if mv is not None else 1.0)))
        val = ll - int_bg - int_tr
        return val if np.isfinite(val) else -1e18

    def fit(self, n_restarts=8, seed=1, verbose=False, t_end=None):
        rng = np.random.default_rng(seed)
        pre = (self.f_dt, self.f_mk)
        T_end = self.T if t_end is None else float(t_end)
        best_f, best_x = -1e18, self.x0.copy()
        hist = []
        for r in range(n_restarts):
            x0 = self.x0 if r == 0 else np.clip(self.x0 + rng.normal(0, 0.5, len(self.x0)),
                                                self.lb + 1e-6, self.ub - 1e-6)
            try:
                res = minimize(lambda x: -self.loglik(x, pre=pre, T_end=T_end), x0, method='L-BFGS-B',
                               bounds=list(zip(self.lb, self.ub)),
                               options=dict(maxiter=500, ftol=1e-12, gtol=1e-8))
                f = -res.fun
            except Exception as e:
                hist.append((r, 'ERR', str(e)[:80])); continue
            hist.append((r, round(float(f), 3), int(res.nit)))
            if verbose:
                print(f"  restart {r}: ll={f:.3f}", flush=True)
            if f > best_f:
                best_f, best_x = f, res.x.copy()
        self.ll_max, self.th = float(best_f), best_x.copy()
        self.fit_hist = hist
        assert np.isfinite(self.ll_max) and self.ll_max > -1e17, f"MLE 发散 ll={self.ll_max}"
        return self.th

    def params_dict(self):
        d = {}
        for nm in self.pnames:
            v = float(self.th[self.idx[nm]])
            key = nm[4:] if nm.startswith('log_') else nm
            d[key] = float(np.exp(v)) if nm.startswith('log_') else v
        return d

    # ---- 强度向量（评分/排名/模拟） ----
    def Kmat(self, th):
        return self.K_from_hop(th, self.HopC)

    def lambda_vec_C(self, th, t, t_hist, si_hist, mk_hist=None, Kmat=None):
        """λ(t,·) |C| 向量（Ktilde=K·q/cbar 语义）"""
        lam = np.exp(th[self.idx['log_mu0']]) * self.mu_base(th, t) * self.q
        n_h = len(t_hist)
        if n_h:
            dt = t - np.asarray(t_hist, float)
            m = (dt > 0) & (dt <= self.W)
            if m.any():
                si_a = np.asarray(si_hist)[m]
                gk = self.g(th, dt[m])
                mv = mk_hist[m] if mk_hist is not None else 1.0
                acc = np.zeros(self.nC)
                for k, i in enumerate(si_a):
                    w = gk[k] * (mv[k] if mk_hist is not None else 1.0)
                    acc += (w / self.cbar(th, i)) * (self.K_from_hop(th, self.HopC[i]) * self.q)
                lam += np.exp(th[self.idx['log_kappa']]) * acc
        return lam

    def Lambda_total(self, th, t, t_hist, si_hist, mk_hist=None, Kmat=None):
        return float(self.lambda_vec_C(th, t, t_hist, si_hist, mk_hist, Kmat).sum())

    # ---- 模拟（Ogata thinning; 目标宇宙=V*, 记录于文档） ----
    def simulate(self, th, T_end, seed=0, t_start=None, dst_sample=None):
        rng = np.random.default_rng(seed)
        t_start = self.t0 if t_start is None else float(t_start)
        kap = np.exp(th[self.idx['log_kappa']])
        qV = np.array([self.q[self.cidx[d]] for d in self.V]); qV = qV / qV.sum()
        # 后代分布: Poff[i,:] ∝ K(hop_i,u)·q_u（行和=1）; 每父事件触发率精确=κ·g
        Kp = self.K_from_hop(th, self.HopPair)            # |V*|×|V*|
        cbar_all = np.array([self.cbar(th, i) for i in range(self.nV)])
        Poff = Kp * qV[None, :] / cbar_all[:, None]
        rowsum = Poff.sum(axis=1)
        Poff[rowsum <= 0] = qV[None, :]
        Poff = Poff / Poff.sum(axis=1, keepdims=True)
        mu0 = np.exp(th[self.idx['log_mu0']])
        ev_t, ev_vi = [], []   # vi: V* 索引
        t = t_start
        while t < T_end:
            # 界含 age=0 的最新事件（右极限 g(0+)=c^{-p}）——否则事件后高强度窗欠采样
            lam_h = 0.0
            for ti, vi in zip(ev_t, ev_vi):
                dt = t - ti
                if 0 <= dt <= self.W:
                    lam_h += float(self.g(th, max(dt, 1e-12)))
            base = mu0 * self.mu_base(th, t)
            if base + kap * lam_h <= 0:
                t += 60.0; continue
            # g 单调递减 → [t, t+w] 内 λ(s) ≤ base(t)+kap·lam_h(t)（含 age=0 右极限）, 有效上界
            M = base + kap * lam_h + 1e-15
            w = rng.exponential(1.0 / M)
            t2 = t + w
            if t2 >= T_end:
                break
            base2 = mu0 * self.mu_base(th, t2)
            lam2 = base2
            for ti, vi in zip(ev_t, ev_vi):
                dt = t2 - ti
                if 0 < dt <= self.W:
                    lam2 += kap * float(self.g(th, dt))
            if rng.random() * M <= lam2:
                comps = [base2]; parents = []
                for k, (ti, vi) in enumerate(zip(ev_t, ev_vi)):
                    dt = t2 - ti
                    if 0 < dt <= self.W:
                        comps.append(kap * float(self.g(th, dt)))
                        parents.append(k)
                comps = np.array(comps)
                k = int(rng.choice(len(comps), p=comps / comps.sum()))
                if k == 0:
                    vi = int(rng.choice(len(qV), p=qV))
                else:
                    vi = int(rng.choice(self.nV, p=Poff[ev_vi[parents[k - 1]]]))
                ev_t.append(t2); ev_vi.append(vi)
            t = t2
        return np.array(ev_t), [self.V[i] for i in ev_vi]

    # ---- 触发分解（EM/Zhuang 概率） ----
    def trigger_probs(self, th, t=None, dst=None, pre=None):
        t = self.t if t is None else np.asarray(t, float)
        si = self.f_si if dst is None else np.array([self.vidx[d] for d in dst], int)
        n = len(t)
        if pre is None:
            dtm = t[None, :] - t[:, None]
            msk = (dtm > 0) & (dtm <= self.W)
        else:
            dtm, msk = pre
        kap = np.exp(th[self.idx['log_kappa']])
        mv = self.mark_vec(th, range(n))
        Kp = self.K_from_hop(th, self.HopPair[si][:, si])
        G = self.g(th, np.maximum(dtm, 1e-12))
        cbar_i = np.array([self.cbar(th, i) for i in si])
        qv = np.array([self.q[self.cidx[d]] for d in (self.dst if dst is None else dst)])
        A = kap * G * Kp * (qv[None, :] / cbar_i[:, None])
        if mv is not None:
            A = A * mv[:, None]
        A[~msk] = 0.0
        mu_ev = np.exp(th[self.idx['log_mu0']]) * np.array([self.mu_base(th, x) for x in t])
        lam = mu_ev * qv + A.sum(axis=0)
        P = A / lam[None, :]            # P[i,j] = 事件 j 由父 i 触发的概率
        p_bg = mu_ev * qv / lam
        return p_bg, P

    # ---- 解析积分 ----
    def Lambda_integral(self, th, a, b, t_hist, si_hist, mk_hist=None):
        cov = self.mu_bg_integral(th, a, b)
        int_bg = np.exp(th[self.idx['log_mu0']]) * cov
        int_tr = 0.0
        if t_hist is not None and len(t_hist):
            kap = np.exp(th[self.idx['log_kappa']])
            tt = np.asarray(t_hist, float)
            lo = np.maximum(a - tt, 0.0)
            hi = np.minimum(b - tt, self.W)
            ok = hi > lo
            if ok.any():
                GW = np.asarray(self.G_W(th, hi[ok])) - np.asarray(self.G_W(th, lo[ok]))
                mv = mk_hist[ok] if mk_hist is not None else 1.0
                int_tr = kap * float(np.sum(GW * mv))   # 后代率精确=κ·g
        return int_bg + int_tr

    def target_rate_integral(self, th, a, b, v_name, t_hist, si_hist, mk_hist=None, Kmat=None):
        """∫_a^b λ(t,v) dt — 前瞻评分"""
        Kmat = self.Kmat(th) if Kmat is None else Kmat
        iv = self.vidx[v_name]
        int_bg = np.exp(th[self.idx['log_mu0']]) * self.q[self.cidx[v_name]] * self.mu_bg_integral(th, a, b)
        int_tr = 0.0
        if t_hist is not None and len(t_hist):
            kap = np.exp(th[self.idx['log_kappa']])
            tt = np.asarray(t_hist, float)
            lo = np.maximum(a - tt, 0.0)
            hi = np.minimum(b - tt, self.W)
            ok = hi > lo
            if ok.any():
                Kv = np.array([self.K_at_hop(th, int(self.HopC[i, iv])) / self.cbar(th, i)
                               for i in np.asarray(si_hist)[ok]])
                GW = np.asarray(self.G_W(th, hi[ok])) - np.asarray(self.G_W(th, lo[ok]))
                mv = mk_hist[ok] if mk_hist is not None else 1.0
                int_tr = kap * float(np.sum(GW * Kv * mv)) * float(self.q[self.cidx[v_name]])
        return int_bg + int_tr

# ---------------- 参考模型（规范强度, ∫Σλ dt = R0·T) ----------------
class RefModel:
    def __init__(self, kind, t_tr, dst_tr, src_tr, C, cidx, hod=None,
                 tau_rec=3600.0, alpha=0.5, beta_knox=10.0, R0=None,
                 HopC=None, W_knox=600.0):
        self.kind = kind
        self.C = list(C); self.cidx = dict(cidx); self.nC = len(self.C)
        self.t_tr = np.asarray(t_tr, float)
        self.dst_tr, self.src_tr = list(dst_tr), list(src_tr)
        self.hod = hod if hod is not None else hour_profile(self.t_tr)
        self.tau_rec, self.alpha, self.beta_knox, self.W_knox = tau_rec, alpha, beta_knox, W_knox
        self.R0 = float(R0) if R0 is not None else len(self.t_tr) / max(self.t_tr.max() - self.t_tr.min(), 1.0)
        self.HopC = HopC
        cnt = np.zeros(self.nC)
        for d in self.dst_tr:
            if d in self.cidx:
                cnt[self.cidx[d]] += 1
        self.pop = (cnt + alpha) / (cnt + alpha).sum()
        if kind == 'edgecount':
            self.srcs = sorted(set(self.src_tr))
            self.ec = {}
            for s, d in zip(self.src_tr, self.dst_tr):
                self.ec[(s, d)] = self.ec.get((s, d), 0) + 1
            self.src_tot = {s: sum(v for (ss, dd), v in self.ec.items() if ss == s) for s in self.srcs}
        if kind == 'self_correct':
            self.theta = 0.0

    def vec(self, t, t_hist=None, dst_hist=None, si_hist=None, src=None, Nv_vec=None):
        """ξ(t,·) |C| 强度向量"""
        if self.kind == 'uniform':
            return np.full(self.nC, self.R0 / self.nC)
        if self.kind == 'pop':
            return self.R0 * self.pop
        if self.kind == 'hod_pop':
            h = int((t % 86400) // 3600)
            return self.R0 * self.hod[h] * self.pop
        if self.kind == 'edgecount':
            if src is not None and src in self.srcs:
                w = np.array([self.ec.get((src, c), 0) + self.alpha * self.nC * self.pop[i]
                              for i, c in enumerate(self.C)])
                tot = self.src_tot[src] + self.alpha * self.nC
                return self.R0 * w / tot
            return self.R0 * self.pop
        if self.kind == 'recency':
            lam = np.full(self.nC, self.R0 / self.nC * 1e-3)
            for tt, d in zip(t_hist, dst_hist):
                dt = t - tt
                if 0 < dt <= 48 * 3600 and d in self.cidx:
                    lam[self.cidx[d]] += np.exp(-dt / self.tau_rec) / self.tau_rec
            return lam
        if self.kind == 'knox_rule':
            base = np.full(self.nC, self.R0 / self.nC)
            if self.HopC is not None and si_hist is not None and len(si_hist):
                tt = np.asarray(t_hist, float)
                recent = (t - tt > 0) & (t - tt <= self.W_knox)
                if recent.any():
                    hot = (self.HopC[np.asarray(si_hist)[recent]] <= 1).any(axis=0)
                    base[hot] *= self.beta_knox
            return base
        if self.kind == 'self_correct':
            if Nv_vec is None:
                Nv_vec = np.zeros(self.nC)
                for d in dst_hist:
                    if d in self.cidx:
                        Nv_vec[self.cidx[d]] += 1
            w = self.pop * np.exp(-self.theta * Nv_vec)
            return self.R0 * w / w.sum()
        raise ValueError(self.kind)

    def fit_self_correct(self):
        """θ≥0 一维 MLE（网格+精修; 似然含阶梯积分的近似, 记录于文档）"""
        def neg(theta):
            th = float(np.asarray(theta).ravel()[0])
            ll = 0.0
            t_hist, dst_hist = self.t_tr, self.dst_tr
            Nv = np.zeros(self.nC)
            for j in range(len(t_hist)):
                lam = self.R0 * self.pop[self.cidx[dst_hist[j]]] * np.exp(-th * Nv[self.cidx[dst_hist[j]]])
                ll += np.log(max(lam, 1e-300))
                Nv[self.cidx[dst_hist[j]]] += 1
            grid = np.arange(self.t_tr.min(), self.t_tr.max(), 600.0)
            tot = 0.0
            Nv = np.zeros(self.nC)
            gi = 0
            for j in range(len(t_hist) + 1):
                t_next = t_hist[j] if j < len(t_hist) else self.t_tr.max()
                while gi < len(grid) and grid[gi] < t_next:
                    pass
                    gi += 1
            # 数值积分: 每 600s 网格点上 Σ_v pop·exp(−θN_v)×R0×600 — N_v 需事件时间序; 直接逐网格重建
            steps = np.arange(self.t_tr.min(), self.t_tr.max(), 600.0)
            for x in steps:
                s = 0.0
                for i, c in enumerate(self.C):
                    nv = 0
                    # O(nC·n·nGrid) 太慢 → 限制 C 到 V* 活跃集 + 其余合并
                break
            # 简化积分: 目标空间总强度≈R0·(1−θ·avg_N/…) 解析近似不可靠 → 用逐事件区间精确重建
            tot = 0.0
            Nv = np.zeros(self.nC)
            bounds = list(t_hist) + [t_hist.max() + 600.0]
            for j in range(len(t_hist)):
                seg = bounds[j + 1] - bounds[j]
                s = float(self.pop @ np.exp(-th * Nv))
                tot += s * seg
                if dst_hist[j] in self.cidx:
                    Nv[self.cidx[dst_hist[j]]] += 1
            return -(ll - self.R0 * tot)
        res = minimize(lambda z: neg(z), 0.0, method='L-BFGS-B', bounds=[(0.0, 5.0)])
        self.theta = float(res.x[0])
        return self.theta

# ---------------- 评分引擎（严格时间序, 逐事件） ----------------
class Scorer:
    """
    对测试事件流 (t_j, v_j, src_j) 逐事件计分; 历史从 (t_hist0, dst_hist0, si_hist0) 起步。
    model: GraphHawkes（用 .th）; refs: {name: RefModel}
    返回 per-event: λ_model, {ref: ξ_ref}, λ_vec(排名用)
    """
    def __init__(self, model, refs=None, C=None, cidx=None, succ_te=None):
        self.m = model
        self.refs = refs or {}
        self.C, self.cidx = C, dict(cidx)
        self.Kmat = model.Kmat(model.th) if model is not None else None
        self.succ_te = succ_te

    def run(self, t_te, dst_te, src_te, t_hist0=None, dst_hist0=None, si_hist0=None, mk_hist0=None,
            rolling_refit=False, split_t=None, train_t=None, train_dst=None, train_src=None, train_succ=None):
        m = self.m
        n_te = len(t_te)
        t_hist0 = [] if t_hist0 is None else list(t_hist0)
        dst_hist0 = [] if dst_hist0 is None else list(dst_hist0)
        si_hist0 = [] if si_hist0 is None else list(si_hist0)
        mk_hist0 = None
        out = dict(lam_at=np.zeros(n_te), lam_vec=[], refs={k: np.zeros(n_te) for k in self.refs},
                   ref_vec={}, Nv=None)
        # IW 的 Nv 计数器
        Nv = np.zeros(self.m.nC)
        for j in range(n_te):
            t, v = float(t_te[j]), dst_te[j]
            if rolling_refit and split_t is not None and t >= split_t:
                # 滚动重拟合: 用截至当前的全部历史
                all_t = np.array(list(train_t) + t_hist0)
                all_dst = list(train_dst) + dst_hist0
                all_src = list(train_src) + [src_te[k] for k in range(len(t_hist0))] if False else None
                m2 = GraphHawkes(all_t, all_dst, train_src + ['?'] * len(t_hist0), m.adj, m.nodes_all,
                                 kernel=m.kernel, gk=m.gk, bg=m.bg, cand=m.cand, W=m.W, qbg=m.qbg, marks=None)
                m2.fit(n_restarts=3, seed=7)
                m.th = m2.th
                self.Kmat = m.Kmat(m.th)
            lam_vec = m.lambda_vec_C(m.th, t, t_hist0, si_hist0, mk_hist0, self.Kmat)
            out['lam_at'][j] = lam_vec[self.cidx[v]] if v in self.cidx else 0.0
            out['lam_vec'].append(lam_vec)
            for rn, rf in self.refs.items():
                xi_vec = rf.vec(t, t_hist0, dst_hist0, si_hist0, src=src_te[j] if src_te is not None else None,
                                Nv_vec=Nv)
                out['refs'][rn][j] = xi_vec[self.cidx[v]] if v in self.cidx else 0.0
                out['ref_vec'].setdefault(rn, []).append(xi_vec)
            # 历史推进
            t_hist0.append(t); dst_hist0.append(v)
            si_hist0.append(m.vidx[v])
            if v in m.cidx:
                Nv[m.cidx[v]] += 1
            mk_hist0 = None
        out['Nv_final'] = Nv
        return out

# ---------------- 统计 ----------------
def igpe(gain_bits):
    g = np.asarray(gain_bits, float)
    n = len(g)
    m = float(g.mean())
    if n > 1 and g.std(ddof=1) > 0:
        t, p = stats.ttest_1samp(g, 0.0)
        se = float(g.std(ddof=1) / np.sqrt(n))
        try:
            pw = float(stats.wilcoxon(g).pvalue)
        except Exception:
            pw = float('nan')
    else:
        t = p = se = pw = float('nan')
    return dict(IGPE_bits=m, se=se, t=float(t), p=float(p), p_wilcoxon=pw, n=n)

def rank_of_true(vec, true_c, tie='mid'):
    """单正例排名; AP=MRR=1/rank; tie: mid|pess|opt"""
    pv = np.asarray(vec, float)
    order = np.argsort(-pv, kind='stable')
    sp = pv[order]
    pos = int(np.where(order == true_c)[0][0])
    g_hi = int(np.sum(sp > sp[pos] + 1e-15))
    g_eq = int(np.sum(np.abs(sp - sp[pos]) <= 1e-15))
    if tie == 'mid':
        rank = g_hi + (g_eq + 1) / 2.0
    elif tie == 'pess':
        rank = g_hi + g_eq
    else:
        rank = g_hi + 1.0
    return 1.0 / rank, rank
