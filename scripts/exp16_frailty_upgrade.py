#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exp16 frailty 升级（R2 主题③ B+ 核心；F1-F6/V1/V3/V4/V7）
问题: exp12 判定的 boost 份额（share_boost=0.72）是否含未建模的逐主机异质性伪成分?
  —— 三条独立文献线预言遗漏随机效应的偏置方向（癫痫 Bayesian mixed Hawkes 2605.22038:
  「省略 random effects→低估背景、高估激发」; Cherkaoui 2025: 忽略外部激发→分支比 0.71 被拟合为
  1.04; Dominoes-or-weather: regime-switching 拟合 Hawkes 产生伪近临界）→ frailty 检验=
  判定框架的内部一致性要求。
升级组件:
  ① flag 侧: Gamma-Poisson EB 从矩法(α=0.0095)升级为边际 NB-MLE 超参版（等暴露）+
     Bühlmann-Straub 暴露加权版（e_v=deg(v)+1 图拓扑攻击面, D1 精算桥; q_v∝(α+n_v)/(β+e_v),
     unseen 主机按暴露反比收缩——直指 M_flag unseen −1.88 bits）
  ② boost 侧: 触发核从 Omori 单尺度升级为双时间尺度指数和 g=w·e^{−t/τ1}+(1−w)·e^{−t/τ2}
     （τ1=300s 分钟级固定, τ2∈{24h,72h,7d} 档位; 只估权重 w——可辨识性惯例 V3/A2）
  ③ 长尾由 flag 静态项承担, 时间支撑差仍是识别支柱（核窗敏感性已有 exp15 W12/24/48h）
协议（F4, 预登记）:
  - train 内时间再切分: 拟合段=[t0, t_v)（前 80% 时间）、验证段=[t_v, t_cut)（后 20%）
  - 候选网格=核(omori/三档 twostage)×flag(mom/mle/mle_deg)=12 配置; α 全部只从拟合段计数估计
  - 验证判据=验证段逐事件 IGPE vs uniform 均值; 平手(差<0.05 bits)取更简约（核 omori 优先、flag mom 优先）
  - 选定配置后全 train 重拟合; 测试段 143 事件只碰一次
预登记判定规则（先落盘后运行）:
  G1 主判定: IGPE(hybrid_frailty − M_hybrid_W24_c[exp12]) 移动块 bootstrap(block=10,B=5000) 95%CI:
     下界>0→frailty 显著升级, 三成分叙事; CI 含 0→不显著, 如实申报回退 hybrid（frailty=阴性升级）;
     上界<0→frailty 有害, 回退并报告。三轨复核=AR(1)+Newey-West HAC（exp14 口径）。
  G2 方向性预言: 若 exp12 boost 份额含异质性伪成分 → frailty 版 boost 边际缩小、flag 边际增大 →
     share_boost_fr<0.72 且 boost 边际点值<+2.318（exp12 版）; 反向/持平→预言不支持, 双向成文。
  G3 margin 形态: frailty 版 hybrid−flag 边际 median/正负占比/Wilcoxon/top-10 份额/钉地板数
     vs exp12 版（median −0.657、正45负98、p=0.8347、top-10 占 51.5%、85/143 钉背景地板）。
功效局限（预登记）: n_train=570<1000（exp Hawkes 判别可决性起点, A3 判别 atlas）;
  η=0→η̂≈0.80 伪自激锚点在案——判别结论一律「模型比较口径」而非「裁决定案」。
退化检测（F6/P4）: a=κ·G_W(W)∈(0.05,0.949) 外标 degenerate（补 a≈0 核死亡分支）;
  α̂→大(>100)→零异质性退化警告（ASTIN: 方差≤均值→后验=先验）。
"""
import sys, json, time, argparse
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
sys.path.insert(0, './scripts')
sys.path.insert(0, './src')
from exp_patch_common import (load_all, fit_constrained, score_Ktilde, eb_flag_q,
                              moving_block_bootstrap, igpe_full, nw_hac, R)
from lanl_pp import GraphHawkes, hour_profile, igpe

A_MAX = 0.95
TAU1 = 300.0
TAU2_GRID = {'ts24h': 86400.0, 'ts72h': 259200.0, 'ts7d': 604800.0}


def eb_flag_mle(counts, e_vec=None, n_restarts=6, seed=3):
    """Gamma-Poisson EB 超参 MLE。等暴露参数化 (α, μ)（(α,β) 脊线不可辨识）;
    暴露加权参数化 (α, logβ), 边际 Σ_v [gammaln(α+n_v)−gammaln(α)−gammaln(n+1)+α log p0+n log(1−p0)],
    p0=β/(β+e_v)。返回 (α, β, nll_min)"""
    n = np.asarray(counts, float)
    equal_e = e_vec is None
    e = np.ones_like(n) if equal_e else np.asarray(e_vec, float)
    ebar = float(e.mean())

    def nll(x):
        a = np.exp(np.clip(x[0], -12, 12))
        if equal_e:
            mu = np.exp(np.clip(x[1], -12, 14))
            p0 = a / (a + mu)
            ll = np.sum(gammaln(a + n) - gammaln(a) - gammaln(n + 1.0)
                        + a * np.log(p0) + n * np.log(mu / (a + mu)))
        else:
            b = np.exp(np.clip(x[1], -12, 20))
            p0 = b / (b + e)
            ll = np.sum(gammaln(a + n) - gammaln(a) - gammaln(n + 1.0)
                        + a * np.log(p0) + n * np.log1p(-p0))
        return -ll if np.isfinite(ll) else 1e18

    rng = np.random.default_rng(seed)
    best = (np.inf, None)
    for r in range(n_restarts):
        x0 = np.array([np.log(0.05), np.log(0.5)]) if r == 0 else \
            np.clip(np.array([np.log(0.05), np.log(0.5)]) + rng.normal(0, 1.0, 2), -11.5, 11.5)
        res = minimize(nll, x0, method='Nelder-Mead',
                       options=dict(maxiter=4000, xatol=1e-10, fatol=1e-10))
        if res.fun < best[0]:
            best = (float(res.fun), res.x.copy())
    a = float(np.exp(best[1][0]))
    if equal_e:
        mu = float(np.exp(best[1][1]))
        beta = a * ebar / mu
    else:
        beta = float(np.exp(best[1][1]))
    return a, beta, best[0]


def flag_q_from(alpha, beta, n, e_vec=None):
    """后验均值收缩 q: 等暴露 q∝(n+α); 暴露加权 q∝(α+n)/(β+e_v)"""
    n = np.asarray(n, float)
    q = n + alpha if e_vec is None else (n + alpha) / (beta + np.asarray(e_vec, float))
    return q / q.sum()


def alpha_profile_ci(counts, e_vec=None, nll_at=None):
    """α 的 profile-似然 95% CI（V7: EB 超参不确定性申报）: α 一维 profile, Δll=1.9207 边界"""
    n = np.asarray(counts, float)
    equal_e = e_vec is None
    e = np.ones_like(n) if equal_e else np.asarray(e_vec, float)
    ebar = float(e.mean())

    def prof(a):
        if equal_e:
            mu_hat = max(n.mean(), (n.sum() + 1) / len(n))  # 等暴露 μ̂ 解析≈样本均值
            p0 = a / (a + mu_hat)
            ll = np.sum(gammaln(a + n) - gammaln(a) - gammaln(n + 1.0) + a * np.log(p0)
                        + n * np.log(mu_hat / (a + mu_hat)))
            return ll
        grid = np.exp(np.linspace(np.log(1e-4), np.log(1e4), 200))
        best = -np.inf
        for b in grid:
            p0 = b / (b + e)
            ll = np.sum(gammaln(a + n) - gammaln(a) - gammaln(n + 1.0) + a * np.log(p0) + n * np.log1p(-p0))
            best = max(best, ll)
        return best

    # α̂ 用全局 MLE 的 x
    a_hat, beta_hat, nll_min = (nll_at if nll_at else (None, None, None))
    grid = np.exp(np.linspace(np.log(1e-4), np.log(1e4), 800))
    lls = np.array([prof(a) for a in grid])
    imax = int(np.argmax(lls))
    thr = lls[imax] - 1.9207
    ok = lls >= thr
    lo_ = grid[ok].min() if ok.any() else float('nan')
    hi_ = grid[ok].max() if ok.any() else float('nan')
    return dict(alpha_profile_mle=float(grid[imax]), ci=[float(lo_), float(hi_)],
                ci_hi_truncated=bool(hi_ >= grid[-1] * 0.999),
                var_minus_mean=float(n.var(ddof=0) - n.mean()),
                note='profile 95% CI (800 点 log 网格); var≤mean(→α→∞ 零异质性)时后验=先验(ASTIN 警告); '
                     'ci_hi_truncated=True 表示上界顶到网格端点(区间更宽)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    t00 = time.time()
    D = load_all()
    adj, nodes, ev = D['adj'], D['nodes'], D['ev']
    t_all, dst_all, src_all = D['t_all'], D['dst_all'], D['src_all']
    cut, t_cut, V_all = D['cut'], D['t_cut'], D['V_all']
    m_h_full = hour_profile(t_all[:cut])
    R0 = cut / (t_cut - t_all[0])
    nC_all = len(nodes)
    cidxC = {c: i for i, c in enumerate(nodes)}
    cntC_full = np.zeros(nC_all)
    for d in dst_all[:cut]:
        cntC_full[cidxC[d]] += 1
    popC = (cntC_full + 0.5) / (cntC_full + 0.5).sum()
    deg = np.zeros(nC_all)
    for s, nbrs in adj.items():
        for w in nbrs:
            if s in cidxC and w in cidxC:
                deg[cidxC[s]] += 1
    deg = deg + 1.0   # e_v = deg+1（图拓扑攻击面, 含孤立主机）

    # ---------- 预登记先落盘 ----------
    out = dict(exp='Exp16_frailty_upgrade', preregistered=dict(
        G1='IGPE(hybrid_frailty − M_hybrid_W24_c) 块bootstrap(block=10,B=5000) CI 下界>0→显著升级; '
           '含0→不显著回退; 上界<0→有害。三轨 AR(1)/HAC 复核',
        G2='方向性预言: share_boost_fr<0.72 且 boost 边际<+2.318 → 支持异质性伪成分假说; 双向成文',
        G3='margin 形态 vs exp12 版(median −0.657/正45负98/p=0.8347/top-10 51.5%/85 钉地板)',
        validation='train 内 80/20 时间再切分; 验证判据=IGPE vs uniform 均值; 平手取简约; 测试集只碰一次',
        power_limitation='n_train=570<1000 判别可决性起点; 结论限「模型比较口径」'),
        A_MAX=A_MAX, TAU1_s=TAU1, TAU2_grid=TAU2_GRID)

    def dump():
        json.dump(out, open(f'{R}/exp16_frailty_upgrade.json', 'w'), indent=1, ensure_ascii=False, default=float)
    dump()
    print('预登记已落盘 exp16_frailty_upgrade.json（先落盘后运行）', flush=True)

    # ---------- 验证切分 ----------
    t0v = float(t_all[0])
    t_v = t0v + 0.8 * (t_cut - t0v)
    cut_v = int(np.searchsorted(t_all, t_v, side='left'))
    if args.smoke:
        cut_v = min(cut_v, 300)
    print(f'验证切分: 拟合段 n={cut_v} [{(t0v):.0f},{t_v:.0f}) 验证段 n={cut-cut_v} [{t_v:.0f},{t_cut:.0f})', flush=True)
    assert cut - cut_v >= 50 and cut_v >= 200, '验证/拟合段样本量不足'
    t_fit, dst_fit, src_fit = t_all[:cut_v], list(dst_all[:cut_v]), list(src_all[:cut_v])
    t_val, dst_val, src_val = t_all[cut_v:cut], list(dst_all[cut_v:cut]), list(src_all[cut_v:cut])
    y_val_pos = len(set(dst_val) & set(V_all))
    assert y_val_pos > 0

    # ---------- 三种 flag 变体（只从拟合段计数估计, 防验证泄漏） ----------
    cnt_fit = np.zeros(nC_all)
    for d in dst_fit:
        cnt_fit[cidxC[d]] += 1
    q_mom_fit, alpha_mom_fit, _ = eb_flag_q(cnt_fit)          # 矩法（exp12 同公式, 重估于拟合段）
    a_mle, b_mle, nll_mle = eb_flag_mle(cnt_fit, e_vec=None)
    q_mle_fit = flag_q_from(a_mle, b_mle, cnt_fit, e_vec=None)
    a_mled, b_mled, nll_mled = eb_flag_mle(cnt_fit, e_vec=deg)
    q_mled_fit = flag_q_from(a_mled, b_mled, cnt_fit, e_vec=deg)
    print(f'flag 变体(拟合段): mom α={alpha_mom_fit:.4g} | mle α={a_mle:.4g} β={b_mle:.4g} | '
          f'mle_deg α={a_mled:.4g} β={b_mled:.4g}', flush=True)
    flag_variants = {'mom': q_mom_fit, 'mle': q_mle_fit, 'mle_deg': q_mled_fit}

    # ---------- 候选网格验证集选模 ----------
    def build(kernel, W, q_custom, tau_stage=None, t_arr=None, dst_arr=None):
        return GraphHawkes(t_arr if t_arr is not None else t_fit,
                           list(dst_arr if dst_arr is not None else dst_fit),
                           list(src_fit), adj, nodes, dst_extra=V_all,
                           kernel=kernel, gk='hop0', bg='hod24', W=W, qbg='custom',
                           q_custom=q_custom, tau_stage=tau_stage)

    kernel_grid = [('omori', 86400.0, None)] + [(k, v, (TAU1, v)) for k, v in TAU2_GRID.items()]
    flag_order = ['mom', 'mle', 'mle_deg']
    kernel_order = ['omori', 'ts24h', 'ts72h', 'ts7d']
    ctx = dict(pop=popC, m_h=m_h_full, R0=R0, flag_q=None, theta_iw=0.0, ECNT={'cnt': {}, 'tot': {}})
    cand = []
    for kn, W, ts in kernel_grid:
        for fn in flag_order:
            cand.append((kn, W, ts, fn))
    if args.smoke:
        cand = [c for c in cand if c[3] in ('mom', 'mle') and c[0] in ('omori', 'ts24h')]
    print(f'== 候选网格 {len(cand)} 配置: 拟合段约束拟合 → 验证段 IGPE 选模 ==', flush=True)
    val_results = {}
    for kn, W, ts, fn in cand:
        key = f'{kn}+{fn}'
        try:
            m = build(kn, W, flag_variants[fn], tau_stage=ts)
            th, ll, a = fit_constrained(m, a_max=A_MAX, n_restarts=4, seed=17, t_end=float(t_v))
            si0 = [m.vidx[d] for d in dst_fit]
            la, _, ra, _ = score_Ktilde(m, th, ['uniform'], t_val, dst_val, src_val,
                                        t_fit, dst_fit, si0, ctx)
            g = np.log2(la / ra['uniform'])
            st = igpe(g)
            val_results[key] = dict(IGPE_bits=st['IGPE_bits'], p=st['p'], n=len(g),
                                    train_ll=round(ll, 2), branch_a=round(a, 4), W=W,
                                    tau_stage=[ts[0], ts[1]] if ts else None)
            print(f'  {key}: val IGPE={st["IGPE_bits"]:+.4f} (n={len(g)}) ll={ll:.1f} a={a:.4f}', flush=True)
        except Exception as e:
            val_results[key] = dict(error=str(e)[:150])
            print(f'  {key} ERROR {e}', flush=True)
    out['validation'] = dict(cut_v=cut_v, t_v=t_v, n_val=cut - cut_v, results=val_results)

    # 选优（预登记: 平手<0.05 bits 取简约; 顺序=kernel_order×flag_order 的逆）
    ok = {k: v for k, v in val_results.items() if 'IGPE_bits' in v and np.isfinite(v['IGPE_bits'])}
    best_key, best_val = None, -1e18
    for kn in kernel_order[::-1]:        # 逆序遍历, 平手时保留先遇到的更简约档
        for fn in flag_order[::-1]:
            k = f'{kn}+{fn}'
            if k in ok and ok[k]['IGPE_bits'] > best_val + 0.05:
                best_val, best_key = ok[k]['IGPE_bits'], k
    # 严格最优再检查是否在 0.05 内被简约档追平（简约优先原则的对称实现）
    strict_best = max((v['IGPE_bits'] for v in ok.values()), default=np.nan)
    for kn in kernel_order:
        for fn in flag_order:
            k = f'{kn}+{fn}'
            if k in ok and abs(ok[k]['IGPE_bits'] - strict_best) <= 0.05:
                best_key, best_val = k, ok[k]['IGPE_bits']
                break
        else:
            continue
        break
    sel_kernel, sel_flag = best_key.split('+')
    print(f'== 选定: {best_key} (val IGPE={best_val:+.4f}) ==', flush=True)
    # runner-up 披露（选模透明度: 点值最高者与最终选定者的关系）
    ru_key = max(ok, key=lambda k: ok[k]['IGPE_bits'])
    out['selected'] = dict(key=best_key, val_IGPE=best_val,
                           val_best_raw=ru_key, val_best_raw_IGPE=float(ok[ru_key]['IGPE_bits']),
                           gap_to_best_raw=float(ok[ru_key]['IGPE_bits'] - best_val),
                           tie_rule='|Δ|≤0.05 bits 取更简约（核 omori 优先、flag mom 优先）',
                           note='若 gap>0: 点值最高配置被简约规则压过, 差额远小于 MDE 1.10 bits 时不构成选模翻转')
    dump()

    # ---------- 全 train 重拟合（选定配置 + 2 中间消融 + 2 成分模型） ----------
    sel_tau2 = TAU2_GRID.get(sel_kernel)
    sel_ts = (TAU1, sel_tau2) if sel_tau2 else None
    sel_W = 86400.0 if sel_kernel == 'omori' else sel_tau2
    # flag 变体重估于全 train（选型判据同拟合段版变体族; α 变体在全 train 重算）
    q_mom, alpha_mom, _ = eb_flag_q(cntC_full)
    a_mle_F, b_mle_F, nll_mle_F = eb_flag_mle(cntC_full, e_vec=None)
    q_mle_F = flag_q_from(a_mle_F, b_mle_F, cntC_full, e_vec=None)
    a_mled_F, b_mled_F, nll_mled_F = eb_flag_mle(cntC_full, e_vec=deg)
    q_mled_F = flag_q_from(a_mled_F, b_mled_F, cntC_full, e_vec=deg)
    flag_full = {'mom': q_mom, 'mle': q_mle_F, 'mle_deg': q_mled_F}
    q_sel = flag_full[sel_flag]
    # 全 train GraphHawkes（q/m_h 由全 train; 结构同 exp12 build）
    def build_full(kernel, W, q_custom, tau_stage=None):
        return GraphHawkes(t_all[:cut], list(dst_all[:cut]), list(src_all[:cut]), adj, nodes,
                           dst_extra=V_all, kernel=kernel, gk='hop0', bg='hod24', W=W,
                           qbg='custom', q_custom=q_custom, tau_stage=tau_stage)

    fits = {}
    print('== 全 train 重拟合（约束 a≤0.95） ==', flush=True)
    for name, kn, W, ts, qv in [
            ('hybrid_fr', sel_kernel, sel_W, sel_ts, q_sel),
            ('hybrid_mle_omori', 'omori', 86400.0, None, q_mle_F),
            ('hybrid_mom_ts', sel_kernel, sel_W, sel_ts, q_mom),
            ('boost_fr', sel_kernel, sel_W, sel_ts, None)]:
        m = build_full(kn, W, qv if qv is not None else popC, tau_stage=ts)
        if qv is None:   # boost: qbg=pop
            m = GraphHawkes(t_all[:cut], list(dst_all[:cut]), list(src_all[:cut]), adj, nodes,
                            dst_extra=V_all, kernel=kn, gk='hop0', bg='hod24', W=W,
                            qbg='pop', tau_stage=ts)
        th, ll, a = fit_constrained(m, a_max=A_MAX, n_restarts=6, seed=23, t_end=t_cut)
        k_decl = m.n_par + (1 if name.startswith('hybrid') else 0)   # +EB α（hybrid 族声明含超参）
        extra = {}
        if kn == 'twostage':
            w = float(1.0 / (1.0 + np.exp(-th[m.idx['logit_w1']])))
            extra = dict(twostage_w_short=w, twostage_mass_long=float(1 - w),
                         note='w=短分量(τ1=300s)权重; 三档 τ2 结果一致的真因在此')
        fits[name] = dict(model=m, th=th, ll=ll, a=a, k=k_decl,
                          BIC=round(-2 * ll + k_decl * np.log(cut), 2),
                          n_par=k_decl, W=W,
                          tau_stage=[ts[0], ts[1]] if ts else None,
                          degenerate=bool(a < 0.05 or a > 0.949), **extra)
        np.save(f'{R}/exp16_th_{name}.npy', th)
        print(f'  {name}: ll={ll:.2f} a={a:.4f} k={k_decl} BIC={fits[name]["BIC"]:.2f} '
              f'degenerate={fits[name]["degenerate"]}', flush=True)
    # 无约束对照（遗漏随机效应偏倚方向复测: 约束 vs 无约束 (μ0,κ) 偏移）
    m_unc = build_full(sel_kernel, sel_W, q_sel, tau_stage=sel_ts)
    m_unc.fit(n_restarts=4, seed=29, t_end=t_cut)
    th_u = m_unc.th
    a_u = float(np.exp(th_u[m_unc.idx['log_kappa']]) * np.asarray(m_unc.G_W(th_u, m_unc.W)))
    fits['hybrid_fr_unconstrained'] = dict(ll=float(m_unc.ll_max), branch_a=a_u,
        mu0=float(np.exp(th_u[m_unc.idx['log_mu0']])), kappa=float(np.exp(th_u[m_unc.idx['log_kappa']])),
        mu0_constrained=float(np.exp(fits['hybrid_fr']['th'][m_unc.idx['log_mu0']])),
        kappa_constrained=float(np.exp(fits['hybrid_fr']['th'][m_unc.idx['log_kappa']])),
        bias_note='癫痫 2605.22038 预言: 遗漏随机效应→低估背景(μ0↓)高估激发(κ↑); '
                  '此处比较约束(含 frailty flag)vs 无约束同构拟合')
    print(f"  无约束对照: ll={m_unc.ll_max:.2f} a={a_u:.4f} μ0={fits['hybrid_fr_unconstrained']['mu0']:.3e} "
          f"κ={fits['hybrid_fr_unconstrained']['kappa']:.4f}", flush=True)
    out['model_family_full'] = {k: {kk: vv for kk, vv in v.items() if kk not in ('model', 'th')}
                                for k, v in fits.items()}

    # flag/boost/EB 申报块
    out['flag_frailty'] = dict(
        full_train=dict(mom_alpha=alpha_mom, mle_alpha=a_mle_F, mle_beta=b_mle_F,
                        mle_deg_alpha=a_mled_F, mle_deg_beta=b_mled_F),
        alpha_mle_profile_ci=alpha_profile_ci(cntC_full, e_vec=None, nll_at=(a_mle_F, b_mle_F, nll_mle_F)),
        alpha_mle_deg_profile_ci=alpha_profile_ci(cntC_full, e_vec=deg, nll_at=(a_mled_F, b_mled_F, nll_mled_F)),
        q_shift=dict(max_abs_log2_ratio=float(np.abs(np.log2(q_mle_F / q_mom)).max()),
                     mean_abs_log2_ratio=float(np.abs(np.log2(q_mle_F / q_mom)).mean()),
                     unseen_max_log2_ratio=float(np.log2(q_mled_F / q_mom)[cntC_full == 0].max()),
                     note='mle/mom 与 mle_deg/mom 的 q 位移; unseen 层=零计数主机的最大位移(V1 检验)'))
    dump()

    # ---------- 测试段评分（只碰一次; K̃ 口径同 exp12） ----------
    print('== 测试段评分 ==', flush=True)
    t_te, dst_te, src_te = t_all[cut:], list(dst_all[cut:]), list(src_all[cut:])
    si0_full = [fits['hybrid_fr']['model'].vidx[d] for d in dst_all[:cut]]
    lam = {}
    ref_store = None
    for name in ['hybrid_fr', 'hybrid_mle_omori', 'hybrid_mom_ts', 'boost_fr']:
        m = fits[name]['model']
        la, lv, ra, rv = score_Ktilde(m, fits[name]['th'], ['uniform', 'pop', 'hod_pop'],
                                      t_te, dst_te, src_te, list(t_all[:cut]), list(dst_all[:cut]),
                                      si0_full, ctx)
        lam[name] = la
        if ref_store is None:
            ref_store = ra
        print(f'  {name}: vs uni={igpe(np.log2(la/ref_store["uniform"]))["IGPE_bits"]:+.3f}', flush=True)
    # flag_fr 闭式（同 exp12 M_flag 语义: λ=μ0·m_h·q_flag）
    from lanl_pp import hour_coverage_seconds
    cov = hour_coverage_seconds(t_all[0], float(t_all[cut - 1]) + 1.0)
    mu0_fr = cut / float(cov.sum())
    lam['flag_fr'] = np.array([mu0_fr * m_h_full[int((t % 86400) // 3600)] * q_sel[cidxC[v]]
                               for t, v in zip(t_te, dst_te)])
    # exp12 现成 lam（防重算漂移, 直接读权威 npy）
    lam['M_hybrid_W24_c'] = np.load(f'{R}/exp12_lam_M_hybrid_W24_c.npy')
    lam['M_flag'] = np.load(f'{R}/exp12_lam_M_flag.npy')
    lam['M_boost_W24_c'] = np.load(f'{R}/exp12_lam_M_boost_W24_c.npy')

    # 逐事件测试 log-loss（V4 竞争模型口径: 同数据 log2 λ 均值, 越大越好）
    out['test_log2lam'] = {k: float(np.log2(lam[k]).mean()) for k in lam}
    out['test_IGPE'] = {}
    for k in lam:
        st = igpe(np.log2(lam[k] / ref_store['uniform']))
        out['test_IGPE'][k] = dict(vs_uniform=st['IGPE_bits'], p=st['p'])
        print(f'  IGPE {k}: {st["IGPE_bits"]:+.4f} (p={st["p"]:.3g})', flush=True)

    # ---------- 判定 margins（预登记 G1-G3） ----------
    print('== 判定 margins（块 bootstrap block=10×5000 + 三轨） ==', flush=True)
    margins = {}
    for a_, b_ in [('hybrid_fr', 'M_hybrid_W24_c'), ('hybrid_fr', 'flag_fr'),
                   ('hybrid_fr', 'boost_fr'), ('hybrid_fr', 'M_flag'),
                   ('hybrid_mle_omori', 'M_hybrid_W24_c'), ('hybrid_mom_ts', 'M_hybrid_W24_c')]:
        g = np.log2(lam[a_] / lam[b_])
        st = igpe_full(g)
        bb = moving_block_bootstrap(g, block=10, n_boot=5000, seed=13)
        pos = int((g > 0).sum()); neg = int((g < 0).sum())
        floor_v = np.median(g)
        n_floor = int((np.abs(g - floor_v) < 1e-9).sum())
        margins[f'{a_}_gt_{b_}'] = dict(
            IGPE_bits=st['IGPE_bits'], boot_ci=[bb['ci_lo'], bb['ci_hi']], p_boot=bb['p_boot_two'],
            t_ar1=st['t_ar1'], p_ar1=st['p_ar1'], p_hac=float(nw_hac(g)['p_hac_normal']),
            median=float(np.median(g)), n_pos=pos, n_neg=neg, p_wilcoxon=st['p_wilcoxon'],
            n_floor=n_floor, top10_share=float(np.sort(g)[::-1][:10].sum() / g.sum()) if g.sum() > 0 else None,
            n=len(g))
        np.save(f'{R}/exp16_margin_{a_}_gt_{b_}.npy', g)
        print(f"  {a_} − {b_}: {st['IGPE_bits']:+.4f} CI=[{bb['ci_lo']:+.3f},{bb['ci_hi']:+.3f}] "
              f"p_boot={bb['p_boot_two']:.3g} median={np.median(g):+.3f} 正{pos}/负{neg} "
              f"p_wil={st['p_wilcoxon']:.3g} 钉地板={n_floor}", flush=True)
    out['margins'] = margins

    # G1 判定
    g1 = margins['hybrid_fr_gt_M_hybrid_W24_c']
    same_as_current = (sel_kernel == 'omori' and sel_flag == 'mom')
    if same_as_current and abs(g1['IGPE_bits']) < 0.01:
        verdict = ('选定配置与现版 M-hybrid 同构（验证集未选出更优变体）→ frailty 升级在此数据上'
                   '无实质变化, 回退 hybrid（如实的阴性升级）')
    elif g1['boot_ci'][0] > 0:
        verdict = 'frailty 显著升级（CI 下界>0）→ 三成分模型叙事'
    elif g1['boot_ci'][1] < 0:
        verdict = 'frailty 有害（CI 上界<0）→ 回退 hybrid 并报告'
    else:
        verdict = '不显著（CI 含 0）→ 如实申报回退 hybrid, frailty 写为阴性升级'
    # G2 方向性预言
    d_boost = margins['hybrid_fr_gt_flag_fr']['IGPE_bits']
    d_flag = margins['hybrid_fr_gt_boost_fr']['IGPE_bits']
    share_fr = d_boost / (d_boost + d_flag) if (d_boost + d_flag) > 0 else float('nan')
    g2 = dict(share_boost_fr=float(share_fr), share_boost_exp12=0.7201817792534264,
              boost_margin_fr=float(d_boost), boost_margin_exp12=2.3181,
              supported=(bool(share_fr < 0.7201817792534264 and d_boost < 2.3181)
                         if not (same_as_current and abs(g1['IGPE_bits']) < 0.01) else None),
              note='支持=boost 份额缩小且边际缩小（异质性伪成分被 frailty 吸收）; 双向成文; '
                   '同构配置时无意义置 None')
    # G3 margin 形态对照（exp12 版从权威 npy 现算, 防誊写漂移）
    g12 = np.load(f'{R}/exp12_margin_M_hybrid_W24_c_gt_M_flag.npy')
    fl12 = float(np.median(g12)); nf12 = int((np.abs(g12 - fl12) < 1e-9).sum())
    g3 = dict(exp12=dict(median=fl12, n_pos=int((g12 > 0).sum()), n_neg=int((g12 < 0).sum()),
                         n_floor=nf12, top10_share=float(np.sort(g12)[::-1][:10].sum() / g12.sum())),
              frailty=dict(median=margins['hybrid_fr_gt_flag_fr']['median'],
                           n_pos=margins['hybrid_fr_gt_flag_fr']['n_pos'],
                           n_neg=margins['hybrid_fr_gt_flag_fr']['n_neg'],
                           n_floor=margins['hybrid_fr_gt_flag_fr']['n_floor'],
                           top10_share=margins['hybrid_fr_gt_flag_fr']['top10_share']),
              note='形态对比: 若均值由少数事件驱动的结构被 frailty 吸收 → median/正负占比应向正侧移动')
    out['verdict_G1'] = verdict
    out['G2_directional'] = g2
    out['G3_margin_shape'] = g3
    out['share_boost_fr'] = float(share_fr)
    print(f'G1: {verdict}', flush=True)
    print(f"G2: share_boost_fr={share_fr:.3f} (exp12 0.720) boost边际={d_boost:+.3f} (exp12 +2.318) "
          f"→ 预言{'支持' if g2['supported'] else ('不支持' if g2['supported'] is not None else '不适用(同构)')}",
          flush=True)

    # seen/unseen 分层
    train_dst = set(dst_all[:cut])
    seen = np.array([d in train_dst for d in dst_te])
    strat = {}
    for k in ['hybrid_fr', 'M_hybrid_W24_c', 'flag_fr', 'M_flag', 'boost_fr']:
        row = {}
        for nm, msk in [('seen', seen), ('unseen', ~seen)]:
            st = igpe(np.log2(lam[k][msk] / ref_store['uniform'][msk]))
            row[nm] = dict(n=int(msk.sum()), vs_uniform=st['IGPE_bits'], p=st['p'])
        strat[k] = row
        print(f"  {k}: seen {row['seen']['vs_uniform']:+.3f} unseen {row['unseen']['vs_uniform']:+.3f}", flush=True)
    out['D_stratified'] = strat

    # sanity 断言（防白做）
    for k in lam:
        assert np.all(np.isfinite(lam[k])) and np.all(lam[k] > 0), f'{k} λ 含 NaN/非正'
        assert abs(igpe(np.log2(lam[k] / ref_store['uniform']))['IGPE_bits']) < 30, f'{k} IGPE 量级异常'
    assert margins['hybrid_fr_gt_M_hybrid_W24_c']['n'] == 143

    json.dump(out, open(f'{R}/exp16_frailty_upgrade.json', 'w'), indent=1, ensure_ascii=False, default=float)
    print(f'落盘 {R}/exp16_frailty_upgrade.json; 总耗时 {(time.time()-t00)/60:.1f} min', flush=True)
    open(f'{R}/exp16.done', 'w').write('done: exp16 frailty 升级判定, 见 exp16_frailty_upgrade.json\n')


if __name__ == '__main__':
    main()
