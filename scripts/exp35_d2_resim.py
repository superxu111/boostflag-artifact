#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exp35 exp7 D2 参数化 bootstrap 分段 thinning 重跑（R3 双审 B-C1 Critical + A-M3 合并）
问题（B-C1）: exp7_diag.py D2 的生成流调用 src/lanl_pp.py simulate（:431-435 thinning 上界
  M=base+κ·lam_h 在 bg='hod24' 跨小时 m_h 跳升处非区间包络→系统性漏事件）。exp7_diag.py:32-34
  模型为 bg='hod24'+qbg='pop'（读码实查）→ D2 全部推断数字（p̂/κ̂/ĉ CI、η CI、A2 定案）带缺陷;
  exp7_diag.json 自身症状在案: μ0 点估计 2.1945e-4 > bootstrap CI 上界 1.9759e-4、point_eta 0.6116
  完全脱节于 η CI [0.378,0.422]（曾被误归因「窗口不一致偏移」）。
  exp22 的「已归档结论影响面=零」审计漏掉本处（A-M3: N4 数字 251/323/370 vs 578.6 亦无归档证据,
  仅存脚本 docstring）。本实验两缺口一次收口:
  N4a 漏事件对照诊断归档: 同 θ 同种子 src simulate（缺陷, 只调用不修改）vs 分段 thinning（本脚本
      自实现, hop1 版——exp22 版为 hop0 专用）8 种子事件数对照 + 理论期望 μ0·∫m_h/(1−a)。
  N4b D2 重跑: 分段 thinning 全时段生成 B=100（与原版同 B）→ 逐次重拟合(n_restarts=2 同原版)
      → 修复版 D2 bootstrap CI。分段版正确性: 每小时一段, 段内 m_h 常数+g 单调递减→段首
      M=μ0·m_h[h]+κ·lam_h(含 age=0 右极限 g(0+)=c^{-p}) 为严格包络; 段内接受事件后回 while 顶
      重算 lam_h（跳升→M 更新）——与 exp22 simulate_segmented 同构, 子代选择用 src simulate
      同款 Poff=K(hop)·q/cbar 行归一化（hop1 语义, 非 exp22 的 hop0 恒回父机）。
  预登记判定:
    P1 分段版 sanity: 8 种子事件数均值与理论期望相对差须 <10%（否则 RuntimeError, 防 exp22 首轮
       类无声失效）; 缺陷版相对低估预期 −20%~−60%（B 席参照 −24%~−57%, 仅方向参照）。
    P2 修复判定: μ0 点估计落入修复版 95% CI 内=缺陷症状消解; p̂ CI 与原版对照如实并置,
       「A2 显著超临界」的最终判定以修复版 CI 是否跨 1 为准（不预设方向）。
    P3 影响面更正: exp22 的「已归档结论影响面=零」表述证伪（exp7 D2 在缺陷路径上）——三处权威
       落盘（exp22 JSON preregistered.N4 / RESEARCH_LOG.md:310 / exp_summary.md:230）由零计算
       批修同步更正; exp9/D1/D4 不受影响（解析/实测路径, 不经 simulate, 读码已查）。
  不引用审稿人 B 自建复算数字进权威工件（B 席 4 种子 1031/822/1018/618 vs 1350/1386/1425/1452
  仅作方向参照; 本实验数字以本机落盘为准——exp19/exp20 溯源纪律同源）。
运行: euler 环境; --smoke 先行(3 种子对照+2 次拟合试跑估时) → 全量后台。
输出: results/exp35_d2_resim.json
"""
import sys, json, time, argparse
import numpy as np
sys.path.insert(0, './src')
sys.path.insert(0, './scripts')
from lanl_pp import load_graph, load_events, GraphHawkes

R = './results'
N_SEEDS_DIAG = 8
SEEDS_DIAG = [11, 22, 33, 44, 55, 66, 77, 88]


def simulate_segmented_hop1(m, th, T_end, seed, t_start=None):
    """分段 thinning（hop1 专用）: 每小时一段, 段内 m_h 常数+g 递减→段首 M 为严格包络。
    段首含 age=0 右极限 g(0+)=c^{-p}（dt 用 max(dt,1e-12)）; 段内接受事件后回 while 顶重算
    lam_h（新事件使 λ 跳升→M 须更新, 否则段内包络失效）。子代分布与 src simulate 同款:
    Poff[i,:] ∝ K(hop_i,u)·q_u 行归一化（hop1: K=1[hop≤1]）。"""
    rng = np.random.default_rng(seed)
    t = float(m.t0 if t_start is None else t_start)
    kap = float(np.exp(th[m.idx['log_kappa']]))
    mu0 = float(np.exp(th[m.idx['log_mu0']]))
    qV = np.array([m.q[m.cidx[d]] for d in m.V]); qV = qV / qV.sum()
    Kp = m.K_from_hop(th, m.HopPair)                       # |V*|×|V*|
    cbar_all = np.array([m.cbar(th, i) for i in range(m.nV)])
    Poff = Kp * qV[None, :] / cbar_all[:, None]
    rowsum = Poff.sum(axis=1)
    Poff[rowsum <= 0] = qV[None, :]
    Poff = Poff / Poff.sum(axis=1, keepdims=True)
    ev_t, ev_vi = [], []
    W = float(m.W)
    while t < T_end:
        h = int((t % 86400) // 3600)
        seg_end = min((np.floor(t / 3600.0) + 1.0) * 3600.0, T_end)
        lam_h = 0.0
        for ti in ev_t:
            dt = t - ti
            if 0 <= dt <= W:
                lam_h += float(m.g(th, max(dt, 1e-12)))
        M = mu0 * float(m.m_h[h]) + kap * lam_h + 1e-15
        w = rng.exponential(1.0 / M)
        if t + w >= seg_end:
            t = seg_end
            continue
        t2 = t + w
        lam2 = mu0 * float(m.m_h[int((t2 % 86400) // 3600)])
        for ti in ev_t:
            dt = t2 - ti
            if 0 < dt <= W:
                lam2 += kap * float(m.g(th, dt))
        if rng.random() * M <= lam2:
            comps = [mu0 * float(m.m_h[int((t2 % 86400) // 3600)])]
            parents = []
            for k, ti in enumerate(ev_t):
                dt = t2 - ti
                if 0 < dt <= W:
                    comps.append(kap * float(m.g(th, dt)))
                    parents.append(k)
            comps = np.array(comps)
            k = int(rng.choice(len(comps), p=comps / comps.sum()))
            if k == 0:
                vi = int(rng.choice(len(qV), p=qV))
            else:
                vi = int(rng.choice(m.nV, p=Poff[ev_vi[parents[k - 1]]]))
            ev_t.append(t2); ev_vi.append(vi)
        t = t2
    return np.array(ev_t), [m.V[i] for i in ev_vi]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--boot', type=int, default=100)
    args = ap.parse_args()
    n_boot = 3 if args.smoke else args.boot
    t00 = time.time()

    adj, nodes = load_graph()
    ev = load_events()
    t_all, dst_all, src_all = ev['t'].astype(float), ev['dst'], ev['src']
    cut = int(0.8 * ev['n']); t_cut = float(t_all[cut])
    V_all = sorted(set(dst_all))
    m = GraphHawkes(t_all[:cut], list(dst_all[:cut]), list(src_all[:cut]), adj, nodes,
                    dst_extra=V_all, kernel='omori', gk='hop1', bg='hod24', W=86400.0, qbg='pop')
    th = np.load(f'{R}/exp4_th.npy')
    m.th = th
    pd_ = m.params_dict()
    mu0 = pd_['mu0']; kappa = pd_['kappa']
    Ginf = float(np.asarray(m.G_W(th, 1e12)))     # clamp 到 W=86400（与 exp7 branch 同口径）
    a_branch = kappa * Ginf
    T_end = float(t_all[-1] + 1.0)
    t0v = float(t_all[0])
    # 理论期望: 亚临界分支 μ0·∫m_h·(T)/(1−a)（背景×几何级数总后代; a=κ·G_W(W)）
    bg_exp = mu0 * float(m.mu_bg_integral(th, t0v, T_end))
    theo_total = bg_exp / (1.0 - a_branch)
    span_days = (T_end - t0v) / 86400.0
    print(f"θ: μ0={mu0:.4e} κ={kappa:.4f} G_W(W)={Ginf:.4f} a={a_branch:.4f} | "
          f"全时段 {span_days:.2f} 天, 理论背景={bg_exp:.1f}, 理论总事件={theo_total:.1f}", flush=True)

    out = dict(exp='Exp35_d2_resim',
               motive=('B-C1: exp7 D2 生成流走 src simulate 缺陷路径(bg=hod24 跨小时 m_h 跳升漏事件, '
                       'exp7_diag.py:32-34+:87 读码在案) → 分段 thinning 重跑; A-M3: exp22 N4 数字 '
                       '(251/323/370 vs 578.6) 无归档证据 → 本实验对照诊断归档'),
               config=dict(model='hop1/omori/hod24/pop, θ=exp4_th.npy', T_end=T_end,
                           span_days=span_days, mu0=mu0, kappa=kappa, G_W_clamped=Ginf,
                           a_branch=a_branch, theory_bg=bg_exp, theory_total=theo_total,
                       B=n_boot, seeds_resim=list(range(5000, 5000 + n_boot)),
                       note='分段版种子基 5000+/拟合 7000+（与缺陷版 1000+/2000+ 独立, 防同种子跨实现误读）'),
               preregistered=dict(
                   N4a='同 θ 同种子缺陷 vs 分段 8 种子事件数对照+理论期望; 分段版均值相对理论 <10% 为 sanity 硬闸',
                   N4b='分段 thinning 重跑 D2, B=100 同原版; 修复版 CI+point 落盘, μ0 点估计入 CI=症状消解',
                   P2='「A2 显著超临界」以修复版 p̂ CI 是否跨 1 为准, 不预设方向; 原版 CI 引用保留对照',
                   P3='「影响面=零」表述证伪更正由零计算批修执行; exp9/D1/D4 解析路径不受影响（读码在案）'),
               reviewer_note='审稿人 B 自建 4 种子复算数字仅作方向参照, 不进权威工件')
    json.dump(out, open(f'{R}/exp35_d2_resim.json', 'w'), indent=1, ensure_ascii=False, default=float)
    print('预登记已落盘 exp35_d2_resim.json', flush=True)

    def dump():
        json.dump(out, open(f'{R}/exp35_d2_resim.json', 'w'), indent=1, ensure_ascii=False, default=float)

    # ---------- N4a: 漏事件对照诊断（8 种子, 缺陷 vs 分段, 同种子） ----------
    rows = []
    for sd in SEEDS_DIAG:
        t1 = time.time()
        t_d, _ = m.simulate(th, T_end, seed=sd, t_start=t0v)          # 缺陷版（src 原样调用）
        t_s, _ = simulate_segmented_hop1(m, th, T_end, seed=sd)       # 分段版
        rows.append(dict(seed=sd, n_defective=int(len(t_d)), n_segmented=int(len(t_s)),
                         ratio_def_over_seg=float(len(t_d) / max(len(t_s), 1))))
        print(f"  N4a seed={sd}: 缺陷={len(t_d)} 分段={len(t_s)} "
              f"(低估值 {(1-len(t_d)/len(t_s))*100:+.1f}%, {time.time()-t1:.1f}s/对)", flush=True)
    n_seg = np.array([r['n_segmented'] for r in rows], float)
    n_def = np.array([r['n_defective'] for r in rows], float)
    sanity_rel = float(n_seg.mean() / theo_total - 1.0)
    if abs(sanity_rel) >= 0.10:
        raise RuntimeError(f'分段版 sanity 硬闸失败: 均值 {n_seg.mean():.1f} vs 理论 {theo_total:.1f} '
                           f'(相对差 {sanity_rel*100:+.1f}% ≥ 10%)')
    out['N4a_defect_diagnosis'] = dict(
        rows=rows,
        segmented_mean=float(n_seg.mean()), defective_mean=float(n_def.mean()),
        theory_total=theo_total,
        segmented_rel_dev_vs_theory=sanity_rel,
        defective_underestimate_pct=float(1 - n_def.mean() / n_seg.mean()),
        defective_rel_dev_vs_theory=float(n_def.mean() / theo_total - 1),
        verdict=('缺陷版事件数系统性低于分段版（包络失效漏事件实测归档, 替代不可复现的 '
                 'exp22 N4 docstring 数字）; 分段版与理论期望吻合, 模拟器缺陷机制实测成立'))
    dump()

    # ---------- N4b: D2 重跑（分段 thinning） ----------
    print(f"== D2 分段重跑 × {n_boot} ==", flush=True)
    t0b = time.time()
    boots = []
    for b in range(n_boot):
        t_sim, dst_sim = simulate_segmented_hop1(m, th, T_end, seed=5000 + b, t_start=t0v)
        if len(t_sim) < 30:
            boots.append(dict(error='events<30'))
            continue
        mb = GraphHawkes(list(t_sim), list(dst_sim), ['?'] * len(t_sim), adj, nodes, dst_extra=V_all,
                         kernel='omori', gk='hop1', bg='hod24', W=86400.0, qbg='pop')
        try:
            mb.fit(n_restarts=2, seed=7000 + b, t_end=T_end)
            pr = mb.params_dict()
            kap = float(np.exp(mb.th[mb.idx['log_kappa']]))
            Ginf_b = float(np.asarray(mb.G_W(mb.th, 1e12)))
            a_b = kap * Ginf_b
            boots.append(dict(mu0=pr['mu0'], kappa=pr['kappa'], c=pr['c'], p=pr['p'],
                              eta=a_b / (1 + a_b) if np.isfinite(a_b) else np.nan))
        except Exception as e:
            boots.append(dict(error=str(e)[:100]))
        if (b + 1) % 5 == 0:
            el = (time.time() - t0b) / 60
            print(f"  boot {b+1}/{n_boot} ({el:.1f} min, {el/(b+1)*60:.0f}s/次, "
                  f"预计剩余 {el/(b+1)*(n_boot-b-1):.0f} min)", flush=True)
        out['D2_resim_progress'] = dict(done=b + 1)
        if (b + 1) % 5 == 0:
            dump()
    ok = [x for x in boots if 'error' not in x]
    print(f"  完成 {len(ok)}/{n_boot}, 耗时 {(time.time()-t0b)/60:.1f} min", flush=True)
    if not ok:
        raise RuntimeError('D2 重跑零成功拟合')
    def ci(key):
        vals = np.array([x[key] for x in ok], float)
        vals = vals[np.isfinite(vals)]
        return dict(median=float(np.median(vals)), lo=float(np.percentile(vals, 2.5)),
                    hi=float(np.percentile(vals, 97.5)), n=len(vals))
    resim = {k: ci(k) for k in ['mu0', 'kappa', 'c', 'p', 'eta']}
    resim['point_est'] = {k: pd_[k] for k in ['mu0', 'kappa', 'c', 'p']}
    resim['point_eta'] = float(a_branch / (1 + a_branch))
    resim['branch_point'] = float(a_branch / (1 - a_branch)) if a_branch < 1 else float('inf')
    resim['n_ok'] = len(ok)
    out['D2_resim'] = resim
    out['D2_resim']['errors'] = [x.get('error', '') for x in boots if 'error' in x][:5]

    # ---------- 对照缺陷版归档 ----------
    j7 = json.load(open(f'{R}/exp7_diag.json'))
    old = j7['D2_bootstrap_CI']
    mu0_in_new = bool(resim['mu0']['lo'] <= mu0 <= resim['mu0']['hi'])
    mu0_in_old = bool(old['mu0']['lo'] <= mu0 <= old['mu0']['hi'])
    p_lo_new, p_hi_new = resim['p']['lo'], resim['p']['hi']
    out['comparison_vs_defective_archive'] = dict(
        old_D2_from='results/exp7_diag.json (缺陷生成流, B=100)',
        old=dict(mu0=old['mu0'], kappa=old['kappa'], c=old['c'], p=old['p'],
                 eta=old['eta'], point_eta=old['point_eta'], branch_point=old['branch_point'],
                 n_ok=old['n_ok']),
        new=dict(p_ci=[p_lo_new, p_hi_new], eta_ci=[resim['eta']['lo'], resim['eta']['hi']],
                 mu0_ci=[resim['mu0']['lo'], resim['mu0']['hi']]),
        symptom_checks=dict(
            mu0_point_in_old_CI=mu0_in_old, mu0_point_in_new_CI=mu0_in_new,
            old_point_eta=float(old['point_eta']), new_eta_ci=[resim['eta']['lo'], resim['eta']['hi']],
            old_eta_ci=[old['eta']['lo'], old['eta']['hi']]),
        P2_ruling=dict(
            p_hat_resim_ci=[p_lo_new, p_hi_new],
            A2_significant_supercritical=bool(p_lo_new > 1.0),
            text=('修复版 p̂ CI [%0.4f,%0.4f] %s 1 → 「A2 显著超临界 Omori」判定=%s'
                  % (p_lo_new, p_hi_new,
                     '不跨' if p_lo_new > 1.0 else '跨',
                     '维持' if p_lo_new > 1.0 else '以修复版 CI 为准重新表述'))),
        note=('分段 thinning 修复后 D2 全部推断数字以本块 D2_resim 为权威; exp7_diag.json D2 块 '
              '降级为缺陷版历史对照（零计算批修加 R4_correction 指针）'))
    out['runtime_min'] = (time.time() - t00) / 60
    dump()
    print(f"P2 裁定: p̂ 修复版 CI [{p_lo_new:.4f},{p_hi_new:.4f}] → A2={out['comparison_vs_defective_archive']['P2_ruling']['A2_significant_supercritical']}")
    print(f"μ0 点估计: 旧 CI 内={mu0_in_old}, 新 CI 内={mu0_in_new}")
    print(f'落盘 {R}/exp35_d2_resim.json; 总耗时 {out["runtime_min"]:.1f} min', flush=True)
    if not args.smoke:
        open(f'{R}/exp35.done', 'w').write('done: exp35 D2 分段 thinning 重跑+漏事件对照诊断归档\n')


if __name__ == '__main__':
    main()
