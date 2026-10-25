#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exp22 无 frailty 同构对照偏倚方向复测（R3 §12.5 清单6 = review1 A-m4/m4 + 拔高记账 m4-m 清欠）
问题: exp16 的 hybrid_fr_unconstrained 块两端均含 frailty flag（只证约束未 binding, exp16
  R2_correction_20261021.hybrid_fr_unconstrained_label 在案）, 测不了「遗漏随机效应」的偏倚方向。
  文献先验三方互证: ①生存分析经典结果——遗漏 frailty→协变量效应向零衰减+幸存者选择;
  ②Filimonov-Sornette 1308.6756——真 n=0 被拟出 n̂≈1、恒定基线强加于异质数据→分支比拉向 1;
  ③癫痫 2605.22038——遗漏随机效应→低估背景、高估激发。
协议（预登记, 先落盘后运行; train 内计算, 不触测试集）:
  N1 真实 train 内同构对照（零重拟合）: exp12 权威拟合 M_boost_W24h_c（q=pooled pop, 无 frailty）
     vs M_hybrid_W24h_c（q=q_flag EB, 含 frailty）参数方向表——μ0 可比（两版 Σq=1, 背景总量=μ0·T）、
     分支比 a=κ·G_W 可比（每事件期望后代数, fit_constrained 注释: 与 mark 分布无关）;
     κ 原值**不可直接比**（乘在不同尖度的 mark 分布上）, 只落盘并注明。
  N2 合成已知 frailty 直接测偏倚: 真实图+真实 θ*（exp12 M_hybrid_W24h_c θ̂, a*=0.3753）+
     生成 mark 分布 q_gen∝q_flag^γ 归一化（γ=1/2/4 = 弱/中/强异质性档）, 分段 thinning 模拟
     事件流（m_h 跳变安全——见 N4）; 估计器 A=q_gen 拟合（正确设定, 参数恢复对照）、
     估计器 B=pooled pop 拟合（遗漏 frailty）→ (μ̂0,κ̂,â) 偏倚方向 vs 真值; 3 种子/档。
  N3 判定三预案（跑前预写, 不预设何者成立）:
     预案1（a 上偏+μ0 低估实测确认）→ 成文=「主机异质性被吸收进自激项与背景低估, 与 exp12
       boost/flag 分解互证; 分支比上偏=伪临界方向」;
     预案2（μ̂0/â 反向或近零）→ 成文=「本数据异质性弱（boost 集中约 1/3 事件）, 偏倚在
       n=570 下不可测」, 联动 q_flag EB 吸收与 q×hop0 共线混杂检查;
     预案3（混合）→ 成文=混合分解（μ0 与 a 各向何方）。
  N4 模拟器缺陷诊断（本轮实测发现, 必落盘）: src/lanl_pp.py simulate 的 thinning 上界
     M=μ0·m_h[hod(t)]+κ·lam_h(t) 在跨小时 m_h 跳升处不是区间包络（g 项递减但 m_h 跳升）→
     系统性漏事件——本机实测 3 种子 251/323/370 vs 理论 578.6（−45%）。本实验改用分段
     thinning（每小时一段: 段内 m_h 常数+g 递减→段首 M 为严格包络）, 脚本内自包含实现,
     **不修改已归档 src 语义**。已归档结论影响面=零: exp2 复原检验用 bg='const'（无 m_h
     跳变, scripts/exp2_recovery.py:40-42 实查）, 不触发该缺陷。
输出: results/exp22_nofrailty_contrast.json
"""
import sys, json, time
import numpy as np
sys.path.insert(0, './scripts')
sys.path.insert(0, './src')
from exp_patch_common import load_all, eb_flag_q, fit_constrained, R
from lanl_pp import GraphHawkes, hour_profile

GAMMAS = [1.0, 2.0, 4.0]
SEEDS = [101, 202, 303]


def simulate_segmented(m, th, T_end, seed, t_start=None):
    """分段 thinning 模拟（hop0+omori+hod24 专用）: 每小时一段, 段内 m_h 常数+g 递减→
    段首 M=μ0·m_h[h]+κ·lam_h 为严格包络; 段首含 age=0 右极限 g(0+)=c^{-p}（exp2 修复沿用）。
    段内接受事件后回 while 顶重算（lam_h 跳升→M 失效须更新）。"""
    rng = np.random.default_rng(seed)
    t = float(m.t0 if t_start is None else t_start)
    kap = float(np.exp(th[m.idx['log_kappa']]))
    mu0 = float(np.exp(th[m.idx['log_mu0']]))
    qV = np.array([m.q[m.cidx[d]] for d in m.V]); qV = qV / qV.sum()
    ev_t, ev_vi = [], []
    W = float(m.W)
    while t < T_end:
        h = int((t % 86400) // 3600)
        seg_end = (np.floor(t / 3600.0) + 1.0) * 3600.0
        lam_h = 0.0
        for ti in ev_t:
            dt = t - ti
            if 0 <= dt <= W:
                lam_h += float(m.g(th, max(dt, 1e-12)))
        M = mu0 * float(m.m_h[h]) + kap * lam_h + 1e-15
        w = rng.exponential(1.0 / M)
        if w >= seg_end - t or t + w >= T_end:
            t = min(seg_end, T_end)
            continue
        t2 = t + w
        lam2 = mu0 * float(m.m_h[int((t2 % 86400) // 3600)])
        for ti in ev_t:
            dt = t2 - ti
            if 0 < dt <= W:
                lam2 += kap * float(m.g(th, dt))
        if rng.random() * M <= lam2:
            comps = [mu0 * float(m.m_h[int((t2 % 86400) // 3600)])]; parents = []
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
                # hop0: 子事件恒回同主机（父节点自身）——Poff 语义的 hop0 退化形
                vi = ev_vi[parents[k - 1]]
            ev_t.append(t2); ev_vi.append(vi)
        t = t2
    return np.array(ev_t), [m.V[i] for i in ev_vi]


def main():
    t00 = time.time()
    D = load_all()
    adj, nodes, ev = D['adj'], D['nodes'], D['ev']
    t_all, dst_all, src_all = D['t_all'], D['dst_all'], D['src_all']
    cut, t_cut, V_all = D['cut'], D['t_cut'], D['V_all']
    t0v = float(t_all[0])
    m_h_full = hour_profile(t_all[:cut])
    cntC = np.zeros(len(nodes))
    cidxC = {c: i for i, c in enumerate(nodes)}
    for d in dst_all[:cut]:
        cntC[cidxC[d]] += 1
    popC = (cntC + 0.5) / (cntC + 0.5).sum()
    q_flag, alpha_eb, eb_diag = eb_flag_q(cntC)

    out = dict(exp='Exp22_nofrailty_contrast',
               preregistered=dict(
                   N1='真实 train 内同构对照: exp12 M_boost_W24h_c(无 frailty) vs M_hybrid_W24h_c(含 frailty) '
                      'μ0/a 方向表（κ 原值不可比只落盘）; 零重拟合',
                   N2='合成已知 frailty: θ*=exp12 hybrid θ̂, q_gen∝q_flag^γ（γ=1/2/4）, 分段 thinning 模拟, '
                      '估计器 A=q_gen（正确设定）/B=pooled pop（遗漏 frailty）, 3 种子/档',
                   N3_verdict_plans=dict(
                       plan1='a 上偏+μ0 低估实测确认→「异质性吸收进自激项+背景低估, exp12 互证; a 上偏=伪临界方向」',
                       plan2='反向/近零→「本数据异质性弱, n=570 不可测」, 联动 EB 吸收与 q×hop0 共线检查',
                       plan3='混合→混合分解如实写'),
                   N4='src simulate 漏采样缺陷诊断落盘（实测 −45%）; 本实验用分段 thinning 自实现; '
                      '已归档结论影响面=零（exp2 用 bg=const 不触发）',
                   no_test_touch='全部 train 内计算, 不触测试段'))
    json.dump(out, open(f'{R}/exp22_nofrailty_contrast.json', 'w'), indent=1, ensure_ascii=False, default=float)
    print('预登记已落盘 exp22_nofrailty_contrast.json', flush=True)

    def dump():
        json.dump(out, open(f'{R}/exp22_nofrailty_contrast.json', 'w'), indent=1,
                  ensure_ascii=False, default=float)

    # ---------- N1 真实 train 内对照（零重拟合, 读 exp12 权威） ----------
    j12 = json.load(open(f'{R}/exp12_boost_flag.json'))
    fr = j12['model_family']['M_hybrid_W24h_c']     # q_flag EB（含 frailty）
    nofr = j12['model_family']['M_boost_W24h_c']    # pooled pop（无 frailty）
    q_disp = np.log2(np.asarray(q_flag) / popC)
    out['N1_real_train_contrast'] = dict(
        with_frailty=dict(mu0=fr['params']['mu0'], kappa=fr['params']['kappa'],
                          branch_a=fr['branch_a'], ll=fr['train_ll'], c=fr['params']['c'], p=fr['params']['p']),
        no_frailty=dict(mu0=nofr['params']['mu0'], kappa=nofr['params']['kappa'],
                        branch_a=nofr['branch_a'], ll=nofr['train_ll'], c=nofr['params']['c'], p=nofr['params']['p']),
        ratios_nofr_over_fr=dict(
            mu0=nofr['params']['mu0'] / fr['params']['mu0'],
            branch_a=nofr['branch_a'] / fr['branch_a'],
            kappa_raw='不可直接比（乘在不同尖度 mark 分布上; hybrid κ 乘尖 q_flag, boost κ 乘平 pop）',
            ll_diff=nofr['train_ll'] - fr['train_ll']),
        q_flag_vs_pop_log2ratio=dict(
            mean=float(q_disp.mean()), median=float(np.median(q_disp)),
            q05=float(np.percentile(q_disp, 5)), q95=float(np.percentile(q_disp, 95)),
            min=float(q_disp.min()), max=float(q_disp.max()),
            n_hosts_above_2x=int((q_disp > 1).sum()), n_hosts_below_half=int((q_disp < -1).sum())),
        literature_prior='μ0 低估（<1）+分支比上偏（>1, 向 1/伪临界方向）=三方文献先验方向')
    dump()
    r_ = out['N1_real_train_contrast']['ratios_nofr_over_fr']
    print(f"N1: μ0 nofr/fr={r_['mu0']:.4f}, a nofr/fr={r_['branch_a']:.4f}, "
          f"Δll={r_['ll_diff']:+.1f}", flush=True)

    # ---------- N2 合成已知 frailty ----------
    m_host = GraphHawkes(t_all[:cut], list(dst_all[:cut]), list(src_all[:cut]), adj, nodes,
                         dst_extra=V_all, kernel='omori', gk='hop0', bg='hod24', W=86400.0,
                         qbg='custom', q_custom=q_flag)   # θ̂ 索引宿主（idx/vidx 借用）
    th_star = np.load(f'{R}/exp12_th_M_hybrid_W24h_c.npy')
    mu0_star = float(np.exp(th_star[m_host.idx['log_mu0']]))
    kap_star = float(np.exp(th_star[m_host.idx['log_kappa']]))
    a_star = kap_star * float(m_host.G_W(th_star, m_host.W))
    T_end = float(t_all[cut])

    # 模拟器正确性 sanity（N4 修复验证）: γ=1 档 3 种子事件数 vs 理论期望
    synth = {}
    for gam in GAMMAS:
        q_gen = np.power(np.asarray(q_flag, float), gam)
        q_gen = q_gen / q_gen.sum()
        m_gen = GraphHawkes(t_all[:cut], list(dst_all[:cut]), list(src_all[:cut]), adj, nodes,
                            dst_extra=V_all, kernel='omori', gk='hop0', bg='hod24', W=86400.0,
                            qbg='custom', q_custom=q_gen)   # 每档独立构造——γ 须真实进入生成过程
        rows = []
        for sd in SEEDS:
            ts, ds = simulate_segmented(m_gen, th_star, T_end, seed=sd)
            n_ev = len(ts)
            if n_ev < 50:
                rows.append(dict(seed=sd, n=n_ev, error='events<50, skip'))
                continue
            cnts = {}
            for dd in ds:
                cnts[dd] = cnts.get(dd, 0) + 1
            top10_share = float(sum(sorted(cnts.values())[::-1][:10]) / n_ev)
            # 估计器 A（正确设定）
            mA = GraphHawkes(list(ts), list(ds), ['?'] * n_ev, adj, nodes, dst_extra=V_all,
                             kernel='omori', gk='hop0', bg='hod24', W=86400.0,
                             qbg='custom', q_custom=q_gen)
            thA, llA, aA = fit_constrained(mA, a_max=0.95, n_restarts=4, seed=sd)
            # 估计器 B（遗漏 frailty: pooled pop——同 exp12 M_boost qbg='pop' 语义）
            mB = GraphHawkes(list(ts), list(ds), ['?'] * n_ev, adj, nodes, dst_extra=V_all,
                             kernel='omori', gk='hop0', bg='hod24', W=86400.0, qbg='pop')
            thB, llB, aB = fit_constrained(mB, a_max=0.95, n_restarts=4, seed=sd)
            row = dict(seed=sd, n=int(n_ev), sim_top10_host_share=top10_share,
                       est_A=dict(mu0=float(np.exp(thA[mA.idx['log_mu0']])),
                                  kappa=float(np.exp(thA[mA.idx['log_kappa']])), a=float(aA), ll=float(llA)),
                       est_B=dict(mu0=float(np.exp(thB[mB.idx['log_mu0']])),
                                  kappa=float(np.exp(thB[mB.idx['log_kappa']])), a=float(aB), ll=float(llB)),
                       bias_B_over_true=dict(mu0=float(np.exp(thB[mB.idx['log_mu0']]) / mu0_star),
                                             kappa=float(np.exp(thB[mB.idx['log_kappa']]) / kap_star),
                                             a=float(aB / a_star)),
                       recovery_A_over_true=dict(mu0=float(np.exp(thA[mA.idx['log_mu0']]) / mu0_star),
                                                 a=float(aA / a_star)))
            rows.append(row)
            print(f"  γ={gam} seed={sd}: n={n_ev} | B/true: μ0×{row['bias_B_over_true']['mu0']:.3f} "
                  f"a×{row['bias_B_over_true']['a']:.3f} | A/true: μ0×{row['recovery_A_over_true']['mu0']:.3f} "
                  f"a×{row['recovery_A_over_true']['a']:.3f}", flush=True)
        synth[f'gamma_{gam:g}'] = dict(q_sharpening_gam=gam,
                                       q_top10_mass=float(np.sort(q_gen)[::-1][:10].sum()),
                                       rows=rows)
        dump()

    # 汇总方向判定（跨档跨种子中位）
    def med(key):
        vals = []
        for g, blk in synth.items():
            for r in blk['rows']:
                if 'bias_B_over_true' in r:
                    vals.append(r['bias_B_over_true'][key])
        return float(np.median(vals)) if vals else float('nan')

    summ = dict(mu0_bias_B_median=med('mu0'), a_bias_B_median=med('a'), kappa_bias_B_median=med('kappa'))
    mu0_down = summ['mu0_bias_B_median'] < 0.98
    a_up = summ['a_bias_B_median'] > 1.02
    if mu0_down and a_up:
        plan = 'plan1'
        text = ('先验方向实测确认: 遗漏 frailty→背景低估×%s、分支比上偏×%s（伪临界方向）——'
                '主机异质性部分被吸收进自激项, 与 exp12 boost/flag 分解互证'
                % (f"{summ['mu0_bias_B_median']:.3f}", f"{summ['a_bias_B_median']:.3f}"))
    elif (summ['mu0_bias_B_median'] > 1.02) or (summ['a_bias_B_median'] < 0.98):
        plan = 'plan2'
        text = '方向与先验相反或近零——先查 q_flag EB 吸收与 q×hop0 共线混杂, 如实报（不预设）'
    else:
        plan = 'plan3'
        text = '混合方向——按 μ0 与 a 各自方向写混合分解'
    # γ 生效性检查：三档 top10 事件占比须随 γ 递增（防生成过程未随 γ 变化的无声失效——首轮 bug 教训:
    # simulate 宿主固定致三档数据相同, B 偏倚逐位相同暴露之）
    t10 = [float(np.median([r['sim_top10_host_share'] for r in synth[f'gamma_{g:g}']['rows']
                            if 'sim_top10_host_share' in r])) for g in GAMMAS]
    if not (t10[0] < t10[1] < t10[2]):
        raise RuntimeError(f'γ 生效性检查失败: top10 事件占比不随 γ 递增 {t10}')
    out['N2_synth'] = synth
    out['N2_verdict'] = dict(summary_median_B_over_true=summ, plan=plan, text=text,
                             gamma_effectiveness_top10_share_median=t10,
                             true_values=dict(mu0=mu0_star, kappa=kap_star, branch_a=a_star),
                             note='档间趋势: γ↑（异质性更强）偏倚应放大; 逐档数值见 N2_synth')
    print(f"N2 verdict: plan={plan} {summ}", flush=True)
    dump()

    json.dump(out, open(f'{R}/exp22_nofrailty_contrast.json', 'w'), indent=1,
              ensure_ascii=False, default=float)
    print(f'落盘 {R}/exp22_nofrailty_contrast.json; 总耗时 {(time.time()-t00)/60:.1f} min', flush=True)
    open(f'{R}/exp22.done', 'w').write('done: exp22 no-frailty 同构对照偏倚方向（真实 train+合成已知 frailty）\n')


if __name__ == '__main__':
    main()
