#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exp13 多步长前瞻因果协议重跑 + 技能曲线（审稿A/B-C2 / 方案二b）
修复: exp10_horizon.py 模型触发历史与 persistence 窗口均只用训练段（:60-61/:71）——测试段 16.07 天
  386 个逐小时格点中, 越过 t_cut+24h 后模型退化为纯背景, 越过 t_cut+H 后 persistence 退化为纯
  popularity 型——「两宇宙×三时距全胜」主要度量冻结历史下的静态判别优势, 非自激发前瞻价值。
本脚本（预登记于运行前）:
  P1 严格因果在线历史: 模型与 persistence 在每个格点 τ 均只见 t<τ 的全部事件（train+test 已发生）
  P2 基线补齐: persistence / pop(train 气候学) / uniform —— exp10 文档串承诺而 JSON 缺失
  P3 块bootstrap CI（6h 连续块×1000）: AP(model−persistence), AP(model−pop)
  P4 逐 τ 平均 log-loss 技能曲线 SS(τ)=1−LL_model/LL_baseline（气象学逐 lead 技能协议;
     SS 的 MSE 形式不照抄, log-loss 为主, 曲线仅作图）——1h 输给 persistence 属地震学/气象学
     「短 lead persistence 强」常态, 如实成文
  P5 targets_train 宇宙（train-only 目标, A-m4 泄漏修复）与原 targets/1hop 宇宙并列
  P6 Exp8b 校准同改因果历史重算 ECE（C5 修复）
"""
import sys, json, time
import numpy as np
sys.path.insert(0, './scripts')
sys.path.insert(0, './src')
from exp_patch_common import load_all, R
from lanl_pp import GraphHawkes, hour_profile
from sklearn.metrics import average_precision_score

RNG = np.random.default_rng(29)


def grid_scores_causal(m, th, U, t_all, dst_all, t_cut, T_end, H, causal=True):
    """逐 τ: 模型 ∫_τ^{τ+H} λ dt（因果或冻结历史）、persistence 计数、真值。返回 S_model,S_persist,Y"""
    uidx = {u: i for i, u in enumerate(U)}
    nU = len(U)
    qU = np.array([m.q[m.cidx[u]] for u in U])
    qU_idx = np.array([m.cidx[u] for u in U])
    KHopC = m.K_from_hop(th, m.HopC)
    KHopC_U = KHopC[:, qU_idx]          # 预切片（一次, 防 386×3×2 次重复 gather）
    cbar = np.array([m.cbar(th, i) for i in range(m.nV)])
    kap = float(np.exp(th[m.idx['log_kappa']]))
    mu0 = float(np.exp(th[m.idx['log_mu0']]))
    taus = np.arange(t_cut, T_end, 3600.0)
    S_model = np.zeros((len(taus), nU))
    S_persist = np.zeros((len(taus), nU))
    Y = np.zeros((len(taus), nU))
    tt_all = np.asarray(t_all, float)
    ds_all = np.asarray(dst_all)
    n_all = len(t_all)
    n_tr = int(np.searchsorted(tt_all, t_cut, side='left'))
    for gi, tau in enumerate(taus):
        hit = (tt_all >= tau) & (tt_all < tau + H)
        for d in ds_all[hit]:
            if d in uidx:          # targets_train 宇宙: 宇宙外(未见主机)攻击不计数
                Y[gi, uidx[d]] = 1
        # 模型: 背景 + 触发（历史=τ 前全部事件 或 仅 train）
        cov = m.mu_bg_integral(th, tau, tau + H)
        S_model[gi] = mu0 * cov * qU
        if causal:
            hist = np.where(tt_all < tau)[0]
        else:
            hist = np.arange(n_tr)
        if len(hist):
            dt = tau - tt_all[hist]
            okk = (dt > 0) & (dt <= m.W)
            h_idx = hist[okk]
            if len(h_idx):
                lo = np.maximum(tau - tt_all[h_idx], 0.0)
                hi = np.minimum(tau + H - tt_all[h_idx], m.W)
                GW = np.asarray(m.G_W(th, hi)) - np.asarray(m.G_W(th, lo))
                si_a = np.array([m.vidx[d] for d in ds_all[h_idx]])
                acc = np.zeros(nU)
                acc = (GW / cbar[si_a]) @ KHopC_U[si_a] * qU
                S_model[gi] += kap * acc
        # persistence: (τ−H, τ) 内被击主机（因果: 全部 t<τ 事件; 冻结: 仅 train）
        if causal:
            ph_idx = hist
        else:
            ph_idx = hist
        ph = (tt_all[ph_idx] >= tau - H) & (tt_all[ph_idx] < tau)
        for d in ds_all[ph_idx[ph]]:
            if d in uidx:          # 与 Y 同口径: 宇宙外主机不进 persistence 计数
                S_persist[gi, uidx[d]] += 1.0
        S_persist[gi] += qU   # 平滑底=popularity（与 exp10 冻结版语义一致: 1e-6·q·1e6）, 对 persistence 有利（保守）
    return taus, S_model, S_persist, Y


def block_boot_ap_diff(yf, mf, pf, n_tau, nU, block=6, B=1000):
    """块 bootstrap（τ 连续块重采样）AP 差 CI 与双侧 p; yf/mf/pf 为 (n_tau×nU) 展平, τ 级索引须扩到 cell 级"""
    def ap(y, s):
        return average_precision_score(y, s) if y.sum() > 0 else float('nan')
    obs = ap(yf, mf) - ap(yf, pf)
    nb = int(np.ceil(n_tau / block))
    cell_off = np.arange(nU)
    diffs = np.empty(B)
    for b in range(B):
        starts = RNG.integers(0, n_tau - block + 1, size=nb)
        tau_idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n_tau]
        idx = (tau_idx[:, None] * nU + cell_off[None, :]).ravel()
        yy, mm, pp = yf[idx], mf[idx], pf[idx]
        diffs[b] = ap(yy, mm) - ap(yy, pp)
    diffs = diffs[np.isfinite(diffs)]   # 重采样正例为空的 replicant 剔除
    if len(diffs) < 0.5 * B:
        raise RuntimeError(f'块bootstrap 有效 replicant 不足: {len(diffs)}/{B}')
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_two = float(min(1.0, 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())))
    return obs, float(lo), float(hi), p_two, len(diffs)


def logloss_curve(S, Y, eps=1e-12):
    P = 1.0 - np.exp(-np.maximum(S, 0.0))
    P = np.clip(P, eps, 1 - eps)
    ll = -(Y * np.log(P) + (1 - Y) * np.log(1 - P))
    return ll.mean(axis=1)   # 每 τ 平均 cell log-loss


def main():
    t00 = time.time()
    D = load_all()
    adj, nodes, ev = D['adj'], D['nodes'], D['ev']
    t_all, dst_all = D['t_all'], D['dst_all']
    cut, t_cut, V_all = D['cut'], D['t_cut'], D['V_all']
    T_end = float(t_all[-1])
    m = GraphHawkes(t_all[:cut], list(dst_all[:cut]), list(ev['src'][:cut]), adj, nodes,
                    dst_extra=V_all, kernel='omori', gk='hop1', bg='hod24', W=86400.0, qbg='pop')
    m.th = np.load(f'{R}/exp4_th.npy')
    cntC = np.zeros(m.nC)
    for d in dst_all[:cut]:
        cntC[m.cidx[d]] += 1
    popC = (cntC + 0.5) / (cntC + 0.5).sum()
    R0 = cut / (t_cut - t_all[0])

    V_tr = sorted(set(dst_all[:cut]))
    universes = {
        'targets': list(V_all),
        'targets_train': list(V_tr),
        'targets_1hop': sorted(set(V_all) | {u for v in V_all for u in adj.get(v, ())}),
    }
    out = dict(exp='Exp10R_causal_horizon', horizons_h=[1, 3, 6],
               protocol='因果在线历史(τ 前全部事件 train+test)同喂模型与 persistence; 块bootstrap(6h×1000) CI',
               note_vs_exp10='exp10_horizon.json 为冻结历史口径(模型与 persistence 均只用 train)——作废为'
                             '「冻结历史静态判别对照」; 本文件为权威因果协议版; 逐宇宙增量落盘',
               universes={})

    def dump():
        json.dump(out, open(f'{R}/exp13_causal_forecast.json', 'w'), indent=1, ensure_ascii=False, default=float)

    # ---------- P6: Exp8b 因果历史校准（前置, 最关键小宇宙先出） ----------
    print('== Exp8b 因果历史校准 ==', flush=True)
    U = universes['targets']
    taus, S_m, _, Y = grid_scores_causal(m, m.th, U, t_all, list(dst_all), t_cut, T_end, 3600.0, causal=True)
    Pprob = 1.0 - np.exp(-np.maximum(S_m, 0.0))
    pflat, yflat = Pprob.ravel(), Y.ravel()
    qs = np.quantile(pflat[pflat > 0], np.linspace(0, 1, 11)) if (pflat > 0).any() else np.array([0, 1])
    bins = np.digitize(pflat, qs)
    rel, ece = [], 0.0
    for b in range(1, 11):
        msk = bins == b
        if msk.sum() == 0:
            continue
        conf = float(pflat[msk].mean()); acc = float(yflat[msk].mean()); w = float(msk.mean())
        rel.append(dict(bin=b, n=int(msk.sum()), conf=conf, acc=acc, weight=w))
        ece += w * abs(conf - acc)
    out['exp8b_causal'] = dict(n_grid_points=int(len(taus)), universe=len(U),
                               positive_rate=float(yflat.mean()), n_positive=int(yflat.sum()),
                               max_pred_prob=float(pflat.max()), reliability=rel, ECE=float(ece),
                               note='因果历史口径; 冻结历史版见 exp8b_calibration.json(ECE=0.0007)')
    print(f"  因果 ECE={ece:.4f} max_p={pflat.max():.4f} 正例={int(yflat.sum())}", flush=True)
    dump()

    for uname, U in universes.items():
        uidx = {u: i for i, u in enumerate(U)}
        nU = len(U)
        causal_only = (nU > 1000)   # 1-hop 大宇宙: 仅因果口径点值(冻结版见 exp10_horizon.json), 跳过 bootstrap/技能曲线
        res = {}
        for H in [3600.0, 3 * 3600.0, 6 * 3600.0]:
            taus, S_m_c, S_p_c, Y = grid_scores_causal(m, m.th, U, t_all, list(dst_all), t_cut, T_end, H, causal=True)
            n_tau = len(taus)
            qU = np.array([m.q[m.cidx[u]] for u in U])
            S_pop = np.tile(R0 * qU, (n_tau, 1))
            S_uni = np.full((n_tau, nU), R0 / nU)
            yf = Y.ravel()
            ap = lambda s: float(average_precision_score(yf, s.ravel())) if yf.sum() > 0 else float('nan')
            r = dict(n_pred_points=int(n_tau), n_pos=int(yf.sum()), base_rate=float(yf.mean()),
                     PR_AUC_model_causal=ap(S_m_c), PR_AUC_persistence_causal=ap(S_p_c),
                     PR_AUC_pop_climatology=ap(S_pop), PR_AUC_uniform=ap(S_uni))
            if not causal_only:
                _, S_m_f, S_p_f, _ = grid_scores_causal(m, m.th, U, t_all, list(dst_all), t_cut, T_end, H, causal=False)
                r['PR_AUC_model_frozen'] = ap(S_m_f)
                r['PR_AUC_persistence_frozen'] = ap(S_p_f)
                # 块 bootstrap CI: model−persistence / model−pop（因果口径; 大宇宙降 B 防爆炸）
                Bb = 1000 if nU <= 1000 else 200
                d1 = block_boot_ap_diff(yf, S_m_c.ravel(), S_p_c.ravel(), n_tau, nU, B=Bb)
                d2 = block_boot_ap_diff(yf, S_m_c.ravel(), S_pop.ravel(), n_tau, nU, B=Bb)
                r['boot_model_minus_persistence'] = dict(diff=d1[0], ci=[d1[1], d1[2]], p_two=d1[3], n_valid=d1[4])
                r['boot_model_minus_pop'] = dict(diff=d2[0], ci=[d2[1], d2[2]], p_two=d2[3], n_valid=d2[4])
                # 逐 τ log-loss 技能曲线
                ll_m = logloss_curve(S_m_c, Y); ll_p = logloss_curve(S_p_c, Y); ll_pop = logloss_curve(S_pop, Y)
                with np.errstate(divide='ignore', invalid='ignore'):
                    ss_p = 1.0 - ll_m / ll_p
                    ss_pop = 1.0 - ll_m / ll_pop
                r['skill_curve'] = dict(
                    SS_vs_persistence_mean=float(np.nanmean(ss_p)),
                    SS_vs_pop_mean=float(np.nanmean(ss_pop)),
                    SS_vs_persistence_last48taus=[float(x) for x in ss_p[-48:]],
                    frac_taus_positive_SS_vs_persistence=float(np.nanmean(ss_p > 0)),
                    frac_taus_positive_SS_vs_pop=float(np.nanmean(ss_pop > 0)))
            else:
                r['note'] = '大宇宙(1-hop, |C|=15k): 仅因果口径点值; 冻结版对照见 exp10_horizon.json; bootstrap/技能曲线略(算力保护)'
            res[f'{int(H//3600)}h'] = r
            out['universes'][uname] = res
            dump()
            print(f"  [{uname} {int(H//3600)}h] 因果: model={r['PR_AUC_model_causal']:.4f} "
                  f"persist={r['PR_AUC_persistence_causal']:.4f} pop={r['PR_AUC_pop_climatology']:.4f}"
                  + (f" | 冻结: model={r['PR_AUC_model_frozen']:.4f} persist={r['PR_AUC_persistence_frozen']:.4f}"
                     f" | SS_persist均值={r['skill_curve']['SS_vs_persistence_mean']:+.3f}" if not causal_only else ""),
                  flush=True)

    json.dump(out, open(f'{R}/exp13_causal_forecast.json', 'w'), indent=1, ensure_ascii=False, default=float)
    print(f'落盘 {R}/exp13_causal_forecast.json; 总耗时 {(time.time()-t00)/60:.1f} min', flush=True)
    open(f'{R}/exp13.done', 'w').write('done\n')

if __name__ == '__main__':
    main()
