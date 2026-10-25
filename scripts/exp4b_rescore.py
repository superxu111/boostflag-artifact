#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exp4b 重评分（K̃=K·q/c̄ 模型一致口径）— 修复 exp34 主评分路径的核失配
背景: exp34 score_model 用 gw@Kmat(旧语义, 无 q/c̄), 似然拟合为 K̃ 语义 → 冻结 IGPE 被通胀(+8.90);
     C2/C4 路径(lambda_vec_C)才是模型一致(+4.81/+4.74 互证)。本脚本以 K̃ 口径重出全部模型侧指标。
内容: 主模型(hop1) vs 全部参考的 IGPE/T-test/排名/C3/D3/N-test + C1 候选集(active/train_active 重拟合)
     + decay 配置(AIC 最优)前瞻 IGPE + 逐事件 λ/参考向量落盘
"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, './src')
from lanl_pp import (load_graph, load_events, GraphHawkes, hour_profile,
                     igpe, rank_of_true)
from scipy import stats

R = './results'

def main():
    adj, nodes = load_graph()
    ev = load_events()
    t_all, dst_all, src_all = ev['t'].astype(float), ev['dst'], ev['src']
    cut = int(0.8 * ev['n']); t_cut = float(t_all[cut])
    V_all = sorted(set(dst_all))
    nC_all = len(nodes)
    R0 = cut / (t_cut - t_all[0])

    def build(kern='omori', gk='hop1', cand='all', cand_list=None, th=None, restarts=6):
        m = GraphHawkes(t_all[:cut], list(dst_all[:cut]), list(src_all[:cut]), adj, nodes,
                        dst_extra=V_all, kernel=kern, gk=gk, bg='hod24', W=86400.0,
                        qbg='pop', cand=cand, cand_list=cand_list)
        if th is None:
            m.fit(n_restarts=restarts, seed=11, t_end=t_cut)
        else:
            m.th = th
        return m

    # ---- K̃ 一致评分器 ----
    def score(m, refs_cfg, t_te, dst_te, src_te, t_h0, dst_h0, si_h0):
        th = m.th
        kap = float(np.exp(th[m.idx['log_kappa']]))
        KHopC = m.K_from_hop(th, m.HopC)          # nV×nC（参数相关核底板）
        q = m.q
        n_te = len(t_te)
        lam_at = np.zeros(n_te); ref_at = {k: np.zeros(n_te) for k in refs_cfg}
        lam_vecs = []; ref_vecs = {k: [] for k in refs_cfg}
        t_h = list(t_h0); dst_h = list(dst_h0); si_h = list(si_h0)
        Nv = np.zeros(m.nC)
        pop = REFC['pop']
        for j in range(n_te):
            t, v = float(t_te[j]), dst_te[j]
            lam = np.exp(th[m.idx['log_mu0']]) * m.mu_base(th, t) * q
            if t_h:
                dt = t - np.asarray(t_h)
                msk = (dt > 0) & (dt <= m.W)
                if msk.any():
                    si_a = np.asarray(si_h)[msk]
                    g = m.g(th, dt[msk])
                    gwtil = np.zeros(m.nV)
                    for k, i in enumerate(si_a):
                        gwtil[i] += g[k] / m.cbar(th, i)
                    lam += kap * (gwtil @ KHopC) * q
            iv = m.cidx[v]
            lam_at[j] = lam[iv]; lam_vecs.append(lam)
            for rk in refs_cfg:
                rv = ref_vec(rk, t, t_h, dst_h, si_h, src_te[j], m.C, m.cidx, pop, Nv, m.HopC)
                ref_at[rk][j] = rv[iv]; ref_vecs[rk].append(rv)
            t_h.append(t); dst_h.append(v); si_h.append(m.vidx[v])
            if v in m.cidx:
                Nv[m.cidx[v]] += 1
        return lam_at, lam_vecs, ref_at, ref_vecs

    def ref_vec(kind, t, t_h, dst_h, si_h, src, C, cidx, pop, Nv, HopC):
        nC = len(C)
        if kind == 'uniform':
            return np.full(nC, R0 / nC)
        if kind == 'pop':
            return R0 * pop
        if kind == 'hod_pop':
            return R0 * m_h[int((t % 86400) // 3600)] * pop
        if kind == 'edgecount':
            if src in ECNT['cnt']:
                w = np.array([ECNT['cnt'][src].get(c, 0) for c in C], float) + 0.5 * nC * pop
                return R0 * w / (ECNT['tot'][src] + 0.5 * nC)
            return R0 * pop
        if kind == 'recency1':
            tau = 3600.0
        elif kind == 'recency24':
            tau = 86400.0
        if kind in ('recency1', 'recency24'):
            lam = np.full(nC, 1e-3)
            for tt, d in zip(t_h, dst_h):
                dt = t - tt
                if 0 < dt <= 48 * 3600:
                    lam[cidx[d]] += np.exp(-dt / tau) / tau
            return R0 * lam / lam.sum()
        if kind == 'knox_rule':
            base = np.full(nC, 1.0 / nC)
            tt = np.asarray(t_h, float)
            if len(tt):
                rec = (t - tt > 0) & (t - tt <= 600.0)
                if rec.any():
                    hot = (HopC[np.asarray(si_h)[rec]] <= 1).any(axis=0)
                    base[hot] *= 10.0
            return R0 * base / base.sum()
        if kind == 'iw':
            w = pop * np.exp(-REFC['theta_iw'] * Nv)
            return R0 * w / w.sum()
        raise ValueError(kind)

    m_h = hour_profile(t_all[:cut])
    ECNT = {}
    def build_ecount(C):
        srcs = sorted(set(src_all[:cut]))
        maps = {s: {} for s in srcs}; tot = {s: 0 for s in srcs}
        for tt, s, d in zip(t_all[:cut], src_all[:cut], dst_all[:cut]):
            maps[s][d] = maps[s].get(d, 0) + 1
            tot[s] += 1
        ECNT['cnt'], ECNT['tot'] = maps, tot

    def fit_iw(C, cidx, pop):
        def neg(tv):
            th_ = float(np.asarray(tv).ravel()[0])
            Nv = np.zeros(len(C)); ll = 0.0
            for j in range(cut):
                iv = cidx[dst_all[j]]
                lam = R0 * pop[iv] * np.exp(-th_ * Nv[iv])
                ll += np.log(max(lam, 1e-300))
                Nv[iv] += 1
            Nv = np.zeros(len(C)); tot = 0.0
            for j in range(cut):
                seg = (t_all[j + 1] if j + 1 < cut else t_all[cut - 1] + 1.0) - t_all[j]
                tot += float(pop @ np.exp(-th_ * Nv)) * seg
                Nv[cidx[dst_all[j]]] += 1
            return -(ll - R0 * tot)
        grid = np.linspace(0, 3, 61)
        b = grid[int(np.argmin([neg(g) for g in grid]))]
        from scipy.optimize import minimize_scalar
        return float(minimize_scalar(neg, bounds=(max(0, b - 0.1), b + 0.1), method='bounded').x)

    t_te, dst_te, src_te = t_all[cut:], [dst_all[i] for i in range(cut, ev['n'])], \
        [src_all[i] for i in range(cut, ev['n'])]
    out = dict(exp='Exp4b_rescore_Ktilde', note='K̃=K·q/c̄ 模型一致口径重评分; 修复 exp34 核失配')

    # ---- 主模型 hop1（载入确定性 θ̂） ----
    m = build(th=np.load(f'{R}/exp4_th.npy'))
    cntC = np.zeros(m.nC)
    for d in dst_all[:cut]:
        cntC[m.cidx[d]] += 1
    popC = (cntC + 0.5) / (cntC + 0.5).sum()
    build_ecount(m.C)
    REFC = dict(pop=popC, theta_iw=fit_iw(m.C, m.cidx, popC))
    print(f"IW θ̂={REFC['theta_iw']:.3f}", flush=True)
    si_h0 = [m.vidx[d] for d in dst_all[:cut]]
    refs = ['uniform', 'pop', 'hod_pop', 'edgecount', 'recency1', 'recency24', 'knox_rule', 'iw']
    la, lv, ra, rv = score(m, refs, t_te, dst_te, src_te, list(t_all[:cut]), list(dst_all[:cut]), si_h0)
    res = dict(params=m.params_dict(), n_par=m.n_par)
    gains_all = {}
    for rk in refs:
        g = np.log2(np.asarray(la) / np.asarray(ra[rk]))
        st = igpe(g); gains_all[rk] = g
        ranks_m = []; ranks_p = []; top1 = 0; top5 = 0
        for j in range(len(t_te)):
            sc = np.log2(lv[j] / np.maximum(rv[rk][j], 1e-300))
            _, rm = rank_of_true(sc, m.cidx[dst_te[j]], 'mid')
            _, rp = rank_of_true(sc, m.cidx[dst_te[j]], 'pess')
            ranks_m.append(rm); ranks_p.append(rp)
            top1 += int(rm == 1.0); top5 += int(rm <= 5.0)
        st.update(rank_mid_mean=float(np.mean(ranks_m)), rank_pess_mean=float(np.mean(ranks_p)),
                  top1=top1 / len(t_te), top5=top5 / len(t_te),
                  mean_log2lam=float(np.mean(np.log2(np.maximum(la, 1e-300)))),
                  mean_log2prob=float(np.mean([np.log2(l[m.cidx[d]] / l.sum()) for l, d in zip(lv, dst_te)])))
        res[f'vs_{rk}'] = st
        np.save(f'{R}/exp4_gains_model_vs_{rk}.npy', g)
        print(f"  model vs {rk:10s}: IGPE={st['IGPE_bits']:+.4f} t={st['t']:+.2f} p={st['p']:.2e} "
              f"rank_mid={st['rank_mid_mean']:.0f} top1={st['top1']:.3f}", flush=True)
    # 参考阶梯内
    for a, b in [('hod_pop', 'uniform'), ('edgecount', 'uniform'), ('pop', 'uniform'),
                 ('knox_rule', 'uniform'), ('recency1', 'uniform'), ('iw', 'uniform')]:
        st = igpe(gains_all[a] - gains_all[b])
        res[f'{a}_vs_{b}'] = st
        print(f"  {a} vs {b}: IGPE={st['IGPE_bits']:+.4f} p={st['p']:.2e}", flush=True)
    # C3 累积
    cum = np.cumsum(gains_all['uniform'])
    res['C3_cumulative'] = dict(final=float(cum[-1]), first_negative=int(np.argmax(cum < 0)) if (cum < 0).any() else -1)
    # D3 MDE
    g = gains_all['uniform']
    res['D3_power'] = dict(MDE_bits=float((stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) * g.std(ddof=1) / np.sqrt(len(g))))
    # N-test（解析积分, K̃ 一致）
    n_pred = m.Lambda_integral(m.th, t_cut, t_all[-1] + 1.0, list(t_all[:cut]), si_h0)
    res['N_test'] = dict(expected=float(n_pred), observed=len(t_te),
                         poisson_p=float(stats.poisson.sf(len(t_te) - 1, n_pred)))
    print(f"  N-test: 预测={n_pred:.1f} vs 观测={len(t_te)}", flush=True)
    # 分层（seen/unseen）
    train_dst = set(dst_all[:cut])
    seen = np.array([d in train_dst for d in dst_te])
    for nm, msk in [('seen', seen), ('unseen', ~seen)]:
        res[f'strat_{nm}'] = dict(n=int(msk.sum()),
                                  IGPE_vs_uniform=igpe(gains_all['uniform'][msk])['IGPE_bits'],
                                  p_uniform=igpe(gains_all['uniform'][msk])['p'],
                                  IGPE_vs_pop=igpe(gains_all['pop'][msk])['IGPE_bits'],
                                  p_pop=igpe(gains_all['pop'][msk])['p'])
        print(f"  [{nm} n={int(msk.sum())}] vs uni={res[f'strat_{nm}']['IGPE_vs_uniform']:+.3f} "
              f"vs pop={res[f'strat_{nm}']['IGPE_vs_pop']:+.3f}", flush=True)
    out['main_hop1'] = res

    # ---- decay 配置（AIC 最优）前瞻 ----
    m_d = build(kern='omori', gk='decay', restarts=6)
    cntD = np.zeros(m_d.nC)
    for d in dst_all[:cut]:
        cntD[m_d.cidx[d]] += 1
    popD = (cntD + 0.5) / (cntD + 0.5).sum()
    REFC['pop'] = popD
    la_d, lv_d, ra_d, rv_d = score(m_d, ['uniform', 'pop'], t_te, dst_te, src_te,
                                   list(t_all[:cut]), list(dst_all[:cut]),
                                   [m_d.vidx[d] for d in dst_all[:cut]])
    out['decay_AICbest'] = dict(params=m_d.params_dict(), train_ll=m_d.ll_max,
                                IGPE_vs_uniform=igpe(np.log2(la_d / ra_d['uniform']))['IGPE_bits'],
                                IGPE_vs_pop=igpe(np.log2(la_d / ra_d['pop']))['IGPE_bits'],
                                p_uniform=igpe(np.log2(la_d / ra_d['uniform']))['p'])
    print(f"  decay 配置: vs uni={out['decay_AICbest']['IGPE_vs_uniform']:+.3f} "
          f"vs pop={out['decay_AICbest']['IGPE_vs_pop']:+.3f} (train_ll={m_d.ll_max:.1f})", flush=True)
    REFC['pop'] = popC

    # ---- C1 候选集（重拟合） ----
    for cname, kwargs in [('active', dict(cand_list=None, cand='active')),
                          ('targets', dict(cand_list=list(V_all)))]:
        m_c = build(**kwargs, restarts=4)
        cntX = np.zeros(m_c.nC)
        for d in dst_all[:cut]:
            if d in m_c.cidx:
                cntX[m_c.cidx[d]] += 1
        popX = (cntX + 0.5) / (cntX + 0.5).sum()
        REFC['pop'] = popX
        keep = [j for j, d in enumerate(dst_te) if d in m_c.cidx]
        la_c, lv_c, ra_c, rv_c = score(m_c, ['uniform', 'pop'],
                                       [t_te[j] for j in keep], [dst_te[j] for j in keep],
                                       [src_te[j] for j in keep],
                                       list(t_all[:cut]), list(dst_all[:cut]),
                                       [m_c.vidx[d] for d in dst_all[:cut]])
        entry = dict(cand_size=m_c.nC, n_scored=len(keep))
        for rk in ['uniform', 'pop']:
            g_ = np.log2(la_c / ra_c[rk])
            st = igpe(g_)
            entry[f'vs_{rk}'] = dict(IGPE=st['IGPE_bits'], p=st['p'])
        out[f'C1_{cname}'] = entry
        print(f"  C1[{cname} |C|={m_c.nC} n={len(keep)}]: vs uni={entry['vs_uniform']['IGPE']:+.3f} "
              f"vs pop={entry['vs_pop']['IGPE']:+.3f}", flush=True)
    REFC['pop'] = popC

    json.dump(out, open(f'{R}/exp4b_rescored.json', 'w'), indent=1, ensure_ascii=False, default=float)
    print(f"落盘 {R}/exp4b_rescored.json", flush=True)
    open(f'{R}/exp4b.done', 'w').write('done')

if __name__ == '__main__':
    main()
