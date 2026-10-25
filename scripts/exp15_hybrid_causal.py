#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exp15 M-hybrid（q_flag×hop0）严格因果前瞻 + 增强基线组（R2 主题①；审稿A-M4/审稿B-C2 收口）
动机: exp13 因果协议用 gk='hop1'+exp4_th 冻结参数（exp13_causal_forecast.py:121-123）,
  而 exp12 机制判定旗舰=M-hybrid（q_flag 背景+hop0 自激, exp12_th_M_hybrid_W24h_c.npy）——
  两旗舰实验模型身份不一致（审稿A-M4）。本实验把 M-hybrid 喂入同一严格因果协议, 统一身份;
  1h 是否反超 persistence 双向成文（H2）。
协议（与 exp13 逐位同构, 除注明处）:
  P1 严格因果在线历史: 模型与全部计数型基线在每个格点 τ 只见 t<τ 的全部事件（train+test 已发生）
  P2 基线组: persistence(+pop 底, exp13 原语义) / BL-a persistence×hod / BL-b persistence×q_flag /
     BL-c Markov dst 一阶转移(train 转移计数+popularity 平滑回退; H 小时前瞻=单步分布近似, 预期退化如实报)
     / pop 气候学 / uniform
  P3 块 bootstrap CI（6h 连续 τ 块×1000; 与 exp13 同 block 同 B; 敏感性窗 B=500）:
     model−persistence（主判定）、model−BL-a、model−BL-b、model−BL-c
  P4 技能曲线重设计（替换 exp13 SS_vs_pop 病态列 −7517, 审稿A-M2/整合轮 8.1）:
     逐 τ IGN(log2 proper log score, 泊松转换) vs pop 气候学 → ISS=1−IGN_model/IGN_clim;
     全 τ 与「有正例 τ」子集双报 + IGN 差值(model−clim)分位数(5/25/50/75/95) —— 差值口径免比值病态
  P5 seen/unseen 分层 + 前/后半(测试段 τ 对半)分层 AP（H3/H7; 非平稳敏感性）
  P6 主判定多重比较: model−persistence 6 检验（2 宇宙×3 时距）×6 Bonferroni + BH 复核
     （对齐 exp13 权威口径: 仅 targets_train 6h 存活的先例）
模型: M-hybrid = hod24·q_flag(Gamma-Poisson EB 矩法 α=0.0095, exp12 同款)背景 + hop0 自激 Omori 核
  主版 W=24h 约束拟合(exp12 判定主版 θ̂); 敏感性 W∈{12,48}h 约束版
预登记成败判据（运行前落盘）:
  J1 主判定=model−persistence ×6 Bonferroni 后存活格数与 exp13 hop1 版（仅 targets_train 6h）对比;
  J2 1h 是否反超 persistence 双向成文: 反超=机制判定前瞻面闭合; 仍输=「自激成分 1h 未转化为因果排序优势」如实写;
  J3 BL-a 若吃掉 model−persistence 显著性 → 「模型赢在背景而非触发」; 若不吃掉 → 衰减形状有独立贡献。
正确性检查（运行前置）: 复制的评分函数先用 hop1+exp4_th.npy 复算 exp13 targets 6h 两口径,
  与权威 JSON 差须 <1e-9（协议复制无漂移证据）才继续。
"""
import sys, json, time, argparse
import numpy as np
sys.path.insert(0, './scripts')
sys.path.insert(0, './src')
from exp_patch_common import load_all, eb_flag_q, R
from lanl_pp import GraphHawkes, hour_profile
from exp13_causal_forecast import block_boot_ap_diff  # 与 exp13 同款 bootstrap 实现(同 seed 起点)
from sklearn.metrics import average_precision_score

RNG = np.random.default_rng(29)


def grid_scores_causal_v2(m, th, U, t_all, dst_all, t_cut, T_end, H, causal=True,
                          q_popvec=None):
    """复制自 exp13_causal_forecast.grid_scores_causal（协议逐位一致）, 唯一改动:
    拆分「模型背景 mark 分布 q_bg」(=m.q, M-hybrid 为 q_flag) 与「基线 pop 底 q_popvec」
    (persistence/pop 基线的平滑底; exp13 中两者恰同为 popularity 故未拆)。
    返回 taus, S_model, S_persist, Y（S_persist=counts+q_popvec 底, exp13 原语义）。"""
    uidx = {u: i for i, u in enumerate(U)}
    nU = len(U)
    qU = np.array([m.q[m.cidx[u]] for u in U])          # 模型背景=q_flag（M-hybrid）
    qP = (np.array([q_popvec[m.cidx[u]] for u in U]) if q_popvec is not None else qU)
    qU_idx = np.array([m.cidx[u] for u in U])
    KHopC = m.K_from_hop(th, m.HopC)
    KHopC_U = KHopC[:, qU_idx]
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
            if d in uidx:
                Y[gi, uidx[d]] = 1
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
        ph = (tt_all[hist] >= tau - H) & (tt_all[hist] < tau)
        for d in ds_all[hist[ph]]:
            if d in uidx:
                S_persist[gi, uidx[d]] += 1.0
        S_persist[gi] += qP
    return taus, S_model, S_persist, Y


def ap_of(yf, s):
    return float(average_precision_score(yf, s.ravel())) if yf.sum() > 0 else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    t00 = time.time()
    D = load_all()
    adj, nodes, ev = D['adj'], D['nodes'], D['ev']
    t_all, dst_all, src_all = D['t_all'], D['dst_all'], D['src_all']
    cut, t_cut, V_all = D['cut'], D['t_cut'], D['V_all']
    T_end = float(t_all[-1])
    m_h = hour_profile(t_all[:cut])
    R0 = cut / (t_cut - t_all[0])
    cntC = np.zeros(len(nodes))
    cidxC = {c: i for i, c in enumerate(nodes)}
    for d in dst_all[:cut]:
        cntC[cidxC[d]] += 1
    popC = (cntC + 0.5) / (cntC + 0.5).sum()
    q_flag, alpha_eb, eb_diag = eb_flag_q(cntC)

    # BL-c Markov dst 一阶转移（train-only 转移表 + popularity 平滑回退, λ=1）
    trans_cnt, trans_tot = {}, {}
    for i in range(cut - 1):
        a, b = dst_all[i], dst_all[i + 1]
        trans_cnt.setdefault(a, {})
        trans_cnt[a][b] = trans_cnt[a].get(b, 0) + 1
        trans_tot[a] = trans_tot.get(a, 0) + 1
    lam_sm = 1.0
    pop_nodes = np.zeros(len(nodes))
    for c, i in cidxC.items():
        pop_nodes[i] = popC[i]

    def markov_vec(last):
        """全节点一阶转移分布（含 popularity 平滑）; last 未知→pop 回退"""
        base = pop_nodes.copy()
        if last is not None and last in trans_tot:
            w = np.array([trans_cnt[last].get(c, 0) for c in nodes], float)
            base = (w + lam_sm * pop_nodes) / (trans_tot[last] + lam_sm)
        return base

    V_tr = sorted(set(dst_all[:cut]))
    universes = {'targets': list(V_all), 'targets_train': list(V_tr)}

    # ---------- 模型构建 ----------
    def build(W):
        return GraphHawkes(t_all[:cut], list(dst_all[:cut]), list(src_all[:cut]), adj, nodes,
                           dst_extra=V_all, kernel='omori', gk='hop0', bg='hod24', W=W,
                           qbg='custom', q_custom=q_flag)

    windows = {'24h': 86400.0} if args.smoke else {'12h': 43200.0, '24h': 86400.0, '48h': 172800.0}
    models = {}
    for wn, W in windows.items():
        m = build(W)
        m.th = np.load(f'{R}/exp12_th_M_hybrid_W{wn}_c.npy')
        models[wn] = m
    # 正确性前置检查: hop1+exp4_th 复算 exp13 targets 6h, 须与权威 JSON 差 <1e-9
    m_ref = GraphHawkes(t_all[:cut], list(dst_all[:cut]), list(ev['src'][:cut]), adj, nodes,
                        dst_extra=V_all, kernel='omori', gk='hop1', bg='hod24', W=86400.0, qbg='pop')
    m_ref.th = np.load(f'{R}/exp4_th.npy')
    _, S_m_chk, S_p_chk, _ = grid_scores_causal_v2(
        m_ref, m_ref.th, universes['targets'], t_all, list(dst_all), t_cut, T_end,
        6 * 3600.0, causal=True, q_popvec=popC)
    from exp13_causal_forecast import grid_scores_causal as g13
    _, S_m_o, S_p_o, _ = g13(m_ref, m_ref.th, universes['targets'], t_all, list(dst_all),
                             t_cut, T_end, 6 * 3600.0, causal=True)
    dchk = float(np.abs(S_m_chk - S_m_o).max() + np.abs(S_p_chk - S_p_o).max())
    # 与权威 JSON 对表（Y 矩阵重算）
    j13 = json.load(open(f'{R}/exp13_causal_forecast.json'))
    chk = dict(fn_replication_maxdiff=dchk)
    _, _, _, Y_chk = grid_scores_causal_v2(m_ref, m_ref.th, universes['targets'], t_all,
                                           list(dst_all), t_cut, T_end, 6 * 3600.0, causal=True, q_popvec=popC)
    chk['ap_model_vs_json'] = dict(mine=ap_of(Y_chk.ravel(), S_m_chk.ravel()),
                                   json=j13['universes']['targets']['6h']['PR_AUC_model_causal'])
    chk['ap_persist_vs_json'] = dict(mine=ap_of(Y_chk.ravel(), S_p_chk.ravel()),
                                     json=j13['universes']['targets']['6h']['PR_AUC_persistence_causal'])
    chk['pass'] = bool(dchk < 1e-9 and
                       abs(chk['ap_model_vs_json']['mine'] - chk['ap_model_vs_json']['json']) < 1e-9 and
                       abs(chk['ap_persist_vs_json']['mine'] - chk['ap_persist_vs_json']['json']) < 1e-9)
    print('== 前置正确性检查（复制函数 vs exp13 权威） ==', flush=True)
    print(json.dumps(chk, indent=1, default=float), flush=True)
    if not chk['pass']:
        raise RuntimeError('复制评分函数与 exp13 权威不一致, 终止（防协议漂移）')

    out = dict(exp='Exp15_hybrid_causal_forecast',
               model='M-hybrid (q_flag EB × hop0 Omori, exp12_th_M_hybrid_W24h_c 主版; W12/W48 敏感性)',
               preregistered=dict(
                   J1='主判定=model−persistence ×6 Bonferroni 存活格数 vs exp13 hop1 版(仅 targets_train 6h)',
                   J2='1h 反超 persistence 双向成文',
                   J3='BL-a 吃掉显著性→赢在背景; 不吃掉→衰减形状有独立贡献'),
               protocol='严格因果在线历史; 块bootstrap 6h×1000(主)/500(敏感性窗); '
                        'BL-a=persistence×hod, BL-b=persistence×q_flag, BL-c=Markov一阶转移(λ=1 pop平滑)',
               correctness_check=chk, universes={})

    def dump():
        json.dump(out, open(f'{R}/exp15_hybrid_causal.json', 'w'), indent=1, ensure_ascii=False, default=float)

    Hs = [6 * 3600.0] if args.smoke else [3600.0, 3 * 3600.0, 6 * 3600.0]
    unis = {'targets': universes['targets']} if args.smoke else universes
    boot_main = 100 if args.smoke else 1000
    boot_sens = 50 if args.smoke else 500

    for uname, U in unis.items():
        uidx = {u: i for i, u in enumerate(U)}
        nU = len(U)
        qflagU = np.array([q_flag[cidxC[u]] for u in U]); qflagU = qflagU / qflagU.sum()
        qpopU = np.array([popC[cidxC[u]] for u in U]); qpopU = qpopU / qpopU.sum()
        seen_mask_U = np.array([u in set(V_tr) for u in U])
        res = {}
        for wn, m in models.items():
            for H in Hs:
                taus, S_m, S_p, Y = grid_scores_causal_v2(
                    m, m.th, U, t_all, list(dst_all), t_cut, T_end, H, causal=True, q_popvec=popC)
                n_tau = len(taus)
                # 因果在线计数型基线
                S_bla = np.zeros_like(S_p)   # persistence×hod
                S_blb = np.zeros_like(S_p)   # persistence×q_flag
                S_blm = np.zeros_like(S_p)   # Markov dst 一阶
                ds_all = np.asarray(dst_all)
                tt_all = np.asarray(t_all, float)
                for gi, tau in enumerate(taus):
                    hist = np.where(tt_all < tau)[0]
                    counts = np.zeros(nU)
                    for d in ds_all[hist[(tt_all[hist] >= tau - H) & (tt_all[hist] < tau)]]:
                        if d in uidx:
                            counts[uidx[d]] += 1.0
                    hod = int((tau % 86400) // 3600)
                    S_bla[gi] = m_h[hod] * (counts + qpopU)
                    S_blb[gi] = qflagU * (counts + R0)
                    last = ds_all[hist[-1]] if len(hist) else None
                    mv = markov_vec(last)
                    S_blm[gi] = R0 * np.array([mv[cidxC[u]] for u in U])
                S_pop = np.tile(R0 * qpopU, (n_tau, 1))
                S_uni = np.full((n_tau, nU), R0 / nU)
                yf = Y.ravel()
                r = dict(n_pred_points=int(n_tau), n_pos=int(yf.sum()), base_rate=float(yf.mean()),
                         PR_AUC_model=ap_of(yf, S_m), PR_AUC_persistence=ap_of(yf, S_p),
                         PR_AUC_persist_x_hod=ap_of(yf, S_bla), PR_AUC_persist_x_qflag=ap_of(yf, S_blb),
                         PR_AUC_markov_dst1=ap_of(yf, S_blm),
                         PR_AUC_pop_climatology=ap_of(yf, S_pop), PR_AUC_uniform=ap_of(yf, S_uni))
                mainwin = (wn == '24h')
                Bb = boot_main if mainwin else boot_sens
                for bname, sv in [('model_minus_persistence', S_p), ('model_minus_BLa', S_bla),
                                  ('model_minus_BLb', S_blb), ('model_minus_BLc', S_blm)]:
                    if (not mainwin) and bname != 'model_minus_persistence':
                        continue
                    d1 = block_boot_ap_diff(yf, S_m.ravel(), sv.ravel(), n_tau, nU, B=Bb)
                    r[bname] = dict(diff=d1[0], ci=[d1[1], d1[2]], p_two=d1[3], n_valid=d1[4])
                if mainwin:
                    # P4 技能曲线重设计: IGN 差值 + ISS（免比值病态, 审稿A-M2）
                    def ign_curve(S):
                        P = 1.0 - np.exp(-np.maximum(S, 0.0))
                        P = np.clip(P, 1e-12, 1 - 1e-12)
                        ll = -(Y * np.log2(P) + (1 - Y) * np.log2(1 - P))
                        return ll.mean(axis=1)
                    ig_m, ig_c = ign_curve(S_m), ign_curve(S_pop)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        iss = 1.0 - ig_m / ig_c
                    pos_tau = Y.sum(axis=1) > 0
                    dq = np.percentile(ig_m - ig_c, [5, 25, 50, 75, 95])
                    r['skill_ign'] = dict(
                        ISS_all_taus_mean=float(np.nanmean(iss)),
                        ISS_positive_taus_mean=float(np.nanmean(iss[pos_tau])) if pos_tau.any() else None,
                        frac_positive_taus=float(pos_tau.mean()),
                        IGN_model_mean=float(ig_m.mean()), IGN_clim_mean=float(ig_c.mean()),
                        IGN_diff_quantiles_5_25_50_75_95=[float(x) for x in dq],
                        note='IGN=逐格 log2 proper score 泊松转换; ISS=1−IGN_model/IGN_clim; '
                             '差值分位数替代 exp13 病态比值列 SS_vs_pop(−7517)')
                    # P5 分层: seen/unseen cell + 前/后半 τ（H3/H7）
                    yf2 = Y.ravel()
                    strat = {}
                    for nm, cm in [('seen', seen_mask_U), ('unseen', ~seen_mask_U)]:
                        strat[f'cell_{nm}'] = dict(n_pos=int(Y[:, cm].ravel().sum()),
                                                   PR_AUC_model=ap_of(Y[:, cm].ravel(), S_m[:, cm]),
                                                   PR_AUC_persistence=ap_of(Y[:, cm].ravel(), S_p[:, cm]))
                    half = n_tau // 2
                    strat['tau_first_half'] = dict(n_pos=int(Y[:half].sum()),
                                                   PR_AUC_model=ap_of(Y[:half].ravel(), S_m[:half]),
                                                   PR_AUC_persistence=ap_of(Y[:half].ravel(), S_p[:half]))
                    strat['tau_second_half'] = dict(n_pos=int(Y[half:].sum()),
                                                    PR_AUC_model=ap_of(Y[half:].ravel(), S_m[half:]),
                                                    PR_AUC_persistence=ap_of(Y[half:].ravel(), S_p[half:]))
                    r['stratified'] = strat
                res[f'{wn}_{int(H//3600)}h'] = r
                out['universes'][uname] = res
                dump()
                extra = (f" BLa={r['PR_AUC_persist_x_hod']:.4f} BLb={r['PR_AUC_persist_x_qflag']:.4f} "
                         f"BLm={r['PR_AUC_markov_dst1']:.4f}") if mainwin else ""
                bp = r.get('model_minus_persistence', {})
                print(f"  [{uname} {wn} {int(H//3600)}h] model={r['PR_AUC_model']:.4f} "
                      f"persist={r['PR_AUC_persistence']:.4f}{extra} | "
                      f"diff={bp.get('diff', float('nan')):+.4f} CI=[{bp.get('ci', [float('nan')]*2)[0]:+.4f},"
                      f"{bp.get('ci', [float('nan')]*2)[1]:+.4f}] p={bp.get('p_two', float('nan')):.3f}", flush=True)

    # ---------- P6: ×6 Bonferroni + BH（主窗 model−persistence, 2 宇宙×3 时距） ----------
    if not args.smoke:
        fam = []
        for uname in universes:
            for H in Hs:
                key = f'24h_{int(H//3600)}h'   # 主窗 24h; 存储键=窗名_时距
                fam.append((key, out['universes'][uname][key]['model_minus_persistence']['p_two']))
        pvals = np.array([p for _, p in fam])
        order = np.argsort(pvals)
        mfdr = np.empty(len(pvals))
        prev = 0.0
        for rk, i in enumerate(order[::-1]):
            k = len(pvals) - rk
            v = min(1.0, pvals[i] * len(pvals) / k)
            mfdr[i] = max(v, prev)
            prev = mfdr[i]
        out['multiplicity'] = dict(
            family='model−persistence 主窗(24h) 6 检验（2 宇宙×3 时距）',
            tests=[dict(cell=k, p_two_raw=float(p), p_bonferroni=float(min(1.0, p * 6)),
                        p_BH=float(mfdr[i])) for (k, p), i in zip(fam, range(len(fam)))],
            n_survive_bonferroni=int((pvals * 6 <= 0.05).sum()),
            n_survive_BH_05=int((mfdr <= 0.05).sum()),
            note='对齐 exp13 权威口径先例（hop1 版: 仅 targets_train 6h ×6 存活）')
        dump()
        for t in out['multiplicity']['tests']:
            print(f"  {t['cell']}: raw={t['p_two_raw']:.4f} ×6={t['p_bonferroni']:.4f} BH={t['p_BH']:.4f}", flush=True)

    json.dump(out, open(f'{R}/exp15_hybrid_causal.json', 'w'), indent=1, ensure_ascii=False, default=float)
    print(f'落盘 {R}/exp15_hybrid_causal.json; 总耗时 {(time.time()-t00)/60:.1f} min', flush=True)
    open(f'{R}/exp15.done', 'w').write('done: exp15 M-hybrid 因果前瞻+BL-a/b/c, 见 exp15_hybrid_causal.json\n')


if __name__ == '__main__':
    main()
