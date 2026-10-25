#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exp18 窗长尺度化技能评估协议（R2 跨席 Critical C1 修复 + 定向创新方案一/二落地）
动机（审稿A/B 跨席裁定 validated Critical）：exp15 P4 技能曲线的气候学参考未乘窗长 H——
  exp15_hybrid_causal.py:113 R0=每秒事件率, :232 S_pop=R0·qpopU 为秒级强度, 而模型侧
  :72-73 mu0×mu_bg_integral([τ,τ+H)) 与 :83-89 触发项均为 [τ,τ+H) 窗内积分, 两侧同入
  :250-254 ign_curve 的 P=1−exp(−S) 泊松转换——量纲不可比（1h 差 3600×、6h 差 21600×）,
  「健康口径」ISS_positive 0.33–0.56 无效; exp13 SS_vs_pop（−1865/−4490/−7517 等）同根因。
  CSEP cell-based 教义：参考与模型预报必须在同一 [τ,τ+H) 窗内积分后再进概率口径
  （预报=格×窗内率、检验在窗内计数）。
协议（预登记, 先落盘后运行）:
  K1 ×H 修正: S_pop_xH = R0·H·qpopU（秒级重算, 与 B 席独立复算 −0.127/−0.091/−0.008 对表）;
     全部 ISS/IGN 数字以本实验为修正后权威（exp15 skill_ign 与 exp13 SS_vs_pop 作废）。
  K2 Hamill-Juras 参考代表性升级: 池化 pop×R0 参考在 ×H 修复后仍有第二层缺陷（池化气候学
     → 分层样本上虚假技能）→ 增设逐 τ hod-profile 窗积分气候学参考 S_hod = m_h[hod(τ)]·R0·H·qpopU
     （=BL-a 的 IGN 版去掉 persistence 项; m_h=hour_profile train-only 含 0.5 伪计数平滑,
     qpopU 含 0.5 平滑——空格 IGN_clim→0 防护已内建）; model−ref 逐 τ IGN 差做块 bootstrap
     （block=6 τ, B=1000, 同 exp13/15 块结构）+seen/unseen 格分层。
  K3 hop1-vs-hybrid 分层对照（零新拟合, 收口审稿A-M3）: exp13 冻结模型（hop1+exp4_th, 无新拟合）
     在同一因果协议下重评分, 补 seen/unseen 分层 AP——检验 LOG:231「q_flag 背景在新目标退化拖累
     3–6h」归因; 归因成立与否都双向成文。
  K4 Molchan 图（正例 τ 面, 定向创新方案二）: miss rate ν vs 报警占比 τ, 模型/persistence 双轨迹,
     面积技能分 ASS=1−2·Area, τ 块 bootstrap CI（B=500）——替代「BLc 显著性无族校正不可引」
     的报警面语义。
  K5 决策曲线 DCA（方案二）: NB(p_t)=TP/N−(p_t/(1−p_t))·FP/N, model/persistence/treat-all/treat-none
     四曲线×三时距; 只报点值曲线不做 NB 的 CI（PMC10243069 警告 net benefit 推断实践有害）。
  K6 探索性全族多重比较（审稿A-m1/m4）: 预登记族=model−persistence 6 检验; 本块把主窗 model−BLc 6
     +敏感性窗 model−persistence 12 并入=24 检验「探索性全族」（事后定族, 只作引用警示）,
     对 exp15 JSON 已存 p_two 重算 ×24 Bonferroni+BH——纯算术, 零重跑。
正确性前置检查: ①hybrid S 矩阵复算 AP 与 exp15 权威 JSON 逐位同（<1e-9）; ②hop1 复算 AP 与
  exp13 权威 JSON 逐位同（<1e-9）——不过检查即终止（防协议漂移）。
模型: 主窗 W24 hybrid（exp12_th_M_hybrid_W24h_c）+ exp13 冻结 hop1（exp4_th）——均无新拟合。
输出: results/exp18_skill_scale_protocol.json
"""
import sys, json, time, argparse
import numpy as np
sys.path.insert(0, './scripts')
sys.path.insert(0, './src')
from exp_patch_common import load_all, eb_flag_q, R
from lanl_pp import GraphHawkes, hour_profile
from exp15_hybrid_causal import grid_scores_causal_v2
from sklearn.metrics import average_precision_score

RNG = np.random.default_rng(31)   # 独立于 exp15 的 29, 防流耦合


def ap_guard(y, s):
    """unseen 层可能 0 正例（targets_train unseen n_pos=0）——exp15 ap_of 同款守卫"""
    return float(average_precision_score(y, s)) if np.asarray(y).sum() > 0 else float('nan')


# ---------------- K1/K2: ×H 修正技能面 ----------------
def ign_curve_log2(S, Y):
    """与 exp15 :250-254 同款: P=1−exp(−S) → 逐格 log2 proper score, 返回逐 τ 均值与逐格矩阵"""
    P = 1.0 - np.exp(-np.maximum(S, 0.0))
    P = np.clip(P, 1e-12, 1 - 1e-12)
    ll = -(Y * np.log2(P) + (1 - Y) * np.log2(1 - P))
    return ll.mean(axis=1), ll


def block_boot_mean_diff(dll, n_tau, nU, block=6, B=1000, rng=None):
    """逐格 IGN 差 (model−ref) 的 τ 块 bootstrap 均值 CI/p（块结构同 exp13 block_boot_ap_diff）"""
    rng = rng or RNG
    dll = np.asarray(dll, float).ravel()   # (n_tau, nU) 展平后做平面索引
    obs = float(dll.mean())
    nb = int(np.ceil(n_tau / block))
    cell_off = np.arange(nU)
    diffs = np.empty(B)
    for b in range(B):
        starts = rng.integers(0, n_tau - block + 1, size=nb)
        tau_idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n_tau]
        idx = (tau_idx[:, None] * nU + cell_off[None, :]).ravel()
        diffs[b] = dll[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_two = float(min(1.0, 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())))
    return obs, float(lo), float(hi), p_two


def skill_block(S_m, Y, refs, n_tau, nU, seen_mask, B, rng, tag=''):
    """对一组参考（窗积分口径）计算修正后 IGN/ISS + bootstrap + seen/unseen 分层"""
    ig_m, ll_m = ign_curve_log2(S_m, Y)
    yf = Y.ravel()
    pos_tau = Y.sum(axis=1) > 0
    out = {}
    for rname, S_r in refs.items():
        ig_r, ll_r = ign_curve_log2(S_r, Y)
        with np.errstate(divide='ignore', invalid='ignore'):
            iss = 1.0 - ig_m / ig_r
        dll = ll_m - ll_r
        d = dict(IGN_model_mean=float(ig_m.mean()), IGN_ref_mean=float(ig_r.mean()),
                 ISS_all_taus_mean=float(np.nanmean(iss)),
                 ISS_positive_taus_mean=float(np.nanmean(iss[pos_tau])) if pos_tau.any() else None,
                 frac_positive_taus=float(pos_tau.mean()),
                 IGN_diff_quantiles_5_25_50_75_95=[float(x) for x in np.percentile(ig_m - ig_r, [5, 25, 50, 75, 95])])
        obs, lo, hi, p = block_boot_mean_diff(dll, n_tau, nU, block=6, B=B, rng=rng)
        d['IGN_diff_blockboot'] = dict(mean=obs, ci=[lo, hi], p_two=p, B=B, block=6,
                                       note='逐格 IGN 差(model−ref) τ 块 bootstrap; 负=模型更优')
        strat = {}
        for nm, cm in [('seen', seen_mask), ('unseen', ~seen_mask)]:
            dll_s = dll[:, cm]
            # 层内 ISS 与总体同构: 1 − 层内均值(IGN_model)/层内均值(IGN_ref), 逐 τ
            ig_m_s = ll_m[:, cm].mean(axis=1)
            ig_r_s = ll_r[:, cm].mean(axis=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                iss_s = 1.0 - ig_m_s / ig_r_s
            st = dict(IGN_diff_mean=float(dll_s.mean()), n_pos=int(Y[:, cm].sum()))
            st['ISS_positive_taus_mean'] = (float(np.nanmean(iss_s[pos_tau])) if pos_tau.any() else None)
            o, l, h, pp = block_boot_mean_diff(dll_s, n_tau, int(cm.sum()), block=6, B=max(200, B // 2), rng=rng)
            st['IGN_diff_blockboot'] = dict(mean=o, ci=[l, h], p_two=pp)
            strat[f'cell_{nm}'] = st
        d['stratified'] = strat
        out[f'vs_{rname}'] = d
    return out


# ---------------- K4: Molchan 图 ----------------
def molchan_curve(y, s):
    """descending 排序; 返回 (alarm_frac, miss_rate) 含 (0,1) 起点; ASS=1−2·Area"""
    n_pos = int(y.sum())
    order = np.argsort(-s, kind='stable')
    yy = y[order]
    k = np.arange(1, len(y) + 1)
    af = k / len(y)
    miss = 1.0 - np.cumsum(yy) / n_pos
    aff = np.concatenate([[0.0], af])
    mss = np.concatenate([[1.0], miss])
    area = float(np.trapz(mss, aff))
    return aff, mss, area


def molchan_block(y, s, n_tau, nU, block=6, B=500, rng=None):
    rng = rng or RNG
    _, _, area = molchan_curve(y, s)
    nb = int(np.ceil(n_tau / block))
    cell_off = np.arange(nU)
    areas = np.empty(B)
    for b in range(B):
        starts = rng.integers(0, n_tau - block + 1, size=nb)
        tau_idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n_tau]
        idx = (tau_idx[:, None] * nU + cell_off[None, :]).ravel()
        areas[b] = molchan_curve(y[idx], s[idx])[2]
    areas = areas[np.isfinite(areas)]
    lo, hi = np.percentile(areas, [2.5, 97.5])
    ass = 1.0 - 2.0 * area
    ci = [1.0 - 2.0 * hi, 1.0 - 2.0 * lo]   # 面积越小技能越高, 反向映射
    return dict(alarm_frac_full_area=area, ASS=ass, ASS_ci=[float(ci[0]), float(ci[1])],
                B=B, n_valid=int(len(areas)),
                note='Molchan 面积技能分 ASS=1−Area/0.5; τ 块 bootstrap(块=6, 保留格维); '
                     '随机参考对角线 ASS=0')


def molchan_downsample(aff, mss, n=120):
    idx = np.unique(np.linspace(0, len(aff) - 1, n).astype(int))
    return dict(alarm_frac=[float(x) for x in aff[idx]], miss_rate=[float(x) for x in mss[idx]])


# ---------------- K5: 决策曲线 DCA ----------------
def dca_curves(S_m, S_p, Y, n_thr=45):
    P_m = 1.0 - np.exp(-np.maximum(S_m, 0.0))
    P_p = 1.0 - np.exp(-np.maximum(S_p, 0.0))
    yf = Y.ravel().astype(float)
    N = len(yf)
    prev = float(yf.mean())
    thr = np.geomspace(1e-5, 0.5, n_thr)
    out = dict(threshold=[float(x) for x in thr], prevalence=prev,
               odds=[float(x) for x in thr / (1 - thr)])
    for nm, P in [('model', P_m), ('persistence', P_p)]:
        p = P.ravel()
        nb = []
        for pt in thr:
            al = p >= pt
            tp = float((yf[al] == 1).sum()) / N
            fp = float((yf[al] == 0).sum()) / N
            nb.append(tp - (pt / (1 - pt)) * fp)
        out[f'NB_{nm}'] = [float(x) for x in nb]
    out['NB_treat_all'] = [float(prev - (pt / (1 - pt)) * (1 - prev)) for pt in thr]
    out['NB_treat_none'] = [0.0] * n_thr
    out['note'] = ('NB=TP/N−(p_t/(1−p_t))·FP/N, 阈值 p_t 的 odds=每容忍 1 误报所需真报赔率; '
                   '只报点值曲线不做 NB 的 CI（Vickers & Elkin 2006; PMC10243069 警告 net benefit '
                   '区间推断实践有害）; 漏报成本≫误报→低 p_t 端为主')
    return out


# ---------------- K6: 探索性全族多重比较（纯算术, 读 exp15 JSON） ----------------
def expanded_multiplicity():
    j15 = json.load(open(f'{R}/exp15_hybrid_causal.json'))
    tests = []
    for u in ['targets', 'targets_train']:
        for k, r in j15['universes'][u].items():
            wn, hh = k.split('_')
            tests.append((f'{u}:{k}:model_minus_persistence', r['model_minus_persistence']['p_two'],
                          'persistence'))
            if 'model_minus_BLc' in r:
                tests.append((f'{u}:{k}:model_minus_BLc', r['model_minus_BLc']['p_two'], 'BLc'))
    pvals = np.array([p for _, p, _ in tests])
    m = len(pvals)
    bonf = np.minimum(1.0, pvals * m)
    # BH
    order = np.argsort(pvals)
    bh = np.empty(m)
    prev = 0.0
    for rk, i in enumerate(order[::-1]):
        kk = m - rk
        v = min(1.0, pvals[i] * m / kk)
        bh[i] = max(v, prev)
        prev = bh[i]
    out = dict(family=('探索性全族 24 检验（事后定族, 非预登记）= 主窗 model−persistence 6（预登记族, '
                       '见 exp15 multiplicity）+ 主窗 model−BLc 6 + 敏感性窗 W12/W48 model−persistence 12'),
               tests=[dict(cell=tests[i][0], p_two_raw=float(pvals[i]), p_bonferroni_x24=float(bonf[i]),
                           p_BH=float(bh[i]), series=tests[i][2]) for i in range(m)],
               n_survive_bonferroni_x24=int((bonf <= 0.05).sum()),
               note=('用途=引用警示: 主判定仍以预登记 6 检验族为准; BLc 的 1h 两格在预登记族外, '
                     '并入任何合理扩族（12/24 检验）后 targets 1h p≈0.048 边缘、targets_train 1h '
                     '不存活——BLc 显著性不得作为独立结论引用'))
    return out


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

    universes = {'targets': list(V_all), 'targets_train': sorted(set(dst_all[:cut]))}
    Hs = [6 * 3600.0] if args.smoke else [3600.0, 3 * 3600.0, 6 * 3600.0]
    unis = {'targets': universes['targets']} if args.smoke else universes
    B_main = 100 if args.smoke else 1000
    B_mol = 100 if args.smoke else 500

    # 模型（均无新拟合）: 主窗 hybrid + exp13 冻结 hop1
    m_hyb = GraphHawkes(t_all[:cut], list(dst_all[:cut]), list(src_all[:cut]), adj, nodes,
                        dst_extra=V_all, kernel='omori', gk='hop0', bg='hod24', W=86400.0,
                        qbg='custom', q_custom=q_flag)
    m_hyb.th = np.load(f'{R}/exp12_th_M_hybrid_W24h_c.npy')
    m_hop1 = GraphHawkes(t_all[:cut], list(dst_all[:cut]), list(ev['src'][:cut]), adj, nodes,
                         dst_extra=V_all, kernel='omori', gk='hop1', bg='hod24', W=86400.0, qbg='pop')
    m_hop1.th = np.load(f'{R}/exp4_th.npy')

    j15 = json.load(open(f'{R}/exp15_hybrid_causal.json'))
    j13 = json.load(open(f'{R}/exp13_causal_forecast.json'))

    out = dict(exp='Exp18_window_scaled_skill_protocol',
               motive=('跨席 Critical C1: exp15 S_pop/exp13 SS_vs_pop 参考未乘窗长 H（秒级率 vs 窗内积分, '
                       '1h 差 3600×、6h 差 21600×）; 修复=一切进概率口径的参考先做 [τ,τ+H) 窗内积分 '
                       '(CSEP cell-based 同窗积分语义) + Hamill-Juras 逐 τ 代表性参考升级'),
               preregistered=dict(
                   K1='×H 修正后 ISS_positive 预期反号（B 席独立复算 −0.127/−0.091/−0.008）; '
                      '本实验为修正后 ISS/IGN 权威, exp15 skill_ign 与 exp13 SS_vs_pop 作废',
                   K2='逐 τ hod 窗积分参考下 model−ref IGN 差块 bootstrap 符号 = 调和表校准面列',
                   K3='hop1-vs-hybrid seen/unseen 分层对照 = LOG:231「q_flag 背景退化拖累」归因检验, 双向成文',
                   K4='Molchan ASS(model) vs ASS(persistence) 逐时距 + τ 块 bootstrap CI',
                   K5='DCA 点值曲线×3 时距（无 CI, 阈值带叙事）',
                   K6='探索性全族 24 检验 ×24 Bonferroni+BH（引用警示, 非主判定）'),
               correctness_checks={},
               universes={}, molchan={}, dca={}, expanded_multiplicity=expanded_multiplicity())

    def dump():
        json.dump(out, open(f'{R}/exp18_skill_scale_protocol.json', 'w'), indent=1,
                  ensure_ascii=False, default=float)

    for uname, U in unis.items():
        uidx = {u: i for i, u in enumerate(U)}
        nU = len(U)
        qpopU = np.array([popC[cidxC[u]] for u in U]); qpopU = qpopU / qpopU.sum()
        seen_mask_U = np.array([u in set(universes['targets_train']) for u in U])
        res = {}
        for H in Hs:
            # ---- hybrid（主窗 W24） ----
            taus, S_m, S_p, Y = grid_scores_causal_v2(m_hyb, m_hyb.th, U, t_all, list(dst_all),
                                                      t_cut, T_end, H, causal=True, q_popvec=popC)
            n_tau = len(taus)
            yf = Y.ravel()
            ap_m = float(average_precision_score(yf, S_m.ravel())) if yf.sum() else float('nan')
            ap_p = float(average_precision_score(yf, S_p.ravel())) if yf.sum() else float('nan')
            key = f'{int(H // 3600)}h'
            j15r = j15['universes'][uname][f'24h_{key}']
            chk_h = dict(ap_model_mine=ap_m, ap_model_exp15=j15r['PR_AUC_model'],
                         ap_persist_mine=ap_p, ap_persist_exp15=j15r['PR_AUC_persistence'])
            chk_h['pass'] = bool(abs(ap_m - j15r['PR_AUC_model']) < 1e-9 and
                                 abs(ap_p - j15r['PR_AUC_persistence']) < 1e-9)
            # ---- hop1（exp13 冻结模型, 零新拟合） ----
            _, S_m1, S_p1, Y1 = grid_scores_causal_v2(m_hop1, m_hop1.th, U, t_all, list(dst_all),
                                                      t_cut, T_end, H, causal=True, q_popvec=popC)
            assert np.array_equal(Y, Y1), f'{uname} {key}: Y 矩阵不一致'
            ap_m1 = float(average_precision_score(yf, S_m1.ravel()))
            j13r = j13['universes'][uname][key]
            chk_h1 = dict(ap_model_mine=ap_m1, ap_model_exp13=j13r['PR_AUC_model_causal'])
            chk_h1['pass'] = bool(abs(ap_m1 - j13r['PR_AUC_model_causal']) < 1e-9)
            out['correctness_checks'][f'{uname}_{key}'] = dict(hybrid_vs_exp15=chk_h, hop1_vs_exp13=chk_h1)
            dump()
            if not (chk_h['pass'] and chk_h1['pass']):
                raise RuntimeError(f'{uname} {key} 复算 AP 与权威 JSON 不一致, 终止（防协议漂移）')

            # ---- 参考组（全部窗积分口径） ----
            S_pop_xH = np.tile(R0 * H * qpopU, (n_tau, 1))          # K1: 池化气候学 ×H 修正
            S_hod = np.zeros((n_tau, nU))                            # K2: 逐 τ hod 窗积分气候学
            S_bla = np.zeros((n_tau, nU))                            # BL-a（计数口径, 复算供 IGN 对照）
            tt_all = np.asarray(t_all, float)
            ds_all = np.asarray(dst_all)
            for gi, tau in enumerate(taus):
                hod = int((tau % 86400) // 3600)
                S_hod[gi] = m_h[hod] * R0 * H * qpopU
                hist = np.where(tt_all < tau)[0]
                counts = np.zeros(nU)
                for d in ds_all[hist[(tt_all[hist] >= tau - H) & (tt_all[hist] < tau)]]:
                    if d in uidx:
                        counts[uidx[d]] += 1.0
                S_bla[gi] = m_h[hod] * (counts + qpopU)
            refs = dict(pop_xH=S_pop_xH, hod_xH=S_hod, BLa=S_bla)
            res[key] = dict(
                n_tau=int(n_tau), n_pos=int(yf.sum()),
                base_rate=float(yf.mean()),
                AP=dict(model=ap_m, persistence=ap_p, hop1=ap_m1),
                skill_scaled=skill_block(S_m, Y, refs, n_tau, nU, seen_mask_U, B_main, RNG),
                hop1_skill_scaled=skill_block(S_m1, Y, dict(pop_xH=S_pop_xH, hod_xH=S_hod),
                                              n_tau, nU, seen_mask_U, B_main, RNG),
                # K3: hop1-vs-hybrid 分层对照（零新拟合; 检验 LOG:231 归因）
                hop1_vs_hybrid_stratified=dict(
                    note='同一因果协议下 hop1(exp13 冻结) vs hybrid(exp15) 的 seen/unseen 格 AP 对照; '
                         'targets_train unseen 层 0 正例→AP=NaN（exp15 同口径）',
                    seen=dict(hybrid=j15['universes'][uname][f'24h_{key}']['stratified']['cell_seen']['PR_AUC_model'],
                              hop1=ap_guard(Y[:, seen_mask_U].ravel(), S_m1[:, seen_mask_U].ravel())),
                    unseen=dict(hybrid=j15['universes'][uname][f'24h_{key}']['stratified']['cell_unseen']['PR_AUC_model'],
                                hop1=ap_guard(Y[:, ~seen_mask_U].ravel(), S_m1[:, ~seen_mask_U].ravel()))),
                molchan=dict(
                    model=molchan_block(yf, S_m.ravel(), n_tau, nU, B=B_mol),
                    persistence=molchan_block(yf, S_p.ravel(), n_tau, nU, B=B_mol),
                    curves=dict(model=molchan_downsample(*molchan_curve(yf, S_m.ravel())[:2]),
                                persistence=molchan_downsample(*molchan_curve(yf, S_p.ravel())[:2])),
                    n_pos=int(yf.sum()), n_points=int(len(yf))),
                dca=dca_curves(S_m, S_p, Y))
            out['universes'][uname] = res
            dump()
            sb = res[key]['skill_scaled']['vs_hod_xH']
            print(f"  [{uname} {key}] AP chk pass | ISS_pop_xH pos={res[key]['skill_scaled']['vs_pop_xH']['ISS_positive_taus_mean']:+.4f} "
                  f"all={res[key]['skill_scaled']['vs_pop_xH']['ISS_all_taus_mean']:+.4f} | "
                  f"ISS_hod pos={sb['ISS_positive_taus_mean']:+.4f} | "
                  f"IGNdiff(model−hod) CI=[{sb['IGN_diff_blockboot']['ci'][0]:+.4f},{sb['IGN_diff_blockboot']['ci'][1]:+.4f}] "
                  f"p={sb['IGN_diff_blockboot']['p_two']:.3f} | "
                  f"ASS model={res[key]['molchan']['model']['ASS']:.4f} persist={res[key]['molchan']['persistence']['ASS']:.4f}", flush=True)

    # K1 对表: B 席独立复算预期（targets 宇宙 1h/3h/6h ISS_positive = −0.127/−0.091/−0.008）
    if not args.smoke:
        pred = {'1h': -0.127, '3h': -0.091, '6h': -0.008}
        cmp_b = {}
        for k, pv in pred.items():
            mine = out['universes']['targets'][k]['skill_scaled']['vs_pop_xH']['ISS_positive_taus_mean']
            cmp_b[k] = dict(mine=mine, reviewerB=pv, diff=mine - pv)
        out['cross_check_reviewerB'] = dict(
            target='B 席 /tmp/review_r2b/check_ign.py ×H 独立复算（ISS_positive_taus_mean, targets 宇宙）',
            comparison=cmp_b,
            note='实现细节差异（如 m_h 平滑已内建/参考构造）允许小差; 大偏差须排查')
        dump()

    json.dump(out, open(f'{R}/exp18_skill_scale_protocol.json', 'w'), indent=1,
              ensure_ascii=False, default=float)
    print(f'落盘 {R}/exp18_skill_scale_protocol.json; 总耗时 {(time.time()-t00)/60:.1f} min', flush=True)
    open(f'{R}/exp18.done', 'w').write('done: exp18 窗长尺度化技能协议（×H 修正+hod 参考+Molchan/DCA+hop1 对照）\n')


if __name__ == '__main__':
    main()
