#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exp20 hod 参考真窗积分收口（R3 §12.5 清单4 = A-m1+B-m3 合并）
问题: exp18 的 hod 参考实现 S_hod[gi]=m_h[hod(τ)]·R0·H·qpopU 用 τ 起点钟点小时率×整窗
  （scripts/exp18_skill_scale_protocol.py:316），而 τ 网格不对齐钟点（1h 窗也跨钟点）——
  违反「参考率必须配同层暴露时间」教义（间接标准化 E=Σ_j λ_ref,j×PT_j）。
  真窗积分=分段常数率下 Σ_h m_h[h]×(窗内该小时覆盖秒数)（src/lanl_pp.py:53 hour_coverage_seconds 在库）。
协议（预登记, 先落盘后运行）:
  T1 S_hod_true[gi] = R0·H·qpopU·m_h_eff(τ,H),  m_h_eff = Σ_h m_h[h]·cov_h(τ,τ+H)/H
     （=与旧版同构, 唯一差 m_h[hod]→覆盖率加权 m_h_eff; CSEP cell 语义: 参考为连续率函数×窗）
  T2 Q5 裁定（执行人定夺落盘）: BL-a 的 hod 项同步改真积分 S_bla_true = m_h_eff·(counts+qpopU)
     ——理由: BL-a 与 hod_xH 同为「小时率×暴露」型参考, 只改其一则两参考实现不同构、
     BL-a 对照仍带近似误差; 口径统一一次收口防口径分裂; 旧版 BL-a 数字保留对照, 定义变化显式标注。
  T3 smoke 先行（单宇宙单时距）: 前置正确性检查=①hybrid/hop1 复算 AP 与 exp15/exp13 权威 <1e-9;
     ②同 RNG(seed=31) 同维度流下 vs_pop_xH 块与 exp18 落盘逐位一致（参考矩阵外输出零漂移证据）;
     ③vs_hod_xH 的 IGN_model_mean/IGN_diff 分位数与 exp18 一致（模型侧未动）, 仅 IGN_ref_mean 变。
  T4 全量: 2 宇宙×3 时距; 落盘近似 vs 真积分逐格相对差（S 矩阵层面+IGN diff CI 变化）
     +Q3 裁定（vs_hod ISS_all 病态列是否恢复可读, 二选一落盘）。
  T5 旧近似数字收口: exp18 JSON 脚本化加 R3_correction_20261021 字段（原键值零改动断言）,
     权威指针改指本实验真积分版。
不引用审稿人 B 自建复算数字进权威工件（相对差/真积分 CI 以本实验落盘值为准——exp19 溯源纪律同源）。
输出: results/exp20_hod_true_integral.json
"""
import sys, json, time, argparse, warnings
import numpy as np
sys.path.insert(0, './scripts')
sys.path.insert(0, './src')
from exp_patch_common import load_all, eb_flag_q, R
from lanl_pp import GraphHawkes, hour_profile, hour_coverage_seconds
from exp15_hybrid_causal import grid_scores_causal_v2
from sklearn.metrics import average_precision_score

RNG = np.random.default_rng(31)   # 与 exp18 同 seed: 同维度 bootstrap 消耗同量随机数 → pop_xH 列逐位可比
warnings.filterwarnings('ignore', category=RuntimeWarning)


def ap_guard(y, s):
    return float(average_precision_score(y, s)) if np.asarray(y).sum() > 0 else float('nan')


def ign_curve_log2(S, Y):
    P = 1.0 - np.exp(-np.maximum(S, 0.0))
    P = np.clip(P, 1e-12, 1 - 1e-12)
    ll = -(Y * np.log2(P) + (1 - Y) * np.log2(1 - P))
    return ll.mean(axis=1), ll


def block_boot_mean_diff(dll, n_tau, nU, block=6, B=1000, rng=None):
    """与 exp18 同款: 逐格 IGN 差 (model−ref) 的 τ 块 bootstrap; 空层(nU=0)守卫→NaN+null p_two.
    注意: 空层分支仍按 exp18 语义消耗 B×nb 个随机数（保持同 seed 下 RNG 流位置逐位对齐——
    exp18 空层也走完整循环, 差别仅是 idx 为空数组）"""
    rng = rng or RNG
    dll = np.asarray(dll, float).ravel()
    nb = int(np.ceil(n_tau / block))
    if dll.size == 0:
        for _ in range(B):
            rng.integers(0, n_tau - block + 1, size=nb)
        return float('nan'), float('nan'), float('nan'), None
    obs = float(dll.mean())
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
    """与 exp18 同构; 两处升级: ①空层(层内 0 格)块 p_two→None+structural_empty 标志（清单7 规范, 不重蹈 p_two=0.0 伪值）;
    ②refs 顺序保持 pop_xH 最前（RNG 流与 exp18 对表的前提）"""
    ig_m, ll_m = ign_curve_log2(S_m, Y)
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
            empty = (int(cm.sum()) == 0)
            ig_m_s = ll_m[:, cm].mean(axis=1) if not empty else np.full(n_tau, np.nan)
            ig_r_s = ll_r[:, cm].mean(axis=1) if not empty else np.full(n_tau, np.nan)
            with np.errstate(divide='ignore', invalid='ignore'):
                iss_s = 1.0 - ig_m_s / ig_r_s
            # 空层也调用（守卫保持 RNG 消耗流与 exp18 逐位对齐, 返回 NaN+null p_two）
            o, l, h, pp = block_boot_mean_diff(dll_s, n_tau, int(cm.sum()), block=6,
                                               B=max(200, B // 2), rng=rng)
            st = dict(IGN_diff_mean=(float('nan') if empty else float(dll_s.mean())),
                      n_pos=int(Y[:, cm].sum()))
            if empty:
                st['structural_empty'] = True
                st['note'] = '层内 0 格（targets_train unseen 层为空宇宙）→ 结构空层, IGN/bootstrap 无定义记 NaN, p_two 记 null'
            st['ISS_positive_taus_mean'] = (float(np.nanmean(iss_s[pos_tau])) if (pos_tau.any() and not empty) else None)
            st['IGN_diff_blockboot'] = dict(mean=o, ci=[l, h], p_two=pp,
                                            **({'structural_empty': True} if empty else {}))
            strat[f'cell_{nm}'] = st
        d['stratified'] = strat
        out[f'vs_{rname}'] = d
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

    m_hyb = GraphHawkes(t_all[:cut], list(dst_all[:cut]), list(src_all[:cut]), adj, nodes,
                        dst_extra=V_all, kernel='omori', gk='hop0', bg='hod24', W=86400.0,
                        qbg='custom', q_custom=q_flag)
    m_hyb.th = np.load(f'{R}/exp12_th_M_hybrid_W24h_c.npy')
    m_hop1 = GraphHawkes(t_all[:cut], list(dst_all[:cut]), list(ev['src'][:cut]), adj, nodes,
                         dst_extra=V_all, kernel='omori', gk='hop1', bg='hod24', W=86400.0, qbg='pop')
    m_hop1.th = np.load(f'{R}/exp4_th.npy')

    j18 = json.load(open(f'{R}/exp18_skill_scale_protocol.json'))
    j15 = json.load(open(f'{R}/exp15_hybrid_causal.json'))
    j13 = json.load(open(f'{R}/exp13_causal_forecast.json'))

    out = dict(exp='Exp20_hod_true_integral',
               motive=('§12.5 清单4: exp18 hod 参考系 τ 起点钟点率×整窗近似（:316）, τ 网格不对齐钟点; '
                       '真窗积分=hour_coverage_seconds 覆盖加权（间接标准化 E=Σλ_ref×暴露时间教义）'),
               preregistered=dict(
                   T1='S_hod_true=R0·H·qpopU·m_h_eff(τ,H), m_h_eff=Σ_h m_h[h]·cov_h/H（分段常数率窗积分）',
                   T2_Q5_ruling='BL-a hod 项同步改真积分（两 hod 型参考实现同构、一次收口; 旧版数字保留对照, '
                                '定义变化显式标注; 不改则须 §12.1 注明两参考不同构）',
                   T3_smoke='AP 对表 <1e-9 + 同 RNG 流 vs_pop_xH 块与 exp18 逐位一致 + 模型侧统计量零漂移',
                   T4='全量落盘近似 vs 真积分逐格相对差+diff CI 变化+Q3 ISS 病态列裁定',
                   T5='exp18 JSON 脚本化加 R3_correction（原键零改动断言）, 权威指针改指本实验',
                   reviewer_note='审稿人 B 自建复算数字（相对差 21.8%/48.3%/60.6%、6h CI）不作权威引用, 只作方向参照'),
               correctness_checks={}, universes={})
    # 预登记先行落盘
    json.dump(out, open(f'{R}/exp20_hod_true_integral.json', 'w'), indent=1, ensure_ascii=False, default=float)
    print('预登记已落盘 exp20_hod_true_integral.json（先落盘后运行）', flush=True)

    def dump():
        json.dump(out, open(f'{R}/exp20_hod_true_integral.json', 'w'), indent=1,
                  ensure_ascii=False, default=float)

    tt_all = np.asarray(t_all, float)
    ds_all = np.asarray(dst_all)
    for uname, U in unis.items():
        uidx = {u: i for i, u in enumerate(U)}
        nU = len(U)
        qpopU = np.array([popC[cidxC[u]] for u in U]); qpopU = qpopU / qpopU.sum()
        seen_mask_U = np.array([u in set(universes['targets_train']) for u in U])
        res = {}
        for H in Hs:
            key = f'{int(H // 3600)}h'
            taus, S_m, S_p, Y = grid_scores_causal_v2(m_hyb, m_hyb.th, U, t_all, list(dst_all),
                                                      t_cut, T_end, H, causal=True, q_popvec=popC)
            n_tau = len(taus)
            yf = Y.ravel()
            ap_m = float(average_precision_score(yf, S_m.ravel())) if yf.sum() else float('nan')
            ap_p = float(average_precision_score(yf, S_p.ravel())) if yf.sum() else float('nan')
            j15r = j15['universes'][uname][f'24h_{key}']
            chk_h = dict(pass_=bool(abs(ap_m - j15r['PR_AUC_model']) < 1e-9 and
                                    abs(ap_p - j15r['PR_AUC_persistence']) < 1e-9))
            _, S_m1, S_p1, Y1 = grid_scores_causal_v2(m_hop1, m_hop1.th, U, t_all, list(dst_all),
                                                      t_cut, T_end, H, causal=True, q_popvec=popC)
            assert np.array_equal(Y, Y1)
            ap_m1 = float(average_precision_score(yf, S_m1.ravel()))
            j13r = j13['universes'][uname][key]
            chk_h1 = dict(pass_=bool(abs(ap_m1 - j13r['PR_AUC_model_causal']) < 1e-9))
            out['correctness_checks'][f'{uname}_{key}'] = dict(
                hybrid_ap_vs_exp15=dict(mine=ap_m, exp15=j15r['PR_AUC_model'], **{'pass': chk_h['pass_']}),
                persistence_ap_vs_exp15=dict(mine=ap_p, exp15=j15r['PR_AUC_persistence']),
                hop1_ap_vs_exp13=dict(mine=ap_m1, exp13=j13r['PR_AUC_model_causal'], **{'pass': chk_h1['pass_']}))
            dump()
            if not (chk_h['pass_'] and chk_h1['pass_']):
                raise RuntimeError(f'{uname} {key} AP 对表失败, 终止（防协议漂移）')

            # ---- 参考组: pop_xH（与 exp18 逐位同）+ hod_true + BLa_true（Q5 同步真积分） ----
            S_pop_xH = np.tile(R0 * H * qpopU, (n_tau, 1))
            S_hod_true = np.zeros((n_tau, nU))
            S_hod_approx = np.zeros((n_tau, nU))
            S_bla_true = np.zeros((n_tau, nU))
            # 旧近似版 BLa（对照用, 不进 bootstrap 流——若进流会破坏 pop_xH 后续 RNG 位置一致性……
            #   实际 exp18 流=pop_xH→hod→BLa 三次 bootstrap; exp20 流=pop_xH→hod_true→BLa_true 同为三次,
            #   维度序列相同 → RNG 位置仍逐位对齐; 旧版对照矩阵直接后算）
            S_bla_approx = np.zeros((n_tau, nU))
            for gi, tau in enumerate(taus):
                cov = hour_coverage_seconds(tau, tau + H)          # (24,) 各小时覆盖秒数
                m_h_eff = float((m_h * cov).sum() / H)
                hod = int((tau % 86400) // 3600)
                S_hod_true[gi] = m_h_eff * R0 * H * qpopU
                S_hod_approx[gi] = m_h[hod] * R0 * H * qpopU
                hist = np.where(tt_all < tau)[0]
                counts = np.zeros(nU)
                for d in ds_all[hist[(tt_all[hist] >= tau - H) & (tt_all[hist] < tau)]]:
                    if d in uidx:
                        counts[uidx[d]] += 1.0
                S_bla_true[gi] = m_h_eff * (counts + qpopU)
                S_bla_approx[gi] = m_h[hod] * (counts + qpopU)

            refs = dict(pop_xH=S_pop_xH, hod_xH_true=S_hod_true, BLa_true=S_bla_true)
            res[key] = dict(
                n_tau=int(n_tau), n_pos=int(yf.sum()),
                AP=dict(model=ap_m, persistence=ap_p, hop1=ap_m1),
                skill_scaled=skill_block(S_m, Y, refs, n_tau, nU, seen_mask_U, B_main, RNG),
                hop1_skill_scaled=skill_block(S_m1, Y, dict(pop_xH=S_pop_xH, hod_xH_true=S_hod_true),
                                              n_tau, nU, seen_mask_U, B_main, RNG))
            # ---- 近似 vs 真积分对照（T4; bootstrap 流之外, 纯后算） ----
            cmp_blk = {}
            _, ll_m_full = ign_curve_log2(S_m, Y)
            for refn, S_t, S_a in [('hod_xH', S_hod_true, S_hod_approx), ('BLa', S_bla_true, S_bla_approx)]:
                rel = (S_t - S_a) / np.where(np.abs(S_a) > 0, S_a, np.nan)
                ig_r_t, ll_r_t = ign_curve_log2(S_t, Y)
                ig_r_a, ll_r_a = ign_curve_log2(S_a, Y)
                dll_t = ll_m_full - ll_r_t
                dll_a = ll_m_full - ll_r_a
                cmp_blk[refn] = dict(
                    S_rel_diff_mean=float(np.nanmean(rel)), S_rel_diff_median=float(np.nanmedian(rel)),
                    S_rel_diff_absmax=float(np.nanmax(np.abs(rel))),
                    IGN_ref_mean_approx=float(ig_r_a.mean()), IGN_ref_mean_true=float(ig_r_t.mean()),
                    IGNdiff_mean_approx=float(dll_a.mean()), IGNdiff_mean_true=float(dll_t.mean()),
                    diff_CI_shift=[float(dll_t.mean() - dll_a.mean())],
                    note='近似=起点钟点小时率×整窗（exp18:316）; true=hour_coverage_seconds 覆盖加权')
            # 近似版 diff 的 bootstrap CI（独立 RNG, 不动主流——用派生 rng）
            for refn, S_a, S_t, rn in [('hod_xH', S_hod_approx, S_hod_true, 'hod'),
                                       ('BLa', S_bla_approx, S_bla_true, 'BLa')]:
                rr = np.random.default_rng(4100 + (abs(hash(f'{uname}{key}{rn}')) % 1000))
                dll_a = ign_curve_log2(S_m, Y)[1] - ign_curve_log2(S_a, Y)[1]
                dll_t = ign_curve_log2(S_m, Y)[1] - ign_curve_log2(S_t, Y)[1]
                for tag, dll in [('approx', dll_a), ('true', dll_t)]:
                    o, l, h, p = block_boot_mean_diff(dll, n_tau, nU, block=6, B=B_main, rng=rr)
                    cmp_blk[refn][f'diff_boot_{tag}'] = dict(mean=o, ci=[l, h], p_two=p)
            res[key]['approx_vs_true'] = cmp_blk
            out['universes'][uname] = res
            dump()
            sb = res[key]['skill_scaled']['vs_hod_xH_true']
            print(f"  [{uname} {key}] APchk ok | ISS_hod_true all={sb['ISS_all_taus_mean']:+.4f} "
                  f"pos={sb['ISS_positive_taus_mean']:+.4f} | "
                  f"IGNdiff(model−hod_true) CI=[{sb['IGN_diff_blockboot']['ci'][0]:+.5f},{sb['IGN_diff_blockboot']['ci'][1]:+.5f}] "
                  f"p={sb['IGN_diff_blockboot']['p_two']} | "
                  f"rel_diff(hod) mean={cmp_blk['hod_xH']['S_rel_diff_mean']:+.4f}", flush=True)

    # ---- 回归对表 exp18 ----
    # 层1 确定性统计量（AP/IGN/ISS/分位数——B 与 RNG 无关）: 全 cell 逐位一致;
    # 层2 pop_xH bootstrap mean/ci: 仅全量模式 RNG 流首 cell（targets/1h）逐位一致
    # （exp18 每 cell 末尾 molchan bootstrap 消耗 RNG, 跨 cell 流位置不可对齐——结构差异非漂移;
    #   smoke 模式 B=100≠exp18 全量 1000, bootstrap 值不比）。
    j18u = j18['universes']
    drift_ap, drift_det, drift_boot = [], [], []
    first = True
    for uname in ['targets', 'targets_train']:
        for key in ['1h', '3h', '6h']:
            if uname not in out['universes'] or key not in out['universes'][uname]:
                continue
            mine = out['universes'][uname][key]
            old = j18u[uname][key]
            if abs(mine['AP']['model'] - old['AP']['model']) > 0 or abs(mine['AP']['hop1'] - old['AP']['hop1']) > 0:
                drift_ap.append(f'{uname}/{key}/AP')
            for sub in ['skill_scaled', 'hop1_skill_scaled']:
                a = mine[sub].get('vs_pop_xH'); b = old[sub].get('vs_pop_xH')
                for kk in ['IGN_model_mean', 'IGN_ref_mean', 'ISS_all_taus_mean',
                           'ISS_positive_taus_mean', 'IGN_diff_quantiles_5_25_50_75_95']:
                    if a[kk] != b[kk]:
                        drift_det.append(f'{uname}/{key}/{sub}/{kk}')
                if (not args.smoke) and first:
                    if (a['IGN_diff_blockboot']['mean'] != b['IGN_diff_blockboot']['mean']
                            or a['IGN_diff_blockboot']['ci'] != b['IGN_diff_blockboot']['ci']):
                        drift_boot.append(f'{uname}/{key}/{sub}/vs_pop_xH_boot')
            first = False
    out['regression_vs_exp18'] = dict(
        ap_and_deterministic_bitwise_identical=(len(drift_ap) == 0 and len(drift_det) == 0),
        drifted_ap=drift_ap, drifted_deterministic=drift_det,
        pop_xH_boot_first_cell_identical=(len(drift_boot) == 0), drifted_boot=drift_boot,
        note='层1=确定性输出（AP/IGN/ISS/分位数）全 cell 逐位; 层2=pop_xH bootstrap 仅全量模式首 cell '
             '（exp18 每 cell 末尾 molchan bootstrap 消耗 RNG 致跨 cell 流不可对齐; smoke B=100 不可比）——'
             '结构差异非漂移')
    dump()
    if drift_ap or drift_det or drift_boot:
        raise RuntimeError(f'回归检查失败: AP={drift_ap[:3]} det={drift_det[:3]} boot={drift_boot[:3]}')
    print('回归检查: AP/确定性统计量全 cell 逐位一致 ✓' +
          ('' if args.smoke else '; pop_xH bootstrap 首 cell 逐位一致 ✓'), flush=True)

    # ---- Q3 裁定: vs_hod ISS_all 是否恢复可读 ----
    q3 = {}
    for uname in out['universes']:
        for key in out['universes'][uname]:
            v = out['universes'][uname][key]['skill_scaled']['vs_hod_xH_true']['ISS_all_taus_mean']
            q3[f'{uname}/{key}'] = float(v)
    pathological = [k for k, v in q3.items() if v < -1.0]
    out['Q3_ISS_ratio_ruling'] = dict(
        iss_all_true=q3,
        pathological_cells=pathological,
        ruling=('仍病态（空闲小时 τ 参考率→0 使 ISS 比值爆炸, 结构性不随真积分修复）→ 成文只报 diff CI+合并口径, '
                'ISS 比值列保留但加病态注' if pathological else '恢复可读, ISS 列可引用'),
        note='病态根因=比值口径（ISS=1−IGN_m/IGN_r 的分母在空闲 τ 上→小）, 非 hod 近似所致')
    dump()

    json.dump(out, open(f'{R}/exp20_hod_true_integral.json', 'w'), indent=1,
              ensure_ascii=False, default=float)
    print(f'落盘 {R}/exp20_hod_true_integral.json; 总耗时 {(time.time()-t00)/60:.1f} min', flush=True)
    open(f'{R}/exp20.done', 'w').write('done: exp20 hod 参考真窗积分收口（含 BLa 同步裁定+近似对照+Q3 裁定）\n')


if __name__ == '__main__':
    main()
