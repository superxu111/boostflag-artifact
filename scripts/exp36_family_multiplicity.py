#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exp36 校准面 12 格头条族校正收口（R3 双审 A-M2/B-M2 双席同题 + R3 定向创新方案一）
问题: §12.4 调和表第 3 行以「model−pop_xH/hod_true 12/12 格 CI 排零（模型优）」为正技能判定词
  （exp_summary.md:191），但该 12 格检验从未做族校正: 最弱格 targets 1h vs hod_true p=0.008
  （B=1000）×12 Bonferroni=0.096 不存活（项目自身纪律: exp4b ×8 / Knox ×12 / exp13 ×6 / K6 ×24）。
  本实验按定向创新方案一收口（事后定族显式申报）:
  F1 显式定族: primary=vs_pop_xH 6 格 + vs_hod_xH_true 6 格=12（=调和表第 3 行判定词格集）;
     expanded=+vs_BLa_true 6 格=18（Q5 收口后同族 hod 型参考）。两族均标注 post-hoc（数字已见后定义）。
  F2 格级三口径并报: raw 块 bootstrap p（B=1000 原版 + B=5000 高分辨率版）+BH-FDR+Bonferroni。
  F3 主检验=Cauchy combination test（Liu & Xie 2020 JASA 115(529):393-402, DOI
     10.1080/01621459.2018.1554485）: T=Σω_j·tan{(0.5−p_j)π}, 等权 ω=1/12, 组合 p=0.5−arctan(T)/π;
     对嵌套窗强相关稳健。警示（Biometrics 2025 ujaf001）: vanilla CCT 非精确有效→经验校准轨道必报。
     p 下限用 Davison-Hinkley (r+1)/(B+1) 语义（exp18 呈报口径, 防 p=0 进 tan 发散）。
  F4 经验校准轨道: 联合块重抽（同一 τ 索引矩阵用于全部格→格间相关自然保留）B_cal=5000,
     零假设化=逐格中心化 dll; T* 经验零分布 vs Cauchy 理论分位数 QQ 表+经验组合 p（(r+1)/(B+1)）。
  F5 敏感性（不进判定）: ①Stouffer 相关校正（单侧模型优方向, C_j=Φ⁻¹(1−p1_j), 相关矩阵=联合重抽
     均值差样本相关, Z=1'C/√(1'Σ̂1)）; ②HMP 等权数值（Goeman et al. 2019 PNAS 争议: 非单点集无强
     FWER → 只报数值+争议注）。Brown(1975) 精神的单侧相关校正由①实现。
  正确性前置（防修复自身引新错第四次复发, A-M1 教训）: exp20 管线零改动复用（import exp20 函数
  +同 seed=31 RNG 流同调用序）→ 重建版 12+6 格 obs/ci/p_two 与 exp20_hod_true_integral.json 落盘值
  逐位对表, 任一不等即 RuntimeError 终止。
  铁律注: 本实验为已见数据（exp20 测试段评分）的统计后处理, 不产生新的模型-数据接触、不新增评分;
  族为事后定义并显式申报, 成文不得称「预登记族」。
输出: results/exp36_family_multiplicity.json
"""
import sys, json, time
import numpy as np
from scipy import stats
sys.path.insert(0, './scripts')
sys.path.insert(0, './src')
import exp20_hod_true_integral as e20
from exp_patch_common import load_all, eb_flag_q, R
from lanl_pp import GraphHawkes, hour_profile
from exp15_hybrid_causal import grid_scores_causal_v2
from sklearn.metrics import average_precision_score

B_HI = 5000
B_CAL = 5000
RNG_HI = np.random.default_rng(36001)   # 高分辨率轨道独立种子
RNG_CAL = np.random.default_rng(36002)  # 联合校准轨道独立种子


def bh_adjust(ps):
    """Benjamini-Hochberg step-up, 返回校正后 p 数组"""
    ps = np.asarray(ps, float); m = len(ps)
    order = np.argsort(ps)
    ranked = ps[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(ranked, 0, 1)
    return out


def cct(ps, weights=None):
    """Cauchy combination test; p 下限裁剪由调用方完成 (r+1)/(B+1)"""
    ps = np.asarray(ps, float)
    w = np.full(len(ps), 1.0 / len(ps)) if weights is None else np.asarray(weights, float)
    T = float(np.sum(w * np.tan((0.5 - ps) * np.pi)))
    p = float(0.5 - np.arctan(T) / np.pi)
    return dict(T_stat=T, p_combined=max(p, 5e-324), n=len(ps), weights='equal')


def vec_block_boot_means(m_row, n_tau, block, B, rng):
    """m_row: (n_tau, K) 每格 τ 均值矩阵; 联合块重抽（同一 τ 索引用于全部 K 格）→ (B, K)"""
    nb = int(np.ceil(n_tau / block))
    starts = rng.integers(0, n_tau - block + 1, size=(B, nb))
    tau_idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)[:, :n_tau]
    return m_row[tau_idx].mean(axis=1)      # (B, K) — mean over τ of τ-均值 = 逐格重抽均值差


def main():
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
    Hs = [3600.0, 3 * 3600.0, 6 * 3600.0]

    m_hyb = GraphHawkes(t_all[:cut], list(dst_all[:cut]), list(src_all[:cut]), adj, nodes,
                        dst_extra=V_all, kernel='omori', gk='hop0', bg='hod24', W=86400.0,
                        qbg='custom', q_custom=q_flag)
    m_hyb.th = np.load(f'{R}/exp12_th_M_hybrid_W24h_c.npy')
    m_hop1 = GraphHawkes(t_all[:cut], list(dst_all[:cut]), list(ev['src'][:cut]), adj, nodes,
                         dst_extra=V_all, kernel='omori', gk='hop1', bg='hod24', W=86400.0, qbg='pop')
    m_hop1.th = np.load(f'{R}/exp4_th.npy')
    j15 = json.load(open(f'{R}/exp15_hybrid_causal.json'))
    j13 = json.load(open(f'{R}/exp13_causal_forecast.json'))

    out = dict(exp='Exp36_family_multiplicity',
               motive=('A-M2/B-M2 双席同题: 12/12 排零校准面头条无族校正（最弱格 p=0.008 ×12=0.096 '
                       '不过项目自身 Bonferroni 标准）→ 方案一「显式定族+相关性稳健组合检验」收口'),
               preregistered=dict(
                   F1='显式定族 primary 12=pop_xH6+hod_true6（调和表第 3 行格集）+expanded 18（+BLa6）, post-hoc 申报',
                   F2='格级三口径: raw p（B=1000 原版+B=5000 高分辨率）+BH+Bonferroni',
                   F3='主检验 CCT（Liu&Xie 2020 JASA, vanilla 非精确有效警示必报; p 下限 (r+1)/(B+1)）',
                   F4='经验校准轨道: 联合块重抽 B_cal=5000, 逐格中心化零假设, T* 经验零分布+经验组合 p',
                   F5='敏感性: Stouffer 相关校正（单侧）+HMP（争议注, 不进判定）',
                   hard_gate='exp20 管线零改动复用, 重建 18 格 obs/ci/p_two 逐位对表失败即终止'),
               post_hoc_declaration='事后定族: 12 格/18 格族在本实验前未预登记, 格 p 值已于 exp18/exp20 可见; '
                                    '本实验将头条判定词从单一格 CI 排零升级为「显式族+组合检验」口径, 成文不得称预登记',
               correctness_checks={}, universes={},
               family_correction={}, combined_tests={})
    json.dump(out, open(f'{R}/exp36_family_multiplicity.json', 'w'), indent=1, ensure_ascii=False, default=float)
    print('预登记已落盘 exp36_family_multiplicity.json', flush=True)

    def dump():
        json.dump(out, open(f'{R}/exp36_family_multiplicity.json', 'w'), indent=1, ensure_ascii=False, default=float)

    # ---------- F0: exp20 管线复刻 + dll 收集（同 seed=31 RNG 流同调用序） ----------
    RNG = np.random.default_rng(31)          # 与 exp20 同 seed（RNG 流对表前提）
    tt_all = np.asarray(t_all, float)
    ds_all = np.asarray(dst_all)
    dll_bank = {}      # (uname, key, ref) -> dll 矩阵 (n_tau, nU)
    meta_bank = {}
    for uname, U in universes.items():
        uidx = {u: i for i, u in enumerate(U)}
        nU = len(U)
        qpopU = np.array([popC[cidxC[u]] for u in U]); qpopU = qpopU / qpopU.sum()
        seen_mask_U = np.array([u in set(universes['targets_train']) for u in U])
        for H in Hs:
            key = f'{int(H // 3600)}h'
            taus, S_m, S_p, Y = grid_scores_causal_v2(m_hyb, m_hyb.th, U, t_all, list(dst_all),
                                                      t_cut, T_end, H, causal=True, q_popvec=popC)
            n_tau = len(taus)
            yf = Y.ravel()
            ap_m = float(average_precision_score(yf, S_m.ravel())) if yf.sum() else float('nan')
            j15r = j15['universes'][uname][f'24h_{key}']
            chk = bool(abs(ap_m - j15r['PR_AUC_model']) < 1e-9)
            _, S_m1, S_p1, Y1 = grid_scores_causal_v2(m_hop1, m_hop1.th, U, t_all, list(dst_all),
                                                      t_cut, T_end, H, causal=True, q_popvec=popC)
            ap_m1 = float(average_precision_score(yf, S_m1.ravel())) if yf.sum() else float('nan')
            j13r = j13['universes'][uname][key]
            chk1 = bool(abs(ap_m1 - j13r['PR_AUC_model_causal']) < 1e-9)
            out['correctness_checks'][f'{uname}_{key}'] = dict(hybrid_ap_vs_exp15=chk, hop1_ap_vs_exp13=chk1)
            if not (chk and chk1):
                raise RuntimeError(f'{uname}/{key} AP 对表失败（前置硬闸）')

            S_pop_xH = np.tile(R0 * H * qpopU, (n_tau, 1))
            S_hod_true = np.zeros((n_tau, nU)); S_bla_true = np.zeros((n_tau, nU))
            for gi, tau in enumerate(taus):
                cov = e20.hour_coverage_seconds(tau, tau + H)
                m_h_eff = float((m_h * cov).sum() / H)
                S_hod_true[gi] = m_h_eff * R0 * H * qpopU
                hist = np.where(tt_all < tau)[0]
                counts = np.zeros(nU)
                for d in ds_all[hist[(tt_all[hist] >= tau - H) & (tt_all[hist] < tau)]]:
                    if d in uidx:
                        counts[uidx[d]] += 1.0
                S_bla_true[gi] = m_h_eff * (counts + qpopU)
            refs = dict(pop_xH=S_pop_xH, hod_xH_true=S_hod_true, BLa_true=S_bla_true)
            # dll 收集（确定性, 不耗 RNG）
            ll_m = e20.ign_curve_log2(S_m, Y)[1]
            for rname, S_r in refs.items():
                dll_bank[(uname, key, rname)] = ll_m - e20.ign_curve_log2(S_r, Y)[1]
                meta_bank[(uname, key, rname)] = dict(n_tau=int(n_tau), nU=int(nU))
            # RNG 流: 与 exp20 逐位同序（skill_scaled 3 ref + stratified + hop1_skill_scaled 2 ref + stratified）
            sk = e20.skill_block(S_m, Y, refs, n_tau, nU, seen_mask_U, 1000, RNG)
            sk1 = e20.skill_block(S_m1, Y, dict(pop_xH=S_pop_xH, hod_xH_true=S_hod_true),
                                  n_tau, nU, seen_mask_U, 1000, RNG)
            out['universes'][uname] = out['universes'].get(uname, {})
            out['universes'][uname][key] = dict(
                skill_scaled={r: sk[f'vs_{r}']['IGN_diff_blockboot'] for r in refs},
                hop1_ran=True)
            dump()
            print(f'  [{uname}/{key}] 管线复刻完成（n_tau={n_tau}, nU={nU}）', flush=True)

    # ---------- 逐位对表 exp20（硬闸） ----------
    j20 = json.load(open(f'{R}/exp20_hod_true_integral.json'))
    fails = []
    for uname in universes:
        for key in ['1h', '3h', '6h']:
            for rname in ['pop_xH', 'hod_xH_true', 'BLa_true']:
                mine = out['universes'][uname][key]['skill_scaled'][rname]
                ref = j20['universes'][uname][key]['skill_scaled'][f'vs_{rname}']['IGN_diff_blockboot']
                if mine['mean'] != ref['mean'] or list(mine['ci']) != list(ref['ci']) or mine['p_two'] != ref['p_two']:
                    fails.append(f'{uname}/{key}/{rname}')
    out['correctness_checks']['exp20_bitwise_regression'] = dict(
        pass_=len(fails) == 0, failed_cells=fails,
        note='18 格 mean/ci/p_two 与 exp20_hod_true_integral.json 落盘值逐位相等（同 seed=31 RNG 流同调用序）')
    dump()
    if fails:
        raise RuntimeError(f'exp20 逐位对表失败: {fails}（防修复自身引新错, 终止）')
    print('exp20 逐位对表: 18 格 obs/ci/p_two 全部相等 ✓', flush=True)

    # ---------- F1+F2: 显式定族 + 格级三口径 ----------
    def cell_p(dll, n_tau, nU, B, rng):
        """向量化 τ-均值块 bootstrap（与 e20.block_boot_mean_diff 同统计量; 求和顺序差≈1e-16 级）"""
        m_row = dll.mean(axis=1)[:, None]                      # (n_tau,1): τ 均值（nU 全取→等价）
        mm = vec_block_boot_means(m_row, n_tau, 6, B, rng)[:, 0]
        lo, hi = np.percentile(mm, [2.5, 97.5])
        r = int((mm <= 0).sum()); s = int((mm >= 0).sum())
        p_two = float(min(1.0, 2.0 * min(r, s) / B))           # 频率口径; 呈报时配 (r+1)/(B+1) 语义
        return dict(mean=float(dll.mean()), ci=[float(lo), float(hi)], p_two=p_two,
                    p_two_dh=float(min(1.0, 2.0 * min(r + 1, s + 1) / (B + 1))), B=B)

    cells = []   # (uname, key, rname, family_tag)
    for uname in universes:
        for key in ['1h', '3h', '6h']:
            for rname in ['pop_xH', 'hod_xH_true']:
                cells.append((uname, key, rname, 'primary12'))
            cells.append((uname, key, 'BLa_true', 'expanded18'))

    # ---- 实现等价性硬闸: 同一 starts 序列下, 向量化 τ-均值版与 exp20 逐格 gather 版须同 diffs ----
    # （exp20 逐位回归已证数据管线一致; 此处证两种 bootstrap 实现在同输入同输出意义上等价——
    #   跨种子 p_two 差异是 MC 噪声, 不能也不应作为硬闸）
    eq_rows = []
    for (uname, key, rname) in [('targets', '1h', 'pop_xH'), ('targets', '6h', 'hod_xH_true'),
                                ('targets_train', '3h', 'BLa_true')]:
        dll = dll_bank[(uname, key, rname)]
        n_tau, nU = dll.shape
        block = 6; B_eq = 400
        rr = np.random.default_rng(46000)
        starts = rr.integers(0, n_tau - block + 1, size=(B_eq, int(np.ceil(n_tau / block))))
        tau_idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(B_eq, -1)[:, :n_tau]
        m_row = dll.mean(axis=1)                       # τ 均值向量 (n_tau,)
        diffs_vec = m_row[tau_idx].mean(axis=1)        # 向量化版
        cell_off = np.arange(nU)
        diffs_gather = np.empty(B_eq)
        for b in range(B_eq):                          # exp20 gather 版（同 starts 展开为同一 tau_idx）
            idx = (tau_idx[b][:, None] * nU + cell_off[None, :]).ravel()
            diffs_gather[b] = dll.ravel()[idx].mean()
        max_d = float(np.max(np.abs(diffs_vec - diffs_gather)))
        eq_rows.append(dict(cell=f'{uname}/{key}/{rname}', max_abs_diff=max_d,
                            pass_=bool(max_d < 1e-12)))
        if max_d >= 1e-12:
            raise RuntimeError(f'实现等价性硬闸失败: {uname}/{key}/{rname} max|Δdiffs|={max_d}')
    out['correctness_checks']['vec_reimpl_equivalence'] = dict(
        pass_=True, rows=eq_rows,
        note='同一 starts 序列下向量化 τ-均值版与 exp20 逐格 gather 版逐元素同 diffs（<1e-12, 浮点求和'
             '顺序差）; 两种子下 p_two 的差异属 MC 噪声（二项计数 sd≈sqrt(p(1−p)/B)）, 不作硬闸——'
             '数据管线一致性已由 exp20_bitwise_regression 逐位硬闸证明')

    eq_checks = []
    grid = {}
    for ci_, (uname, key, rname, fam) in enumerate(cells):
        dll = dll_bank[(uname, key, rname)]
        n_tau = meta_bank[(uname, key, rname)]['n_tau']; nU = meta_bank[(uname, key, rname)]['nU']
        hi_res = cell_p(dll, n_tau, nU, B_HI, RNG_HI)
        ref_p = j20['universes'][uname][key]['skill_scaled'][f'vs_{rname}']['IGN_diff_blockboot']['p_two']
        grid[f'{uname}/{key}/{rname}'] = dict(
            family=fam, mean=hi_res['mean'], ci_hi_res=hi_res['ci'],
            p_raw_B1000=ref_p,
            p_raw_B5000=hi_res['p_two'], p_two_dh_B5000=hi_res['p_two_dh'], B_hi=B_HI)

    for fam, m_ in [('primary12', 12), ('expanded18', 18)]:
        sub = {k: v for k, v in grid.items() if v['family'] == fam}
        ks = list(sub)
        for ver in ['p_raw_B1000', 'p_raw_B5000']:
            ps = np.array([sub[k][ver] for k in ks])
            bonf = np.minimum(ps * m_, 1.0)
            bh = bh_adjust(ps)
            for i, k in enumerate(ks):
                sub[k][f'bonferroni_x{m_}_{ver.split("_")[2]}'] = float(bonf[i])
                sub[k][f'BH_FDR_{ver.split("_")[2]}'] = float(bh[i])
        out['family_correction'][fam] = dict(
            n_cells=m_, definition=('model−pop_xH 6 格 + model−hod_xH_true 6 格' if fam == 'primary12'
                                    else 'primary12 + model−BLa_true 6 格'),
            cells=sub,
            worst_cell=max(ks, key=lambda k: sub[k]['p_raw_B5000']),   # 最弱格=最不显著（p 最大）
            note=('primary12=§12.4 调和表第 3 行判定词格集; expanded18=Q5 收口后同族 hod 型参考全集; '
                  '两族均 post-hoc（见 post_hoc_declaration）'))

    # ---------- F3: CCT 主检验 ----------
    CAUCHY_Q = {0.50: 0.0, 0.90: 3.077683537175253, 0.95: 6.313751514775055,
                0.99: 31.82051595377389, 0.999: 318.30883989456273}
    for fam in ['primary12', 'expanded18']:
        sub = out['family_correction'][fam]['cells']
        ks = list(sub)
        res_fam = {}
        for ver, cap in [('B1000', 1001), ('B5000', B_HI + 1)]:
            raw = np.array([sub[k][f'p_raw_{ver}'] for k in ks])
            ps = np.maximum(raw, 1.0 / cap)                            # DH (r+1)/(B+1) 下限
            c = cct(ps)
            c['p_floor_applied'] = 'Davison-Hinkley (r+1)/(B+1)'
            c['n_cells_at_floor'] = int((raw == 0.0).sum())
            res_fam[f'CCT_{ver}'] = c
        out['combined_tests'][fam] = res_fam
    dump()

    # ---------- F4: 经验校准轨道（联合重抽, primary12） ----------
    fam = 'primary12'
    ks12 = [k for k in grid if grid[k]['family'] == 'primary12']
    m_mat = np.column_stack([dll_bank[tuple(k.split('/'))].mean(axis=1) for k in ks12])   # (n_tau,12)
    n_tau12 = m_mat.shape[0]
    mc = m_mat - m_mat.mean(axis=0, keepdims=True)          # 逐格中心化=零假设化
    Mstar = vec_block_boot_means(mc, n_tau12, 6, B_CAL, RNG_CAL)     # (B_cal,12) 联合重抽
    # 逐格经验双侧 p（批内对称参照, (r+1)/(B+1) 语义）
    pstar = np.empty_like(Mstar)
    for j in range(Mstar.shape[1]):
        col = Mstar[:, j]
        col_sorted = np.sort(col)
        n_ge = np.searchsorted(col_sorted, col, side='right')      # #(x_k >= x_b), 含自身
        n_le = np.searchsorted(col_sorted, col, side='left')       # 上面减自身: n_ge-1
        pstar[:, j] = 2.0 * np.minimum(n_le + 1, n_ge) / (B_CAL + 1.0)   # (r+1)/(B+1) 语义
    eps = 1.0 / (B_CAL + 1.0)
    pstar = np.clip(pstar, eps, 1.0 - eps)   # 对称截断: p*→1 时 tan((0.5−p)π)→−∞ 发散, 与上尾校准无关
    Tstar = np.tan((0.5 - pstar) * np.pi).mean(axis=1)             # 等权 ω=1/12 → mean
    p_obs = np.array([max(grid[k]['p_raw_B5000'], 1.0 / (B_HI + 1)) for k in ks12])
    T_obs = float(np.tan((0.5 - p_obs) * np.pi).mean())
    emp_p = float((1 + int((Tstar >= T_obs).sum())) / (B_CAL + 1))
    emp_q = {f'q{int(q*100)}': float(np.percentile(Tstar, q * 100)) for q in [0.5, 0.9, 0.95, 0.99]}
    cal = dict(B_cal=B_CAL, T_obs_B5000=T_obs, empirical_p_combined=emp_p,
               empirical_Tstar_quantiles=emp_q, cauchy_theory_quantiles=CAUCHY_Q,
               analytic_p_B5000=out['combined_tests']['primary12']['CCT_B5000']['p_combined'],
               zero_hypothesis='逐格中心化 dll（dll−mean）经同一 τ 索引联合块重抽→格间相关自然保留',
               pstar_note='格内经验双侧 p 用批内对称参照（(r+1)/(B+1) 近似, 上限 2B/(B+1) 截断至 1）')
    ratio = emp_p / cal['analytic_p_B5000']
    cal['emp_over_analytic_ratio'] = float(ratio)
    if 0.5 <= ratio <= 2.0:
        cal['calibration_verdict'] = ('经验组合 p/解析组合 p=%.2f（0.5~2 内）→ vanilla CCT 上尾校准可接受'
                                      '; 经验 p 并报为稳健口径' % ratio)
    else:
        cal['calibration_verdict'] = ('经验组合 p/解析组合 p=%.2f 超 0.5~2 → vanilla CCT 校准警示'
                                      '（Biometrics 2025 ujaf001 预警情形）, 判定以经验组合 p 为准' % ratio)
    cal['qq_note'] = ('T* 分位数表基于对称截断 p*（下尾发散系经验 p 边界效应, 与上尾校准无关）; '
                      '校准主判据=经验/解析组合 p 比值')
    out['combined_tests']['primary12']['empirical_calibration'] = cal

    # ---------- F5: 敏感性（Stouffer 相关校正 + HMP） ----------
    means_obs = m_mat.mean(axis=0)                          # 12 格方向（全负=模型优, 读数在案）
    p1 = np.empty(12)                                       # 单侧（模型优方向）经验 p
    for j in range(12):
        col = Mstar[:, j]                                   # 零假设（中心化）联合重抽
        if means_obs[j] < 0:
            p1[j] = (1 + int((col <= means_obs[j]).sum())) / (B_CAL + 1)   # P(null ≤ obs)
        else:
            p1[j] = (1 + int((col >= means_obs[j]).sum())) / (B_CAL + 1)
    C = stats.norm.ppf(1.0 - p1)
    Sighat = np.corrcoef(Mstar.T)
    var_tot = float(np.ones(12) @ Sighat @ np.ones(12))
    Z_st = float(C.sum() / np.sqrt(max(var_tot, 1e-12)))
    p_st = float(1 - stats.norm.cdf(Z_st))
    ps_main = np.array([max(grid[k]['p_raw_B5000'], 1.0 / (B_HI + 1)) for k in ks12])
    hmp = float(12.0 / np.sum(1.0 / ps_main))
    out['combined_tests']['primary12']['sensitivity'] = dict(
        stouffer_corr=dict(Z=Z_st, p_one_sided_model_better=p_st,
                           sigma_corr_note='Σ̂=联合重抽均值差 12×12 样本相关（B_cal 联合）',
                           side_check=dict(all_cells_model_better=bool((means_obs < 0).all()),
                                           means_obs=[float(x) for x in means_obs])),
        HMP=dict(value=hmp, equal_weight=True,
                 ruling='敏感性数值并报, 不进判定——Goeman et al. 2019 PNAS 116(21):10354: 非单点集'
                        '无强 FWER 控制、中度正相关下 I 类膨胀; Wilson 2019 PNAS 回应争议在案'),
        note='Stouffer 相关校正=Brown(1975) 单侧合成的经验方差校正实现; 两敏感性轨道均不改变主判定')
    out['runtime_min'] = (time.time() - t00) / 60
    dump()
    print(f"CCT primary12: B1000 p={out['combined_tests']['primary12']['CCT_B1000']['p_combined']:.3e}, "
          f"B5000 p={out['combined_tests']['primary12']['CCT_B5000']['p_combined']:.3e}, "
          f"经验 p={emp_p:.4f}", flush=True)
    print(f"Stouffer 相关校正: Z={Z_st:.3f} 单侧 p={p_st:.3e}")
    wc = out['family_correction']['primary12']['worst_cell']
    print(f"Bonferroni ×12 最弱格 {wc}: {out['family_correction']['primary12']['cells'][wc]['bonferroni_x12_B5000']:.4f}")
    print(f'落盘 {R}/exp36_family_multiplicity.json; 总耗时 {out["runtime_min"]:.1f} min', flush=True)
    open(f'{R}/exp36.done', 'w').write('done: exp36 校准面 12 格族校正收口（CCT+BH+Bonf+经验校准+敏感性）\n')


if __name__ == '__main__':
    main()
