#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exp21 mle_deg 暴露加权变体测试面一次性评估（R3 §12.5 清单5 = review1 B-M2/m6 + 拔高记账 V1 清欠）
问题: exp16 验证集选模中暴露加权 EB 变体（mle_deg, Bühlmann-Straub 暴露加权）gap=0.0422<0.05
  被简约规则压过（未上测试面）；0.05 bits 平手阈值非功效推导（阈值取 0 则 mle_deg 入选）——
  选模结论对阈值的敏感性须以测试面一次评估收口。
协议（预登记, 先落盘后运行; 与 exp16 同协议——测试集只碰一次）:
  E1 契约声明（Q7）: 本实验 mle_deg = 层级 Gamma-Poisson EB，先验 r_v~Gamma(α,β) 与暴露无关、
     暴露 e_v=deg(v)+1 仅进 Poisson 似然（λ_v=e_v·r_v, 即 GLM 语境 offset(log e) 语义）——
     边际似然 P(n_v|α,β,e_v)=Γ(α+n_v)/(Γ(α)n!)·(β/(β+e_v))^α·(e_v/(β+e_v))^n_v 为**精确边际**;
     后验均值 E[r_v|n_v]=(α+n_v)/(β+e_v)。声明: NB 语境下与 prior-precision weights /
     frequency weights 契约**不等价**（Poisson 下等价）, 本实验采用精确边际版（=exp16 已落盘实现,
     同协议要求——exp16 验证集选模即按此定义）。归因最干净（暴露只进强度不进先验）。
  E2 判定三分支（G1-mdeg, 先落盘后运行）:
     测试段 IGPE(hybrid_mle_deg − M_hybrid_W24_c[exp12 权威 lam]) 移动块 bootstrap(block=10,B=5000,
     seed=13, 与 exp16 margins 同口径) 95%CI:
       下界>0 → mle_deg 显著优 → 选模翻案（成文=权威变体改 mle_deg, 须联动 exp16 selected 披露）;
       上界<0 → mle_deg 有害 → 维持现版并报有害方向;
       CI 含 0  → 无显著差 → 维持现版 hybrid, mle_deg 写为「验证集被简约规则压过+测试面无显著差」。
     成文形态（TOST 语义, W3）: 方向+CI+「本研究不足以确立等价」（n=143 CI 半宽≫0.05 bits 故等价
     不可确立）——「无增量」裸写禁止。
  E3 平手阈值敏感性并报（Q8）: 0.05 bits 阈值下 exp16 验证集选 mom（gap=0.0422 被压过）;
     阈值取 0 则 mle_deg 入选——两阈值下选模分支差异与本测试面结果一并披露（成文义务）。
  E4 测试集只碰一次承诺: 全 train 拟合与全部前置检查在测试段评分之前; 测试段 143 事件评分代码
     执行恰一次。
正确性前置检查: eb_flag_mle(cntC_full, e_vec=deg) 重算 (α,β) 与 exp16 JSON
  flag_frailty.full_train.mle_deg_alpha/mle_deg_beta 逐位一致; 拟合收敛断言; λ 有限正断言。
输出: results/exp21_mle_deg_testface.json
"""
import sys, json, time
import numpy as np
sys.path.insert(0, './scripts')
sys.path.insert(0, './src')
from exp_patch_common import load_all, fit_constrained, score_Ktilde, eb_flag_q, moving_block_bootstrap, igpe_full, R
from lanl_pp import GraphHawkes, hour_profile, hour_coverage_seconds, igpe
from exp16_frailty_upgrade import eb_flag_mle, flag_q_from

TIE = 0.05


def main():
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
    deg = deg + 1.0   # e_v = deg+1（同 exp16）

    out = dict(exp='Exp21_mle_deg_testface',
               preregistered=dict(
                   E1_contract=('mle_deg=层级 Gamma-Poisson EB: 先验 r~Gamma(α,β) 与暴露无关, 暴露 e_v=deg+1 '
                                '仅进 Poisson 似然（offset 语义）, 边际似然精确; NB 语境与 precision-weights/'
                                'frequency-weights 契约不等价（Poisson 下等价）, 采用精确边际版=exp16 已落盘实现'),
                   E2_verdict_branches=('G1-mdeg: IGPE(hybrid_mle_deg − M_hybrid_W24_c) 块bootstrap(block=10,'
                                        'B=5000,seed=13) CI 下界>0→显著优翻案; 上界<0→有害; 含0→无显著差维持现版。'
                                        '成文=TOST 语义「方向+CI+等价不可确立」(n=143 半宽≫0.05), 禁裸写「无增量」'),
                   E3_tie_sensitivity='0.05 bits 阈值=exp16 验证集选 mom; 阈值 0 则 mle_deg 入选——两阈值分支差异并报',
                   E4_one_touch='全 train 拟合与前置检查先于测试段评分; 测试段 143 事件评分执行恰一次',
                   power_note='n_train=570/n_test=143; MDE≈1.10 bits（exp4b D3）≫0.05 平手阈值'),
               correctness_checks={}, tie_threshold_bits=TIE)
    json.dump(out, open(f'{R}/exp21_mle_deg_testface.json', 'w'), indent=1, ensure_ascii=False, default=float)
    print('预登记已落盘 exp21_mle_deg_testface.json（先落盘后运行）', flush=True)

    def dump():
        json.dump(out, open(f'{R}/exp21_mle_deg_testface.json', 'w'), indent=1,
                  ensure_ascii=False, default=float)

    # ---------- 前置正确性检查: EB (α,β) 与 exp16 落盘逐位一致 ----------
    a_mled, b_mled, nll_mled = eb_flag_mle(cntC_full, e_vec=deg)
    j16 = json.load(open(f'{R}/exp16_frailty_upgrade.json'))
    a16 = j16['flag_frailty']['full_train']['mle_deg_alpha']
    b16 = j16['flag_frailty']['full_train']['mle_deg_beta']
    ok_eb = bool(a_mled == a16 and b_mled == b16)
    out['correctness_checks']['eb_alpha_beta_vs_exp16'] = dict(
        mine=dict(alpha=a_mled, beta=b_mled), exp16=dict(alpha=a16, beta=b16), pass_=ok_eb)
    dump()
    if not ok_eb:
        raise RuntimeError(f'EB (α,β) 与 exp16 落盘不一致: mine=({a_mled},{b_mled}) exp16=({a16},{b16})')
    print(f'EB 检查通过: α={a_mled:.6g} β={b_mled:.6g} 与 exp16 逐位一致', flush=True)
    q_mled = flag_q_from(a_mled, b_mled, cntC_full, e_vec=deg)

    # ---------- 全 train 拟合（同 exp16 build_full/fit_constrained 口径） ----------
    m = GraphHawkes(t_all[:cut], list(dst_all[:cut]), list(src_all[:cut]), adj, nodes,
                    dst_extra=V_all, kernel='omori', gk='hop0', bg='hod24', W=86400.0,
                    qbg='custom', q_custom=q_mled)
    th, ll, a = fit_constrained(m, a_max=0.95, n_restarts=6, seed=23, t_end=t_cut)
    out['fit_full_train'] = dict(ll=ll, branch_a=a, kappa=float(np.exp(th[m.idx['log_kappa']])),
                                 mu0=float(np.exp(th[m.idx['log_mu0']])),
                                 c=float(th[m.idx['log_c']]), p=float(th[m.idx['log_p']]),
                                 degenerate=bool(a < 0.05 or a > 0.949),
                                 note='同 exp16 fit_constrained(n_restarts=6, seed=23, a_max=0.95)')
    # 对照: exp12 M_hybrid(q=mom) 与 M_boost(q=pop) 拟合参数（读权威 JSON, 零重拟合）
    j12 = json.load(open(f'{R}/exp12_boost_flag.json'))
    out['reference_fits_exp12'] = {k: j12['model_family'][k]['params'] for k in
                                   ['M_hybrid_W24h_c', 'M_boost_W24h_c']}
    dump()
    print(f"全 train 拟合: ll={ll:.2f} a={a:.4f} κ={out['fit_full_train']['kappa']:.4f} "
          f"μ0={out['fit_full_train']['mu0']:.3e}", flush=True)

    # ---------- 测试段评分（唯一一次） ----------
    print('== 测试段评分（唯一一次） ==', flush=True)
    t_te, dst_te, src_te = t_all[cut:], list(dst_all[cut:]), list(src_all[cut:])
    si0 = [m.vidx[d] for d in dst_all[:cut]]
    ctx = dict(pop=popC, m_h=m_h_full, R0=R0, flag_q=None, theta_iw=0.0, ECNT={'cnt': {}, 'tot': {}})
    la, lv, ra, rv = score_Ktilde(m, th, ['uniform', 'pop', 'hod_pop'], t_te, dst_te, src_te,
                                  list(t_all[:cut]), list(dst_all[:cut]), si0, ctx)
    lam_new = la
    lam_ref = np.load(f'{R}/exp12_lam_M_hybrid_W24_c.npy')   # exp12 权威 lam（防重算漂移）
    lam_boost = np.load(f'{R}/exp12_lam_M_boost_W24_c.npy')
    lam_flag = np.load(f'{R}/exp12_lam_M_flag.npy')
    # sanity 断言
    for nm, v in [('hybrid_mle_deg', lam_new), ('M_hybrid_W24_c', lam_ref)]:
        assert np.all(np.isfinite(v)) and np.all(v > 0), f'{nm} λ 含 NaN/非正'
    assert len(lam_new) == 143, f'测试段事件数 {len(lam_new)}≠143'
    g_vs_mhyb = np.log2(lam_new / lam_ref)
    g_vs_uni = np.log2(lam_new / ra['uniform'])
    st_abs = igpe_full(g_vs_uni)
    st = igpe_full(g_vs_mhyb)
    bb = moving_block_bootstrap(g_vs_mhyb, block=10, n_boot=5000, seed=13)
    out['test_face'] = dict(
        n=143,
        IGPE_vs_uniform=dict(bits=st_abs['IGPE_bits'], p=st_abs['p'], t_ar1=st_abs['t_ar1'], p_ar1=st_abs['p_ar1']),
        margin_vs_M_hybrid_W24_c=dict(
            IGPE_bits=st['IGPE_bits'], boot_ci=[bb['ci_lo'], bb['ci_hi']], p_boot=bb['p_boot_two'],
            median=float(np.median(g_vs_mhyb)), n_pos=int((g_vs_mhyb > 0).sum()),
            n_neg=int((g_vs_mhyb < 0).sum()), p_wilcoxon=st['p_wilcoxon'], n=143))
    np.save(f'{R}/exp21_margin_mle_deg_gt_M_hybrid_W24_c.npy', g_vs_mhyb)
    print(f"mle_deg vs uniform: {st_abs['IGPE_bits']:+.4f} (p={st_abs['p']:.3g})", flush=True)
    print(f"mle_deg − M_hybrid_W24_c: {st['IGPE_bits']:+.4f} CI=[{bb['ci_lo']:+.4f},{bb['ci_hi']:+.4f}] "
          f"p_boot={bb['p_boot_two']:.3g} median={np.median(g_vs_mhyb):+.4f} "
          f"正{int((g_vs_mhyb>0).sum())}/负{int((g_vs_mhyb<0).sum())}", flush=True)

    # seen/unseen 分层
    train_dst = set(dst_all[:cut])
    seen = np.array([d in train_dst for d in dst_te])
    strat = {}
    for nm, msk, lam_v in [('hybrid_mle_deg', seen, lam_new), ('hybrid_mle_deg_unseen', ~seen, lam_new)]:
        st_s = igpe(np.log2(lam_v[msk] / ra['uniform'][msk]))
        strat['seen' if nm == 'hybrid_mle_deg' else 'unseen'] = dict(
            n=int(msk.sum()), vs_uniform=st_s['IGPE_bits'], p=st_s['p'])
    out['D_stratified'] = strat

    # ---------- 判定（E2 三分支）+ E3 阈值敏感性 ----------
    ci_lo, ci_hi = bb['ci_lo'], bb['ci_hi']
    if ci_lo > 0:
        verdict = 'mle_deg 显著优（CI 下界>0）→ 选模翻案, 权威变体改 mle_deg（须联动 exp16 selected 披露）'
    elif ci_hi < 0:
        verdict = 'mle_deg 有害（CI 上界<0）→ 维持现版 hybrid 并报告有害方向'
    else:
        verdict = ('无显著差（CI 含 0）→ 维持现版 hybrid; mle_deg 写为「验证集被简约规则压过+测试面无显著差」; '
                   '成文按 TOST 语义: 方向 %+0.4f bits, CI [%+0.4f,%+0.4f], n=143 CI 半宽≫0.05 bits 故'
                   '「等价」不可确立（本研究不足以确立或排除）' % (st['IGPE_bits'], ci_lo, ci_hi))
    out['verdict_G1_mdeg'] = dict(branch=('significantly_better' if ci_lo > 0 else
                                          'harmful' if ci_hi < 0 else 'no_significant_difference'),
                                  text=verdict,
                                  tie_threshold_disclosure=dict(
                                      threshold_005=('exp16 验证集 gap=0.0422<0.05 → 简约规则选 mom（现版 hybrid）; '
                                                     '本测试面 CI 判定见上'),
                                      threshold_000=('阈值取 0 则 exp16 验证集选 mle_deg; 本测试面 CI 为其'
                                                     '实际前瞻效果——两阈值下选模分支不同, 成文必并报'),
                                      equivalence_note='n=143 CI 半宽≫0.05 bits, TOST 语义下等价不可确立'))
    # 等暴露 MLE 版（exp16 hybrid_mle_omori 测试面）并置对照（验证集第二变体, 数字读 exp16 权威）
    out['exp16_mle_equal_exposure_reference'] = dict(
        test_IGPE_vs_uniform=j16['test_IGPE'].get('hybrid_mle_omori'),
        margin=j16['margins'].get('hybrid_mle_omori_gt_M_hybrid_W24_c'),
        note='等暴露 MLE 版=exp16 已测面变体; 本实验为暴露加权 mle_deg 版补测面（V1 清欠）')
    dump()

    json.dump(out, open(f'{R}/exp21_mle_deg_testface.json', 'w'), indent=1,
              ensure_ascii=False, default=float)
    print(f'落盘 {R}/exp21_mle_deg_testface.json; 总耗时 {(time.time()-t00)/60:.1f} min', flush=True)
    open(f'{R}/exp21.done', 'w').write('done: exp21 mle_deg 暴露加权变体测试面一次评估\n')


if __name__ == '__main__':
    main()
