#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
golden_master_check.py — 权威数字冻结表（golden master）生成与机械校验（R3 §12.5 清单 10/W5）
背景: 誊写漂移三轮复发（R1 B-m1→R2 P1→R3 清单 1）——软件工程实证（DOCER/docrot）表明一次性
  sweep 会衰减, 须机械 diff 进轮末闸门。
用法:
  python scripts/golden_master_check.py --freeze   # 从权威 JSON 快照生成 results/golden_master.json
  python scripts/golden_master_check.py            # 逐键数值比对（默认模式, 供轮末闸门调用）
语义: 数值比较（float 逐位 ==, NaN 双侧同判）非字符串 diff; 验证一致性非正确性——
  golden master 会冻结既有 bug, 故只在 items 1–9 修完后的冻结时点建表（本时点=R3 收口）。
退出码: 全部一致=0; 任何漂移=1（diff 列表打 stdout）。
"""
import sys, json, argparse

R = './results'
GM = f'{R}/golden_master.json'

# 键路径表：文件 -> [键路径 ...]（点分层; 列表用 [i]）
SPEC = {
    'exp4b_rescored.json': [
        'main_hop1.vs_uniform.p_wilcoxon', 'main_hop1.vs_pop.p_wilcoxon',
        'main_hop1.vs_hod_pop.p_wilcoxon', 'main_hop1.vs_edgecount.p_wilcoxon',
        'main_hop1.vs_recency1.p_wilcoxon', 'main_hop1.vs_recency24.p_wilcoxon',
        'main_hop1.vs_knox_rule.p_wilcoxon', 'main_hop1.vs_iw.p_wilcoxon',
        'main_hop1.vs_uniform.IGPE_bits', 'main_hop1.vs_pop.IGPE_bits',
        'main_hop1.vs_hod_pop.IGPE_bits', 'main_hop1.vs_knox_rule.IGPE_bits',
        'main_hop1.D3_power',
    ],
    'exp13_causal_forecast.json': [
        'universes.targets.1h.boot_model_minus_persistence.p_two',
        'universes.targets.3h.boot_model_minus_persistence.p_two',
        'universes.targets.6h.boot_model_minus_persistence.p_two',
        'universes.targets_train.1h.boot_model_minus_persistence.p_two',
        'universes.targets_train.6h.boot_model_minus_persistence.p_two',
        'universes.targets.1h.PR_AUC_model_causal', 'universes.targets.6h.PR_AUC_model_causal',
    ],
    'exp12_boost_flag.json': [
        'model_family.M_hybrid_W24h_c.params.mu0', 'model_family.M_hybrid_W24h_c.params.kappa',
        'model_family.M_hybrid_W24h_c.branch_a', 'model_family.M_boost_W24h_c.params.mu0',
        'model_family.M_boost_W24h_c.branch_a', 'model_family.M_hybrid_W24h_c.train_ll',
        'test_IGPE.M_hybrid_W24_c.vs_uniform',
        'margins.M_hybrid_W24_c_gt_M_flag.IGPE_bits',
        'margins.M_hybrid_W24_c_gt_M_boost_W24_c.IGPE_bits',
        'verdict',
    ],
    'exp11_constrained_ablation.json': [
        'P3_constrained_forward_IGPE[0].AIC_constrained',
        'P3_constrained_forward_IGPE[0].IGPE_vs_uniform',
        'P2_best_constrained.kernel',
    ],
    'exp15_hybrid_causal.json': [
        'universes.targets.24h_1h.PR_AUC_model', 'universes.targets.24h_6h.PR_AUC_model',
        'universes.targets_train.24h_6h.model_minus_persistence.p_two',
    ],
    'exp16_frailty_upgrade.json': [
        'selected.val_IGPE', 'selected.gap_to_best_raw', 'flag_frailty.full_train.mle_deg_alpha',
        'flag_frailty.full_train.mle_deg_beta',
    ],
    'exp18_skill_scale_protocol.json': [
        'universes.targets.1h.skill_scaled.vs_pop_xH.ISS_all_taus_mean',
        'universes.targets.3h.skill_scaled.vs_pop_xH.ISS_all_taus_mean',
        'universes.targets.6h.skill_scaled.vs_pop_xH.ISS_all_taus_mean',
        'universes.targets_train.1h.skill_scaled.vs_pop_xH.ISS_all_taus_mean',
        'universes.targets.1h.AP.model', 'universes.targets.6h.AP.hop1',
        'molchan(removed)', 'dca(removed)',
    ],
    'exp19_ts_kernel_fulltrain.json': [
        'omori_reference.ll', 'twostage.ts72h.w_short', 'twostage.ts7d.ll',
    ],
    'exp20_hod_true_integral.json': [
        'universes.targets.1h.skill_scaled.vs_hod_xH_true.IGN_diff_blockboot.ci[0]',
        'universes.targets.1h.skill_scaled.vs_hod_xH_true.IGN_diff_blockboot.ci[1]',
        'universes.targets.6h.skill_scaled.vs_hod_xH_true.IGN_diff_blockboot.ci[0]',
        'universes.targets.6h.skill_scaled.vs_hod_xH_true.IGN_diff_blockboot.ci[1]',
        'universes.targets_train.6h.skill_scaled.vs_hod_xH_true.IGN_diff_blockboot.ci[0]',
        'universes.targets.1h.skill_scaled.vs_BLa_true.IGN_diff_blockboot.p_two',
        'universes.targets_train.1h.skill_scaled.vs_BLa_true.IGN_diff_blockboot.ci[0]',
        'Q3_ISS_ratio_ruling.ruling',
    ],
    'exp21_mle_deg_testface.json': [
        'test_face.margin_vs_M_hybrid_W24_c.IGPE_bits',
        'test_face.margin_vs_M_hybrid_W24_c.boot_ci[0]',
        'test_face.margin_vs_M_hybrid_W24_c.boot_ci[1]',
        'test_face.margin_vs_M_hybrid_W24_c.p_boot',
        'verdict_G1_mdeg.branch',
    ],
    'exp22_nofrailty_contrast.json': [
        'N1_real_train_contrast.ratios_nofr_over_fr.mu0',
        'N1_real_train_contrast.ratios_nofr_over_fr.branch_a',
        'N2_verdict.summary_median_B_over_true.mu0_bias_B_median',
        'N2_verdict.summary_median_B_over_true.a_bias_B_median',
        'N2_verdict.plan',
    ],
    # ---- R4 补实验轮新增（exp35/36/37） ----
    'exp35_d2_resim.json': [
        'N4a_defect_diagnosis.segmented_mean', 'N4a_defect_diagnosis.defective_mean',
        'N4a_defect_diagnosis.theory_total', 'N4a_defect_diagnosis.defective_underestimate_pct',
        'D2_resim.p.lo', 'D2_resim.p.hi', 'D2_resim.p.median',
        'D2_resim.mu0.lo', 'D2_resim.mu0.hi',
        'D2_resim.eta.lo', 'D2_resim.eta.hi', 'D2_resim.n_ok',
        'comparison_vs_defective_archive.P2_ruling.A2_significant_supercritical',
    ],
    'exp36_family_multiplicity.json': [
        'family_correction.primary12.worst_cell',
        'combined_tests.primary12.CCT_B5000.p_combined',
        'combined_tests.primary12.CCT_B1000.p_combined',
        # ---- R4 注记收口轮新增（§16.5 清单 4③: CCT 格级双侧下限口径重算, 与单侧版双口径并报） ----
        'combined_tests.primary12.CCT_B5000_two_sided_floor.T_stat',
        'combined_tests.primary12.CCT_B5000_two_sided_floor.p_combined',
        'combined_tests.primary12.empirical_calibration.empirical_p_combined',
        'combined_tests.primary12.empirical_calibration.emp_over_analytic_ratio',
        'combined_tests.primary12.sensitivity.stouffer_corr.Z',
        'narrowing_audit.summary.mid_narrow_range_pct[0]',
        'narrowing_audit.summary.mid_narrow_range_pct[1]',
    ],
    'exp37_adaptive_optimism.json': [
        'registry.K_reported_eventlevel', 'keff.K_eff_used',
        'deflation.deflated_lower_bound', 'deflation.deflated_interval[0]',
        'deflation.deflated_interval[1]', 'pbo_cscv.PBO',
    ],
}


def get_path(d, path):
    cur = d
    for part in path.split('.'):
        if part.endswith(')'):   # 特殊标记（如 molchan(removed)）
            return f'__SPECIAL__{part}'
        if '[' in part:
            k, idx = part[:-1].split('[')
            cur = cur[k] if k else cur
            cur = cur[int(idx)]
        else:
            cur = cur[part]
    return cur


def eq(a, b):
    if isinstance(a, float) and isinstance(b, float):
        na, nb = a != a, b != b
        return na and nb if (na or nb) else a == b
    return a == b


NOTE_ASSERT_RE_TRUE = 'bitwise identical=True'
NOTE_ASSERT_RE_FALSE = 'bitwise identical=False'


def annotation_consistency_check():
    """注记布尔自检（R4 扩展, 防 A-M1 类自相矛盾注记被 golden master 永久放行）:
    扫 SPEC 全部 JSON 的字符串值, 凡自嵌 'bitwise identical=X' 布尔的注记, 其与「逐位相同」类
    文字断言必须一致（True↔断言全部逐位相同且无「仅/非」限定; False↔断言含「仅 N 档/非逐位」）。
    必要不充分: 抓自相矛盾注记, 不判注记内容真伪（后者仍靠溯源断言管线）; 已知局限: 会误伤内嵌原样布尔串的更正性/描述性引用——约定更正字段改述为「自嵌布尔值 False」类措辞（R4 起）, 不内嵌 'bitwise identical=X' 原样串。"""
    import re
    viols = []
    def scan(obj, path, fname):
        if isinstance(obj, dict):
            for k, v in obj.items():
                scan(v, f'{path}.{k}', fname)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                scan(v, f'{path}[{i}]', fname)
        elif isinstance(obj, str) and 'bitwise identical=' in obj:
            val_true = NOTE_ASSERT_RE_TRUE in obj
            val_false = NOTE_ASSERT_RE_FALSE in obj
            # 布尔断言唯一性: 同句不得同时含 True 与 False
            if val_true and val_false:
                viols.append(f'{fname}{path}: 同句自嵌 bitwise identical=True 与 =False')
                return
            claims_all = ('逐位相同' in obj) and not re.search(r'仅[^，。；]*档[^，。；]*逐位相同', obj) \
                and ('非' not in obj.split('逐位相同')[0][-6:])
            if val_true and not claims_all:
                viols.append(f'{fname}{path}: bitwise identical=True 但文字断言非「全部逐位相同」')
            if val_false and claims_all:
                viols.append(f'{fname}{path}: 自相矛盾注记——文字断言「逐位相同」与自嵌 bitwise identical=False 同句（A-M1 型）')
    for f in SPEC:
        try:
            d = json.load(open(f'{R}/{f}'))
        except Exception:
            continue
        scan(d, '', f)
    return viols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--freeze', action='store_true')
    args = ap.parse_args()
    if args.freeze:
        gm = {}
        for f, paths in SPEC.items():
            d = json.load(open(f'{R}/{f}'))
            for p in paths:
                try:
                    gm[f'{f}::{p}'] = get_path(d, p)
                except Exception as e:
                    gm[f'{f}::{p}'] = f'__ERROR__ {e}'
        gm['__meta__'] = dict(frozen='2026-10-21 R3 收口冻结轮', note='数值逐位冻结; NaN 双侧同判; '
                              'exp18 molchan/dca 记 removed 标记=键不存在为预期')
        json.dump(gm, open(GM, 'w'), indent=1, ensure_ascii=False, default=float)
        n = len([k for k in gm if not k.startswith('__')])
        print(f'golden master 已冻结: {GM}（{n} 键）')
        return 0
    gm = json.load(open(GM))
    drift = []
    for key, val in gm.items():
        if key.startswith('__'):
            continue
        f, p = key.split('::', 1)
        d = json.load(open(f'{R}/{f}'))
        try:
            cur = get_path(d, p)
        except Exception as e:
            drift.append(f'{key}: 读取失败 {e}')
            continue
        if p.endswith('(removed)'):
            if not (isinstance(cur, str) and cur.startswith('__SPECIAL__')) and (p.split('.')[0] in d):
                drift.append(f'{key}: 预期已删除的键仍存在')
            continue
        if not eq(cur, val):
            drift.append(f'{key}: {val!r} → {cur!r}')
    note_viols = annotation_consistency_check()
    if drift:
        print(f'GOLDEN MASTER 漂移 {len(drift)} 处:')
        for x in drift:
            print(' ', x)
        return 1
    if note_viols:
        print(f'注记布尔自检违例 {len(note_viols)} 处:')
        for x in note_viols:
            print(' ', x)
        return 1
    print(f'golden master 校验通过: {len([k for k in gm if not k.startswith("__")])} 键零漂移 + '
          f'注记布尔自检 {len(SPEC)} 份 JSON 零违例')
    return 0


if __name__ == '__main__':
    sys.exit(main())
