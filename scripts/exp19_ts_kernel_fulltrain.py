#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exp19 ts_full_train_supplement 溯源重算 + 权威 JSON 脚本化勘误标注（审稿A-M4/M2/m3/m4 + A-m1/m2 修复）
动机（审稿A-M4, 本席独有发现）: exp16_frailty_upgrade.json 的 ts_full_train_supplement 块系运行后
  手工补丁——exp16_full.log 与 exp16.done 止于 10:55:37, JSON mtime=10:58:03; 字段名与数字
  （ll −6996.73/−7007.91/−7019.54, w_short 0.9995–0.9997）不在任何归档脚本（grep 零匹配）与运行日志中,
  补丁代码未归档——「权威 JSON=脚本产物、逐位可复现」溯源链破口。
修复:
  F1 以归档代码全 train 重拟合重算: omori 参考拟合须逐位复现 exp16 hybrid_fr（ll=−6960.356626…,
     seed=23/n_restarts=6 同参——管线自证）; twostage×3 档（τ1=300s 固定, τ2∈{24h,72h,7d}, W=τ2）
     约束拟合（a≤0.95）→ ll/BIC(k=5)/branch_a/w_short。新权威=本实验 JSON; 原补丁块标注「未溯源、由
     exp19 取代」。
  F2 脚本化勘误标注（R2_correction_20261021 字典合并, 遵循 exp11/12/14 既有模式; 断言除新增键外
     原键值零改动, 全程留痕日志）: exp16（G1 预登记偏离+依据句更正+溯源修复+unconstrained 标签语义+
     平手阈值补记+网格有效自由度）、exp15（skill_ign ×H 作废声明+事故补录+族标签+BLc 族警示）、
     exp13（SS_vs_pop 根因更正）、exp5（note 补 δ=0/δ=1 分解指针, 二轮 A-m5）。
预登记判定（先落盘后运行）:
  H1: twostage 三档 ll 全部劣于 omori（复现补丁方向）→ 双时间尺度核被数据拒绝结论在脚本留痕下成立;
  H2: w_short≈1（第二分量质量→0）→ F2/F3「长尾由 flag 静态项承担」定量回答在归档代码下成立。
  若与补丁数字超容差不符（>0.05 ll）→ 以本脚本（可复现）为准并如实申报差异。
"""
import sys, json, time, argparse
import numpy as np
sys.path.insert(0, './scripts')
sys.path.insert(0, './src')
from exp_patch_common import load_all, eb_flag_q, fit_constrained, R
from lanl_pp import GraphHawkes

TAU1 = 300.0
TAU2_GRID = {'ts24h': 86400.0, 'ts72h': 259200.0, 'ts7d': 604800.0}
PATCH_REF = dict(ts24h=dict(train_ll=-6996.73, BIC=14025.19, branch_a=0.3359, w_short=0.9995),
                 ts72h=dict(train_ll=-7007.91, BIC=14047.56, branch_a=0.3437, w_short=0.9997),
                 ts7d=dict(train_ll=-7019.54, BIC=14070.8, branch_a=0.3868, w_short=0.9997))


def fits_full_train(smoke=False):
    D = load_all()
    adj, nodes, ev = D['adj'], D['nodes'], D['ev']
    t_all, dst_all, src_all = D['t_all'], D['dst_all'], D['src_all']
    cut, t_cut, V_all = D['cut'], D['t_cut'], D['V_all']
    cntC = np.zeros(len(nodes))
    cidxC = {c: i for i, c in enumerate(nodes)}
    for d in dst_all[:cut]:
        cntC[cidxC[d]] += 1
    q_mom, alpha_mom, _ = eb_flag_q(cntC)   # exp16 同款全 train 矩法 flag（α=0.009497）

    def build(kn, W, ts):
        return GraphHawkes(t_all[:cut], list(dst_all[:cut]), list(src_all[:cut]), adj, nodes,
                           dst_extra=V_all, kernel=kn, gk='hop0', bg='hod24', W=W,
                           qbg='custom', q_custom=q_mom, tau_stage=ts)

    out = dict(exp='Exp19_ts_kernel_fulltrain_provenance',
               motive='exp16 ts_full_train_supplement 运行后手工补丁无溯源（审稿A-M4）——以归档代码重算',
               preregistered=dict(
                   H1='twostage 三档 ll 全部劣于 omori → 双时间尺度核被数据拒绝（脚本留痕版）',
                   H2='w_short≈1 → 长尾由 flag 静态项承担（F2/F3）',
                   tolerance='与补丁数字 |Δll|≤0.05 视为复现; 超容差以本脚本为准并申报差异'),
               omori_reference={},
               twostage={})
    grid = [('ts24h', 86400.0)] if smoke else list(TAU2_GRID.items())

    m0 = build('omori', 86400.0, None)
    th0, ll0, a0 = fit_constrained(m0, a_max=0.95, n_restarts=6, seed=23, t_end=t_cut)
    out['omori_reference'] = dict(ll=float(ll0), branch_a=float(a0), W=86400.0, k=5,
                                  BIC=round(-2 * ll0 + 5 * np.log(cut), 2),
                                  exp16_hybrid_fr_ll=-6960.356626404788,
                                  pipeline_validated=bool(abs(ll0 - (-6960.356626404788)) < 1e-6),
                                  note='omori+mom 全 train 拟合须逐位复现 exp16 hybrid_fr（同 seed/重启数）——'
                                       '管线自证后 twostage 结果才可信')
    print(f"  omori 参考: ll={ll0:.4f} a={a0:.4f} (exp16 hybrid_fr ll=-6960.3566) "
          f"管线自证={'PASS' if out['omori_reference']['pipeline_validated'] else 'FAIL'}", flush=True)
    json.dump(out, open(f'{R}/exp19_ts_kernel_fulltrain.json', 'w'), indent=1, ensure_ascii=False, default=float)

    for kn, tau2 in grid:
        m = build('twostage', tau2, (TAU1, tau2))
        th, ll, a = fit_constrained(m, a_max=0.95, n_restarts=6, seed=23, t_end=t_cut)
        w = float(1.0 / (1.0 + np.exp(-th[m.idx['logit_w1']])))
        rec = dict(ll=float(ll), BIC=round(-2 * ll + 5 * np.log(cut), 2), branch_a=float(a),
                   w_short=w, mass_long=float(1 - w), k=5, W=tau2, tau_stage=[TAU1, tau2],
                   patch_ref=PATCH_REF[kn],
                   ll_vs_patch=float(ll - PATCH_REF[kn]['train_ll']))
        out['twostage'][kn] = rec
        json.dump(out, open(f'{R}/exp19_ts_kernel_fulltrain.json', 'w'), indent=1,
                  ensure_ascii=False, default=float)
        print(f"  {kn}: ll={ll:.2f} BIC={rec['BIC']} a={a:.4f} w_short={w:.4f} "
              f"(补丁 ll={PATCH_REF[kn]['train_ll']}, Δll={rec['ll_vs_patch']:+.4f})", flush=True)
    out['verdict'] = dict(
        H1=bool(all(out['twostage'][k]['ll'] < out['omori_reference']['ll'] for k in out['twostage'])),
        H2=bool(all(out['twostage'][k]['w_short'] > 0.99 for k in out['twostage'])),
        reproduced_within_tolerance=bool(all(abs(out['twostage'][k]['ll_vs_patch']) <= 0.05
                                             for k in out['twostage'])),
        note='H1/H2 成立→双时间尺度核被数据拒绝在脚本留痕下成立; '
             'reproduced_within_tolerance=False 时以本脚本为准并申报与补丁差异')
    return out


def merge_correction(json_path, additions, note_keys_written):
    """R2_correction_20261021 字典合并; 断言除该键外原键值零改动"""
    with open(json_path) as f:
        before = json.load(f)
    after = json.loads(json.dumps(before))   # deep copy
    rc = after.setdefault('R2_correction_20261021', {})
    dup = [k for k in additions if k in rc]
    rc.update(additions)
    rc['_written_by'] = 'scripts/exp19_ts_kernel_fulltrain.py (R2 补实验与拔高人, 2026-10-21)'
    json.dump(after, open(json_path, 'w'), indent=1, ensure_ascii=False, default=float)
    # 零改动断言: 除 R2_correction_20261021 外全部原键逐一相等
    with open(json_path) as f:
        chk = json.load(f)
    for k, v in before.items():
        if k == 'R2_correction_20261021':
            continue
        assert json.dumps(chk[k], sort_keys=True, default=float) == json.dumps(v, sort_keys=True, default=float), \
            f'{json_path}: 原键 {k} 被意外改动!'
    note_keys_written.append((json_path.split('/')[-1], list(additions.keys()), dup))
    print(f'  标注 {json_path.split("/")[-1]}: +{list(additions.keys())}' + (f' (覆盖已有键: {dup})' if dup else ''),
          flush=True)


def apply_annotations():
    """F2: 四个权威 JSON 的脚本化勘误标注（事故必录/溯源修复/作废声明）"""
    written = []
    merge_correction(f'{R}/exp16_frailty_upgrade.json', dict(
        G1_prereg_deviation=(
            '预登记三分支字面判定: 选定配置边际 hybrid_fr−M_hybrid_W24_c CI=[-0.000279,-0.000153] '
            '上界<0 → 应读「有害」分支; 脚本 exp16_frailty_upgrade.py:403-406 增设了预登记外 '
            'same_as_current & |ΔIGPE|<0.01 首分支判「无实质变化」（字面偏离）。实质动作两分支一致'
            '（均回退 hybrid）; 幅度 −0.0002 bits 系同一模型两次拟合之差的常数级小差（正0/负143、'
            'p_wilcoxon=2.6e-25、p_boot=0——单独引用会被误读为 frailty 显著有害）。处置: 等价阈值 '
            '|Δ|<0.01 bits 作为预登记修订显式落盘于本字段与 RESEARCH_LOG §10（2026-10-21 R2 补实验轮）; '
            'G1 实质结论=同构回退不变'),
        G1_basis_sentence_correction=(
            'RESEARCH_LOG:228 曾引未选定配置 hybrid_mle_omori 边际（+0.0346 CI[-0.023,+0.086] p=0.26）'
            '作 G1 依据——更正为选定配置边际（-0.000219 CI[-0.000279,-0.000153], 见 margins.'
            'hybrid_fr_gt_M_hybrid_W24_c）。「EB 升级无显著增量」限定为等暴露 MLE 版（验证集口径）; '
            '暴露加权 mle_deg 版未上测试段（验证集 gap=0.0422<0.05 bits 被简约规则压过, ≪MDE 1.10; '
            '0.05 阈值非功效推导——验证段 n=242 的 MDE 量级≈0.8 bits, 阈值取 0 则 mle_deg 入选——'
            '选模结局对该阈值敏感, 成文须披露）'),
        ts_full_train_supplement_provenance=(
            '本 JSON 的 ts_full_train_supplement 块系运行后手工补丁（exp16_full.log 与 exp16.done 止于 '
            '10:55:37, JSON mtime 10:58:03; 字段名与数字不在任何归档脚本/日志中, 补丁代码未归档）——'
            '溯源链破口。已由 scripts/exp19_ts_kernel_fulltrain.py 以归档代码全 train 重拟合重算'
            '（权威=results/exp19_ts_kernel_fulltrain.json）; 本块数字标注「未溯源、由 exp19 取代」, '
            '科学结论（双时间尺度核被数据拒绝）以 exp19 复算为准'),
        hybrid_fr_unconstrained_label=(
            '该块两端均含 frailty flag（同一 q_sel）——只能证「a≤0.95 约束未 binding」（a=0.375 远离边界, '
            'μ0/κ 与约束版逐位近同）; 测不了原 bias_note 所称「遗漏随机效应→低估背景高估激发」的偏倚方向'
            '（需无 frailty 同构对照, 未做, 列入下轮清单）。bias_note 表述作废'),
        prereg_validation_tie_threshold=(
            'preregistered.validation 原文只写「平手取简约」——0.05 bits 阈值原只在脚本头 docstring 与 '
            'selected.tie_rule; 补记: 验证判据=验证段逐事件 IGPE vs uniform 均值, 平手阈值 |Δ|≤0.05 bits '
            '取更简约（核 omori 优先、flag mom 优先）'),
        validation_grid_effective_dof=(
            '验证网格 ts24h/ts72h/ts7d 三档数字逐位相同（w_short≈0.9995–0.9997, 第二分量质量<0.05%）'
            '——12 配置有效自由度≈6, 选模网格按 6 独立档申报'),
    ), written)
    merge_correction(f'{R}/exp15_hybrid_causal.json', dict(
        skill_ign_invalidated_xH=(
            '各 skill_ign 块的气候学参考 S_pop（exp15_hybrid_causal.py:113 R0=每秒率 × :232 qpopU）'
            '未乘窗长 H, 而模型侧为 [τ,τ+H) 窗内积分（:72-73 背景 + :83-89 触发项）——两侧同入 '
            'P=1−exp(−S) 泊松转换量纲不可比（1h 差 3600×、6h 差 21600×）。ISS_positive_taus_mean '
            '「健康口径 0.33–0.56」作废; ISS_all_taus_mean_pathology 注记的根因诊断（「无正例 τ 分母≈0」）'
            '系误诊——真因=缺 ×H（正确尺度下空 τ 的 IGN_clim 有限）。修正后权威=results/'
            'exp18_skill_scale_protocol.json: ISS_positive 反号（targets −0.1270/−0.0905/−0.0082）、'
            'ISS_all≈+0.450/+0.427/+0.406（模型平均 IGN 优于正确尺度气候学 ~40–45%, 正例 τ 略差, '
            '与 N-test 过预测自洽）'),
        incident_log_backfill=(
            '事故补录（「事故必录」纪律缺口, 二连失: 上轮 B-m3 同型）: ①首跑 logs/exp15_full.log 10:48 '
            '因 multiplicity 族键 KeyError targets_1h 崩溃（:299 硬编码 key）, 脚本 10:56 修复, 11:16 '
            '重跑成功（exp15_full2.log）, 两版数字同 seed 确定性一致; ②本 JSON mtime 11:21:30 晚于 '
            'exp15.done 11:16:23——ISS_all_taus_mean_pathology 字段为运行后注记补丁（无脚本留痕）, '
            '其 ×H 误诊根因已由 exp18 修正口径取代'),
        multiplicity_cell_label=(
            'multiplicity.tests 的 cell 标签两宇宙同名 "24h_1h" 等（首跑 KeyError 修复硬编码所致）无法区分'
            '宇宙——按 tests 数组顺序=targets 1h/3h/6h、targets_train 1h/3h/6h; 扩族引用见 exp18.'
            'expanded_multiplicity（探索性全族 24 检验, 事后定族只作引用警示）'),
        BLc_family_warning=(
            'model−BLc 的 1h 两格（targets p=0.0020 / targets_train p=0.0060）在预登记 6 检验族外; '
            '并入探索性全族 24 检验后 targets 1h ×24=0.048 边缘、targets_train 1h 不存活——BLc 显著性'
            '不得作为独立结论引用（exp18 K6 落盘）'),
    ), written)
    merge_correction(f'{R}/exp13_causal_forecast.json', dict(
        SS_vs_pop_rootcause=(
            'skill_curve.SS_vs_pop_mean（targets −1864.7/−4490.0/−7516.9; targets_train −1982.0/−4800.1/'
            '−8179.9）爆炸根因更正: 此前「无正例 τ 分母≈0」归因系误诊——真因=pop 气候学参考为秒级率未乘'
            '窗长 H（与 exp15 skill_ign 同病, 1h 差 3600×）。正确尺度下空 τ 的 IGN_clim 有限, 比值不爆炸。'
            'SS_vs_pop 系列作废; 修正后 hop1 版 ISS/IGN=exp18_skill_scale_protocol.json 的 '
            'hop1_skill_scaled 块。SS_vs_persistence 不受影响（两侧均计数窗口径）'),
    ), written)
    # exp5: note 追加 δ 分解指针（二轮 A-m5, 原 note 文本保留）
    p5 = f'{R}/exp5_knox_grid.json'
    with open(p5) as f:
        j5 = json.load(f)
    old_note = j5.get('note', '')
    add = ('；δ 分解指针（二轮 A-m5 补）: δ≤1 主格 obs=451 含同主机对（δ=0）——排除同主机后 δ=1 恰好 '
           'obs=103 vs H0=158.3、全部 τ 格 p≥0.88 无信号（exp12 正式化, results/exp12_boost_flag.json '
           'Knox 段）——Knox 聚集信号实为同主机重复受害')
    if 'δ 分解指针' not in old_note:
        j5['note'] = old_note + add
        json.dump(j5, open(p5, 'w'), indent=1, ensure_ascii=False, default=float)
        written.append(('exp5_knox_grid.json', ['note 追加 δ 分解指针'], []))
        print('  标注 exp5_knox_grid.json: note 追加 δ=0/δ=1 分解指针', flush=True)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--annotations-only', action='store_true')
    args = ap.parse_args()
    t00 = time.time()
    if not args.annotations_only:
        out = fits_full_train(args.smoke)
        json.dump(out, open(f'{R}/exp19_ts_kernel_fulltrain.json', 'w'), indent=1,
                  ensure_ascii=False, default=float)
        print(f'落盘 {R}/exp19_ts_kernel_fulltrain.json', flush=True)
    print('== F2 脚本化勘误标注（断言原键值零改动） ==', flush=True)
    written = apply_annotations()
    print(f'完成 {len(written)} 个文件的标注; 总耗时 {(time.time()-t00)/60:.1f} min', flush=True)
    open(f'{R}/exp19.done', 'w').write(
        'done: exp19 ts 核全 train 溯源重算 + exp13/15/16/5 JSON 脚本化勘误标注, '
        '见 exp19_ts_kernel_fulltrain.json\n')


if __name__ == '__main__':
    main()
