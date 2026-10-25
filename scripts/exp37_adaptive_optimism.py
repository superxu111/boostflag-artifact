#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exp37 测试段自适应乐观量化: 搜索空间审计 + Deflated IGPE + PBO/CSCV（R3 双审 B-M1 + 定向创新方案二）
问题（B-M1）: hybrid 族结构由测试段诊断塑造（exp4b seen/unseen 分层→图核降级→exp12 族设计）,
  其后 +5.874 作为 §11.2 旗舰选择两依据之一在同一 143 事件上报告; exp15/18/20/21 依次累积评估
  且项目已无独立 holdout——+5.874 携带无法界定的自适应乐观成分, 从未申报。
本实验把 B-M1 的「披露句」升级为可审计的量化证据（零模型重跑, 全部对在盘 margin/gains npy）:
  A1 试验登记表: 清点全部在测试段 143 事件上报告过比较的配置/比较（JSON 点值 + 逐事件 margin npy）,
     逐项标注来源/性质。格点面（exp13/15/18/20）与 train/验证段（exp11 P1/P2、exp16 validation、
     exp22）另列不计入事件级 K——评估面不同不可混族。
  A2 K_eff: 24 列在盘 143 维 margin/gains 中心化序列的相关矩阵 PCA（90% 方差主成分数）
     + lag-1 自相关调和界 K/(1+(K−1)ρ̄)（ρ̄ 引 exp14 在案 lag1_acf 0.235–0.336 中值, 不重测）。
  A3 Deflated IGPE（Bailey & López de Prado 2014 JPM 的非参迁移, 迁移声明必报）:
     ①correlated max——24 列联合块 bootstrap（block=10 与 exp14 移动块语义一致, B=20000, 逐列
       中心化=零假设化）每次取 24 列 max → E[max_null] 与 q95;
     ②independent max——从 24 列边缘零假设重抽池独立抽 K_eff 个取 max（相关校正后的上界近似）;
     ③deflation 区间 = [IGPE_flag − q95(max_corr), IGPE_flag − mean(max_indep)]; 保守/乐观两口径
       并报, 解读分支按 P1。
  A4 PBO/CSCV（Bailey et al. model-free, 不需固定 hold-out）: 143 事件按 block≈6 分 23 组,
     随机对称切分 IS/OOS ×2000 → IS 最优比较在 OOS 的排名分位 → PBO=P(排名>中位);
  复现注记（Mi-1, 2026-10-24）: registry 的 pair 字段曾依赖链跑 r4_annotation_fixes.py 修复
  （原 :97 从绝对路径 split('_',2) 取段, R 含下划线时泄漏 'attack_path/results/...' 形态）;
  现改为从 basename 直接产出 r4 目标态——重跑本脚本即得可读形态, 链跑 r4_annotation_fixes.py
  仍可执行且幂等（0 处改动）, 冻结数字（golden master 6 键）不受影响。
     限定声明: 对象=24 对比较族而非互斥模型选择集（配置间存在重叠对）, 分割对称性保留、
     「K=独立试验数」解读不作。
  A5 成文素材: double-dipping 披露句模板（Kriegeskorte et al. 2009 Nat Neurosci 12:535-40 映射）
     +「为何不迁移 DP reusable holdout」判定量（Dwork et al. 2015 Science 349:636-8 最坏情形样本
     复杂度悲观; n=143、K_eff 见 A2 下不可行, 作有出处的不迁移判定）。
  预登记判定 P1（跑前落盘, 不预设何者成立）:
     deflated 下界>0 → 成文=「+5.874 获定量防御: 扣除 K_eff 纯噪声试验期望最大值后仍为正」;
     deflated 下界≤0 → 成文=「+5.874 按预登记降格为带校正量的 prequential 一致性证据,
       不作为无偏前瞻验证引用」——两分支皆可辩护, 均为诚实申报。
  迁移声明（防搬硬套）: ①DSR 解析式针对 Sharpe 正态情形, 本实验用非参块 bootstrap 边缘/联合
     零假设分布替代, 不搬 Sharpe 公式; ②试验表必含预登记外探索性比较（exp16 测试面复核
     mle_omori/mom_ts 两行即其例）; ③CSCV 在单一连续事件流上分割、块边界 6 事件粒度,
     与金融回测 bar 级重叠加权不同——限定声明落盘。
输出: results/exp37_adaptive_optimism.json
"""
import sys, json, time, glob
import numpy as np
sys.path.insert(0, './scripts')
R = './results'

IGPE_FLAG = 5.873700271172389   # exp12 test_IGPE.M_hybrid_W24_c.vs_uniform（golden master 冻结值）
BLOCK = 10                       # 与 exp14 移动块 bootstrap 语义一致
B_JOINT = 20000
N_CSCV = 2000
CSCV_BLOCK = 6


def joint_null_max(Mc, B, block, rng):
    """24 列联合块 bootstrap（同一块索引用于全部列）→ (B, K) 零假设均值 + 逐行 max"""
    n, K = Mc.shape
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(B, nb))
    tau_idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)[:, :n]
    means = Mc[tau_idx].mean(axis=1)                    # (B, K)
    return means, means.max(axis=1)


def main():
    t00 = time.time()
    out = dict(
        exp='Exp37_adaptive_optimism',
        motive=('B-M1: 测试段自适应复用的乐观偏差从未申报——模型族搜索空间被测试段诊断塑造且已无'
                '独立 holdout; 方案二把披露句升级为可审计量化（登记表+K_eff+Deflated IGPE+PBO/CSCV）'),
        preregistered=dict(
            A1='试验登记表: 测试段事件级比较全清点（JSON 点值+逐事件 margin npy）; 格点面/验证段/train 内另列',
            A2='K_eff=PCA(90% 方差) 与 lag-1 调和界两口径（ρ̄ 引 exp14 在案, 不重测）',
            A3='Deflated IGPE 非参迁移: correlated max（联合块 bootstrap 零假设化）+independent max（边缘池）'
               '→ deflation 区间两口径',
            A4='PBO/CSCV: 143 事件 block≈6 对称切分×2000, IS 最优比较的 OOS 排名; 限定=比较族非互斥模型集',
            P1=('deflated 下界>0 → 「+5.874 获定量防御」; 下界≤0 → 「降格为带校正量的 prequential '
                '一致性证据, 不作无偏前瞻验证引用」——跑前预写, 不预设方向')),
        migration_note=('非参迁移声明: ①不搬 DSR/Sharpe 解析式（正态情形专用）, 用块 bootstrap 边缘/联合'
                        '零假设分布替代; ②试验表含预登记外探索性比较; ③CSCV 分割为连续事件流 6 事件'
                        '粒度, 与金融 bar 级加权不同; ④比较族含重叠对, K≠独立模型数'),
        registry={}, audit_timeline=[], keff={}, deflation={}, pbo_cscv={}, writing_assets={})
    json.dump(out, open(f'{R}/exp37_adaptive_optimism.json', 'w'), indent=1, ensure_ascii=False, default=float)
    print('预登记已落盘 exp37_adaptive_optimism.json', flush=True)

    def dump():
        json.dump(out, open(f'{R}/exp37_adaptive_optimism.json', 'w'), indent=1, ensure_ascii=False, default=float)

    # ---------- A1: 试验登记表 ----------
    j11 = json.load(open(f'{R}/exp11_constrained_ablation.json'))
    j12 = json.load(open(f'{R}/exp12_boost_flag.json'))
    j16 = json.load(open(f'{R}/exp16_frailty_upgrade.json'))
    j21 = json.load(open(f'{R}/exp21_mle_deg_testface.json'))

    disk_npy = []
    for f in sorted(glob.glob(f'{R}/exp4_gains_model_vs_*.npy')):
        disk_npy.append(dict(file=f.split('/')[-1], pair=f.split('gains_model_vs_')[1][:-4], source='exp4/exp4b',
                             n=int(np.load(f).shape[0]), kind='gains(事件级)', role='hop1 旗舰 vs 参照×8'))
    for pat, src, note in [('exp12_margin_*.npy', 'exp12', '族内+vs flag/hop1 成对比较（预登记族内九对）'),
                           ('exp16_margin_*.npy', 'exp16', 'frailty 升级后测试面复核（含验证网格胜者测试面评估=探索性）'),
                           ('exp21_margin_*.npy', 'exp21', 'mle_deg 测试面一次性评估（预登记, 测试集只碰一次）')]:
        for f in sorted(glob.glob(f'{R}/{pat}')):
            base = f.split('/')[-1][:-4]                       # expNN_margin_<L>_gt_<R>
            left, *rest = base.split('_gt_')
            pair = ' > '.join([left.split('_', 2)[2]] + rest)  # Mi-1: 从 basename 切, 与 r4_annotation_fixes 目标态一致（原 :97 从绝对路径切段, R 含下划线时泄漏路径形态）
            disk_npy.append(dict(file=f.split('/')[-1], pair=pair,
                                 source=src, n=int(np.load(f).shape[0]), kind='margin(事件级)', role=note))
    n_reported = dict(
        exp4b_hop1_vs_8ref=8,
        exp11_P3_constrained_forward=dict(configs=len(j11['P3_constrained_forward_IGPE']), refs=2,
                                          note='vs_uniform+vs_pop 点值; 部分配置与 exp12 家族重叠'),
        exp12_test_IGPE=dict(configs=len(j12['test_IGPE']), refs=3, note='vs_uniform/vs_pop/vs_hod_pop ×10 配置'),
        exp16_test_IGPE=dict(configs=len(j16['test_IGPE']), refs=1,
                             note='vs_uniform ×8（含 hybrid_mle_omori/hybrid_mom_ts 验证网格测试面复核=探索性）'),
        exp21_testface=1,
        grid_face_not_counted=('exp13/15/18/20 格点面（τ×格 IGN/PR-AUC）与 exp16 validation/exp22 train 内'
                               '比较不计入事件级 K——评估面不同不可混族, 另列备考'))
    out['registry'] = dict(n_margin_gains_npy=len(disk_npy), npy=disk_npy, reported_point_values=n_reported,
                           K_npy=len(disk_npy),
                           K_reported_eventlevel=(8 + 2 * len(j11['P3_constrained_forward_IGPE'])
                                                  + 3 * len(j12['test_IGPE']) + len(j16['test_IGPE']) + 1))
    print(f"A1 登记: 在盘 margin/gains npy={len(disk_npy)}, 事件级报告点值 K={out['registry']['K_reported_eventlevel']}", flush=True)

    # 塑造时间线（B-M1 审计核心, 从已归档 JSON/LOG 事实）
    out['audit_timeline'] = [
        dict(step=1, item='exp4b seen/unseen 分层诊断', evidence='results/exp4b_rescored.json',
             role='测试段诊断塑造后续族设计（double-dipping 链起点）'),
        dict(step=2, item='图核降级裁决（hop1 降为格点面参考实现）', evidence='RESEARCH_LOG R1 B-C1 段+exp13',
             role='同一测试段证据决定 hybrid/hop1 分工'),
        dict(step=3, item='exp12 九配置族内比较+旗舰 vs_uniform +5.8737', evidence='results/exp12_boost_flag.json',
             role='搜索空间由 1-2 步塑造后, 在同一 143 事件上报告旗舰数字'),
        dict(step=4, item='exp15/16/18/20/21 依次累积评估', evidence='results/exp15..exp21*.json',
             role='同一测试段重复评估（exp21 一臂除外——其「测试集只碰一次」只约束 mle_deg 臂）'),
        dict(step=5, item='结论: 参数级 prequential 诚实, 但模型族搜索空间被测试诊断塑造',
             evidence='本表', role='成文披露句骨架（A5 模板）')]

    # ---------- A2: K_eff ----------
    Ms, names = [], []
    for d in disk_npy:
        v = np.load(f"{R}/{d['file']}").astype(float)
        Ms.append(v); names.append(d['file'])
    M = np.column_stack(Ms)                      # (143, 24)
    assert M.shape[0] == 143
    Mc = M - M.mean(axis=0, keepdims=True)       # 逐列中心化=零假设化
    C = np.corrcoef(Mc.T)
    eig = np.sort(np.linalg.eigvalsh(C))[::-1]
    frac = np.cumsum(eig) / eig.sum()
    k_pca90 = int(np.searchsorted(frac, 0.90) + 1)
    rho_bar = float(np.mean([0.2353840314598317, 0.33577273928932505]))   # exp14 在案区间中值
    k_harm = float(24 / (1 + 23 * rho_bar))
    out['keff'] = dict(K_npy=24, pca90_components=k_pca90, pca_cumvar90=float(frac[k_pca90 - 1]),
                       harmonization_bound=k_harm, rho_bar=rho_bar, rho_bar_source='exp14_inference_stab.json lag1_acf 0.235–0.336',
                       K_eff_used=k_pca90,
                       note='K_eff 取 PCA 90% 方差口径（保守）; 调和界只作敏感性参考')
    print(f"A2: K_eff(PCA90)={k_pca90}, 调和界={k_harm:.1f}", flush=True)

    # ---------- A3: Deflated IGPE ----------
    rng = np.random.default_rng(37001)
    means_joint, max_corr = joint_null_max(Mc, B_JOINT, BLOCK, rng)
    # independent max: 边缘零假设池（每列独立重抽 B/K_eff 轮, 池内独立取 max）
    pool = []
    n_pool = max(int(B_JOINT * 24 / max(k_pca90, 1) / 100) * 100, 5000)
    for k in range(24):
        col = Mc[:, k]
        nb = int(np.ceil(143 / BLOCK))
        starts = rng.integers(0, 143 - BLOCK + 1, size=(n_pool, nb))
        tidx = (starts[:, :, None] + np.arange(BLOCK)[None, None, :]).reshape(n_pool, -1)[:, :143]
        pool.append(col[tidx].mean(axis=1))
    pool = np.column_stack(pool)                  # (n_pool, 24) 边缘零假设池
    rng2 = np.random.default_rng(37002)
    idx = rng2.integers(0, n_pool, size=(B_JOINT, k_pca90))
    colpick = rng2.integers(0, 24, size=(B_JOINT, k_pca90))
    max_indep = pool[idx, colpick].max(axis=1)   # 行=池样本 idx, 列=随机抽列
    e_corr, q95_corr = float(max_corr.mean()), float(np.percentile(max_corr, 95))
    e_indep = float(max_indep.mean())
    lo = IGPE_FLAG - q95_corr
    hi = IGPE_FLAG - e_indep
    out['deflation'] = dict(
        IGPE_flag=IGPE_FLAG, block=BLOCK, B_joint=B_JOINT,
        max_null_correlated=dict(mean=e_corr, q95=q95_corr),
        max_null_independent=dict(mean=e_indep, q95=float(np.percentile(max_indep, 95))),
        deflated_interval=[float(lo), float(hi)],
        deflated_point_conservative=float(IGPE_FLAG - e_corr),
        P1_branch=('deflated_下界>0: 定量防御' if lo > 0 else 'deflated_下界≤0: 降格为带校正量 prequential 一致性证据'),
        deflated_lower_bound=float(lo),
        note='区间=[IGPE_flag−q95(max_corr), IGPE_flag−mean(max_indep)]: 左端保守（相关 max 偏大→扣得多）, '
             '右端乐观（K_eff 独立抽取）; 两口径并报不择一')
    print(f"A3: max_null corr mean={e_corr:.3f} q95={q95_corr:.3f} | indep mean={e_indep:.3f} | "
          f"deflated 区间=[{lo:.3f},{hi:.3f}] → {out['deflation']['P1_branch']}", flush=True)

    # ---------- A4: PBO/CSCV ----------
    n = 143
    n_blocks = n // CSCV_BLOCK                    # 23 组（余 5 事件不进切分, 声明）
    rng3 = np.random.default_rng(37003)
    ranks, oos_is_corr = [], []
    for _ in range(N_CSCV):
        perm = rng3.permutation(n_blocks)
        is_b = perm[:n_blocks // 2]
        oos_b = perm[n_blocks // 2:]
        is_idx = np.concatenate([np.arange(b * CSCV_BLOCK, (b + 1) * CSCV_BLOCK) for b in is_b])
        oos_idx = np.concatenate([np.arange(b * CSCV_BLOCK, min((b + 1) * CSCV_BLOCK, n)) for b in oos_b])
        is_mean = M[is_idx].mean(axis=0)
        oos_mean = M[oos_idx].mean(axis=0)
        k_star = int(np.argmax(is_mean))
        rank_oos = int((oos_mean > oos_mean[k_star]).sum())      # IS 最优在 OOS 的排名（0=最优）
        ranks.append(rank_oos / 24.0)
        oos_is_corr.append(oos_mean[k_star])
    ranks = np.array(ranks)
    from scipy.stats import spearmanr
    is_all = M[np.concatenate([np.arange(b * CSCV_BLOCK, (b + 1) * CSCV_BLOCK) for b in range(n_blocks // 2)])].mean(axis=0)
    pbo = float((ranks > 0.5).mean())
    out['pbo_cscv'] = dict(
        n_configurations=24, cscv_block=CSCV_BLOCK, n_blocks=n_blocks, n_trials=N_CSCV,
        PBO=pbo,
        rank_quantiles=dict(q50=float(np.percentile(ranks, 50)), q90=float(np.percentile(ranks, 90)),
                            q95=float(np.percentile(ranks, 95))),
        tail_note='余 5 事件不进切分（143=23×6+5）',
        interpretation=('IS 最优比较在 OOS 落入下半区的概率 PBO=%.3f——若高则家族内排名不稳定, '
                        '与 deflation 区间并读' % pbo),
        limitation='对象=24 对比较族（含重叠对）, 非互斥模型选择集; CSCV 结论限定于「比较族排名稳定性」')
    print(f"A4: PBO={pbo:.3f}（IS 最优 OOS 排名分位 q50={np.percentile(ranks, 50):.2f}）", flush=True)

    # ---------- A5: 成文素材 ----------
    out['writing_assets'] = dict(
        disclosure_template=(
            'hybrid 族的搜索空间由测试段诊断迭代塑造（exp4b seen/unseen 分层→图核降级裁决→exp12 九配置'
            '族内比较）, +5.874 bits 为 prequential 一致性证据而非无偏前瞻验证; 事件级测试段累计报告'
            ' K=%d 项比较（在盘逐事件序列 %d 条）, 有效试验数 K_eff(PCA90)=%d; deflation 分析给出'
            '校正区间 [%0.2f, %0.2f] bits——按预登记分支「%s」。'
            % (out['registry']['K_reported_eventlevel'], 24, k_pca90, lo, hi,
               '定量防御' if lo > 0 else '降格申报')),
        double_dipping_map=('Kriegeskorte et al. 2009 Nat Neurosci 12:535-40: 同一数据先选择后选择性分析→'
                            '统计量与选择准则非独立。映射: 测试段诊断塑造模型族（选择）→同段报 +5.87（选择性'
                            '分析）; 处方=非选择性分析（不可得, 无 holdout）/零假设下独立性（不成立）/披露+'
                            '校正（本实验 A3/A4）——取第三处方'),
        dp_not_migrated=dict(
            refs='Dwork et al. 2015 Science 349:636-8 reusable holdout/Thresholdout; '
                 '保证至多约 n^2 个过拟合函数内有效, 最坏情形样本复杂度悲观需极大样本',
            ruling='n=143 事件、K_eff=%d 下 DP 路不可落地（143^2≈2e4 个函数的预算在本项目轮次结构下'
                   '不适用且实现不可行）→ 作有出处的不迁移判定, DP 仅入 Limitation 讨论' % k_pca90),
        prequential_restatement='参数级 prequential 诚实（无参数级 peeking, exp14 稳定性在案）与族级'
                                '自适应乐观（本实验量化）两层须分开表述, 禁止混写')
    out['runtime_min'] = (time.time() - t00) / 60
    dump()
    print(f'落盘 {R}/exp37_adaptive_optimism.json; 总耗时 {out["runtime_min"]:.1f} min', flush=True)
    open(f'{R}/exp37.done', 'w').write('done: exp37 搜索空间审计+Deflated IGPE+PBO/CSCV（B-M1 量化收口）\n')


if __name__ == '__main__':
    main()
