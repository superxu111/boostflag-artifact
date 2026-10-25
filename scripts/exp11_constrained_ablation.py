#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exp11 消融修复：分支比约束重拟合 + 主模型选择论证（审稿A-C1 / 审稿B-M1 / 方案二a）
背景（双审 C1 同根因）:
  - exp_summary §2 将 exp_hop 的 AIC 13975.8 误记为主模型 hop1（实测 hop1 AIC=15454.99, 排第4, 与最优差≈1488）
  - AIC 前排 decay/exp_hop 的无约束 MLE 贴临界 a=κ·G_W(24h)≈0.996（MLE 退化换脊: κ↑+μ0→0）
  - 主模型选择无论证; B4「无图核」消融（hop0-only）缺席
本脚本（预登记于运行前）:
  P1 由已存参数复算全部无约束配置的 a=κ·G_W(24h) → 退化诊断表（不重跑, 复用已核验 exp34_main.json）
  P2 全网格 + 新增 hop0 行（B4 无图核消融）在平稳性约束 a≤0.95 下重拟合 → 约束/无约束 AIC 双表
  P3 主模型选择论证: 预登记规则=「约束 AIC 排序 + 前瞻 IGPE + 参数简约共同呈现; hop1 为先验设定的机制基线
     （4参数、跨跳常数分支比）而非 AIC 最优」——诚实呈现, 不再声称 AIC 最优/紧随
  P4 C1_targets train-only 目标集敏感性（审稿A-m4: 原 301 宇宙含测试信息）
  a≤0.95 是平稳性先验选择而非数据推断（SPOCK 非可辨识警告）——如实声明
"""
import sys, json, time
import numpy as np
sys.path.insert(0, './scripts')
sys.path.insert(0, './src')
from exp_patch_common import (load_all, fit_constrained, branch_ratio_unconstrained,
                              score_Ktilde, eb_flag_q, R)
from lanl_pp import GraphHawkes, hour_profile, igpe

A_MAX = 0.95

def main():
    t00 = time.time()
    D = load_all()
    adj, nodes, ev = D['adj'], D['nodes'], D['ev']
    t_all, dst_all, src_all = D['t_all'], D['dst_all'], D['src_all']
    cut, t_cut, V_all = D['cut'], D['t_cut'], D['V_all']
    out = dict(exp='Exp11_constrained_ablation', a_max=A_MAX,
               note='a≤0.95 为平稳性先验约束(非数据推断); 无约束数字复用已核验 exp34_main.json')

    # ---------- P1: 已存无约束配置的分支比/退化诊断 ----------
    print('== P1 无约束 a=κ·G_W(24h) 复算 ==', flush=True)
    e34 = json.load(open(f'{R}/exp34_main.json'))
    diag = []
    for row in e34['exp3']['ablation']:
        if 'error' in row:
            continue
        cfg = row['cfg']
        a = branch_ratio_unconstrained(row['params'], kernel=cfg['kernel'], W=float(cfg['W']))
        deg = (a > 0.95) or (row['params'].get('p', 0) >= 2.0 - 1e-6) or (row['params'].get('mu0', 1) < 1e-5)
        diag.append(dict(cfg={k: str(v) for k, v in cfg.items()}, n_par=row['n_par'],
                         train_ll=row['train_ll'], AIC=row['AIC'], a_kappaGW=round(a, 4),
                         degenerate_flag=bool(deg)))
        print(f"  {cfg.get('kernel'):6s}/{cfg.get('gk',''):8s}/{cfg.get('bg',''):9s}/{cfg.get('qbg',''):8s}/W={cfg.get('W',''):8s} "
              f"AIC={row['AIC']:9.2f} a={a:.4f} degenerate={deg}", flush=True)
    out['P1_unconstrained_diagnostics'] = diag

    # ---------- P2: 约束重拟合全网格 + hop0 行 ----------
    print('== P2 约束(a≤0.95)重拟合 ==', flush=True)
    grid = []
    for kern in ['omori', 'exp']:
        for gk in ['hop0', 'hop1', 'hop2', 'decay', 'exp_hop']:
            if kern == 'exp' and gk == 'hop0':
                continue  # exp 核 AIC 已远离前排, hop0 语义仅在 omori 下有意义
            grid.append(dict(kernel=kern, gk=gk, bg='hod24', qbg='pop', W=86400.0))
    for bg in ['const', 'fourier2']:
        grid.append(dict(kernel='omori', gk='hop1', bg=bg, qbg='pop', W=86400.0))
    for qbg in ['uniform', 'degree']:
        grid.append(dict(kernel='omori', gk='hop1', bg='hod24', qbg=qbg, W=86400.0))
    for W in [600.0, 3600.0]:
        grid.append(dict(kernel='omori', gk='hop1', bg='hod24', qbg='pop', W=W))

    def build(cfg):
        return GraphHawkes(t_all[:cut], list(dst_all[:cut]), list(src_all[:cut]), adj, nodes,
                           dst_extra=V_all, kernel=cfg['kernel'], gk=cfg['gk'], bg=cfg['bg'],
                           W=cfg['W'], qbg=cfg['qbg'], cand='all')

    cntC = None  # 评分用上下文（主宇宙）
    m_ref = None
    cons = []
    for cfg in grid:
        key = f"{cfg['kernel']}/{cfg['gk']}/{cfg['bg']}/{cfg['qbg']}/W={cfg['W']}"
        try:
            m = build(cfg)
            th, ll, a = fit_constrained(m, a_max=A_MAX, n_restarts=6, seed=11)
            pd_ = m.params_dict()
            row = dict(cfg={k: str(v) for k, v in cfg.items()}, n_par=m.n_par,
                       train_ll_constrained=round(ll, 2), AIC_constrained=round(2 * m.n_par - 2 * ll, 2),
                       a_constrained=round(a, 4), params=pd_)
            # 无约束对照（复用已存）
            match = [d for d in diag if d['cfg'] == row['cfg']]
            if match:
                row['train_ll_unconstrained'] = match[0]['train_ll']
                row['AIC_unconstrained'] = match[0]['AIC']
                row['a_unconstrained'] = match[0]['a_kappaGW']
            cons.append(row)
            print(f"  [C] {key} k={m.n_par} ll={ll:.1f} AIC={row['AIC_constrained']:.1f} a={a:.4f}", flush=True)
        except Exception as e:
            cons.append(dict(cfg={k: str(v) for k, v in cfg.items()}, error=str(e)[:120]))
            print(f"  [C] {key} ERROR {e}", flush=True)
    okc = [r for r in cons if 'error' not in r]
    bestc = min(okc, key=lambda r: r['AIC_constrained'])
    out['P2_constrained_table'] = cons
    out['P2_best_constrained'] = bestc['cfg']
    print(f"  约束 AIC 最优: {bestc['cfg']} (AIC={bestc['AIC_constrained']})", flush=True)

    # ---------- P3: 候选主模型约束下前瞻 IGPE（uniform/pop 双锚） ----------
    print('== P3 约束下前瞻 IGPE ==', flush=True)
    m_h = hour_profile(t_all[:cut])
    R0 = cut / (t_cut - t_all[0])
    si_h0_full = None
    fwd = []
    for cfgkey in ['omori/hop0/hod24/pop/W=86400.0', 'omori/hop1/hod24/pop/W=86400.0',
                   'omori/hop2/hod24/pop/W=86400.0', 'omori/decay/hod24/pop/W=86400.0',
                   'omori/exp_hop/hod24/pop/W=86400.0']:
        row = next((r for r in okc if r['cfg']['kernel'] + '/' + r['cfg']['gk'] + '/' +
                    r['cfg']['bg'] + '/' + r['cfg']['qbg'] + '/W=' + r['cfg']['W'] == cfgkey), None)
        if row is None:
            continue
        cfg = dict(kernel=row['cfg']['kernel'], gk=row['cfg']['gk'], bg=row['cfg']['bg'],
                   qbg=row['cfg']['qbg'], W=float(row['cfg']['W']))
        m = build(cfg)
        # 重拟合一次以持有 th（cons 循环中的 m 已释放）
        th, ll, a = fit_constrained(m, a_max=A_MAX, n_restarts=6, seed=11)
        cntC = np.zeros(m.nC)
        for d in dst_all[:cut]:
            cntC[m.cidx[d]] += 1
        popC = (cntC + 0.5) / (cntC + 0.5).sum()
        ctx = dict(pop=popC, m_h=m_h, R0=R0, flag_q=None, theta_iw=0.0, ECNT={'cnt': {}, 'tot': {}})
        si_h0 = [m.vidx[d] for d in dst_all[:cut]]
        la, lv, ra, rv = score_Ktilde(m, th, ['uniform', 'pop'], t_all[cut:], list(dst_all[cut:]),
                                      list(src_all[cut:]), list(t_all[:cut]), list(dst_all[:cut]), si_h0, ctx)
        gu = np.log2(la / ra['uniform']); gp = np.log2(la / ra['pop'])
        st_u, st_p = igpe(gu), igpe(gp)
        fwd.append(dict(cfg=cfgkey, AIC_constrained=row['AIC_constrained'], a=row['a_constrained'],
                        IGPE_vs_uniform=st_u['IGPE_bits'], p_uniform=st_u['p'],
                        IGPE_vs_pop=st_p['IGPE_bits'], p_pop=st_p['p']))
        print(f"  {cfgkey}: vs uni={st_u['IGPE_bits']:+.3f} (p={st_u['p']:.1e}) vs pop={st_p['IGPE_bits']:+.3f}", flush=True)
    out['P3_constrained_forward_IGPE'] = fwd

    # ---------- P4: C1_targets train-only 目标集（A-m4） ----------
    print('== P4 C1 train-only 目标宇宙 ==', flush=True)
    V_tr = sorted(set(dst_all[:cut]))
    m_t = GraphHawkes(t_all[:cut], list(dst_all[:cut]), list(src_all[:cut]), adj, nodes,
                      dst_extra=V_tr, kernel='omori', gk='hop1', bg='hod24', W=86400.0,
                      qbg='pop', cand_list=V_tr)
    th_t, ll_t, a_t = fit_constrained(m_t, a_max=A_MAX, n_restarts=6, seed=11)
    cntT = np.zeros(m_t.nC)
    for d in dst_all[:cut]:
        if d in m_t.cidx:
            cntT[m_t.cidx[d]] += 1
    popT = (cntT + 0.5) / (cntT + 0.5).sum()
    ctx = dict(pop=popT, m_h=m_h, R0=R0, flag_q=None, theta_iw=0.0, ECNT={'cnt': {}, 'tot': {}})
    keep = [j for j, d in enumerate(dst_all[cut:]) if d in m_t.cidx]
    si_h0 = [m_t.vidx[d] for d in dst_all[:cut]]
    la, lv, ra, rv = score_Ktilde(m_t, th_t, ['uniform', 'pop'],
                                  t_all[cut:][keep], [dst_all[cut:][j] for j in keep],
                                  [src_all[cut:][j] for j in keep],
                                  list(t_all[:cut]), list(dst_all[:cut]), si_h0, ctx)
    c1t = dict(cand_size=len(V_tr), n_scored=len(keep),
               note='train-only 目标集: 无测试信息泄漏的 C1 敏感性(对照原 301 宇宙含测试段新目标)')
    for rk in ['uniform', 'pop']:
        g_ = np.log2(la / ra[rk]); st = igpe(g_)
        c1t[f'vs_{rk}'] = dict(IGPE=st['IGPE_bits'], p=st['p'])
        print(f"  targets_train(|C|={len(V_tr)}, n={len(keep)}) vs {rk}: IGPE={st['IGPE_bits']:+.3f} p={st['p']:.1e}", flush=True)
    out['P4_C1_targets_train_only'] = c1t

    json.dump(out, open(f'{R}/exp11_constrained_ablation.json', 'w'), indent=1, ensure_ascii=False, default=float)
    print(f'落盘 {R}/exp11_constrained_ablation.json; 总耗时 {(time.time()-t00)/60:.1f} min', flush=True)
    open(f'{R}/exp11.done', 'w').write('done\n')

if __name__ == '__main__':
    main()
