#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exp12 Boost vs Flag 前瞻机制判定（审稿B-C1 / 方案一旗舰）
问题重构: 同主机重复攻击是 boost（攻击者保 foothold 回访=真状态依赖, Hawkes hop0 自激语义）
  还是 flag（主机静态易感/高价值=不可观测异质性, frailty/EB 收缩语义）? —— 犯罪学 repeat
  victimization 40 年 boost/flag 之争的逐字同构; 统计对应 Heckman 真状态依赖 vs 不可观测异质性。
模型族（全部 train-only 拟合, 测试段 143 事件逐事件 IGPE=CSEP log2(λ_model/ξ_ref)）:
  M-boost  = hod24·pop 背景 + hop0 自激核（无图核; 即审稿B自建 hop0-only 模型的正式化）
  M-flag   = hod24·q_flag 背景, 无触发核; q_flag=Gamma-Poisson 经验 Bayes 每主机静态风险收缩
             （α 矩法估计自 train 计数过散; 对未见主机 posterior=先验 → 天然解释 unseen 层）
  M-hybrid = hod24·q_flag 背景 + hop0 自激核（嵌套两者）
  参照     = 主模型 hop1（冻结 θ̂, 已验证）与 uniform/pop/hod_pop 锚
预登记判定规则（本脚本运行前落盘, 不看结果）:
  R1 boost 成分存在 ⇔ IGPE(hybrid vs flag) 的块bootstrap 95%CI 下界>0（自激核在静态异质性
     之外仍有时间上前瞻信息; 识别假设=核时间支撑 W 有界, 做 W∈{12,24,48h} 敏感性）
  R2 flag 成分存在 ⇔ IGPE(hybrid vs boost) 的 95%CI 下界>0
  R3 成分构成: share_boost=(hybrid−flag)/[(hybrid−flag)+(hybrid−boost)]（bits 份额, 描述性）
  R4 若 hybrid≈flag（CI 含 0）→ hop0 信号≈静态异质性, 机制叙事=flag 主导（B席 hop0-only 优势
     的异质性解释成立）; 若 hybrid≈boost → boost 主导; 双向成文
辅助证据:
  A. 主模型 hop1 的 EM 触发概率质量按 hop 分解（hop0 vs hop1 份额; B席"top-15 全同主机"复核）
  B. Knox δ=1 恰好（排除同主机 δ=0 对, near-repeat 惯例）6 个 τ 格 × 2000 同 hour 置换
  C. 攻击者连续性: 同主机相邻攻击对（τ≤600s/3600s/86400s）中同一源主机占比
     （boost 的"同一 offender 回访"佐证 vs flag 的"多 attacker 打同一洞"）
  D. seen/unseen 分层 IGPE（flag 对未见主机=先验的预期检验; B席 m5/m8 同源问题）
功效语境: 权威 MDE=1.1043 bits（exp4b D3）
"""
import sys, json, time
from collections import deque
import numpy as np
sys.path.insert(0, './scripts')
sys.path.insert(0, './src')
from exp_patch_common import (load_all, fit_constrained, score_Ktilde, eb_flag_q,
                              moving_block_bootstrap, R)
from lanl_pp import GraphHawkes, hour_profile, hour_coverage_seconds, igpe

A_MAX = 0.95


def flag_model_ll(t_tr, dst_tr, m_h, q_flag, cidx):
    """M-flag 闭式 MLE: λ=μ0·m_h(t)·q_flag(v); μ0*=n/∫m_h dt; 返回 (ll, mu0)"""
    cov = hour_coverage_seconds(t_tr[0], float(t_tr[-1]) + 1.0)
    int_mh = float(cov.sum())
    mu0 = len(t_tr) / int_mh
    ll = 0.0
    for tt, d in zip(t_tr, dst_tr):
        ll += np.log(mu0 * m_h[int((tt % 86400) // 3600)] * q_flag[cidx[d]])
    ll -= mu0 * int_mh
    return float(ll), float(mu0), int_mh


def knox_delta1(D, gi, t, hod, taus, NB=2000, seed=7):
    """δ=1 恰好（排除同主机 δ=0）Knox 置换检验"""
    n = len(t)
    tri = np.triu_indices(n, k=1)
    dp = D[gi[:, None], gi[None, :]][tri]
    dt = np.abs(t[:, None] - t[None, :])[tri].astype(np.int64)
    mask_d = (dp == 1)
    rng = np.random.default_rng(seed)
    res = []
    for tau in taus:
        mask = mask_d & (dt <= tau)
        obs = int(mask.sum())
        sims = np.empty(NB)
        tperm = np.empty(n, dtype=np.int64)
        for k in range(NB):
            perm = np.arange(n)
            for h in np.unique(hod):
                m_ = np.where(hod == h)[0]
                perm[m_] = m_[rng.permutation(len(m_))]
            np.take(t, perm, out=tperm)
            dt_p = np.abs(tperm[:, None] - tperm[None, :])[tri]
            sims[k] = int((mask_d & (dt_p <= tau)).sum())
        p = float((np.sum(sims >= obs) + 1) / (NB + 1))
        res.append(dict(delta=1, tau=tau, obs=obs, h0_mean=round(float(sims.mean()), 1), p=p))
        print(f"  Knox δ=1 τ≤{tau}s: obs={obs} H0={sims.mean():.1f} p={p:.4g}", flush=True)
    return res


def main():
    t00 = time.time()
    D = load_all()
    adj, nodes, ev = D['adj'], D['nodes'], D['ev']
    t_all, dst_all, src_all = D['t_all'], D['dst_all'], D['src_all']
    cut, t_cut, V_all = D['cut'], D['t_cut'], D['V_all']
    m_h = hour_profile(t_all[:cut])
    R0 = cut / (t_cut - t_all[0])
    cntC = np.zeros(len(nodes))
    cidxC = {c: i for i, c in enumerate(nodes)}
    for d in dst_all[:cut]:
        cntC[cidxC[d]] += 1
    popC = (cntC + 0.5) / (cntC + 0.5).sum()
    q_flag, alpha_eb, eb_diag = eb_flag_q(cntC)
    print(f'EB flag: α={alpha_eb:.4g} 诊断={eb_diag}', flush=True)
    ctx_pop = dict(pop=popC, m_h=m_h, R0=R0, flag_q=None, theta_iw=0.0, ECNT={'cnt': {}, 'tot': {}})
    ctx_flag = dict(ctx_pop, flag_q=q_flag)

    out = dict(exp='Exp12_boost_flag_mechanism',
               preregistered_rules=dict(
                   R1='boost 成分存在 ⇔ IGPE(hybrid vs flag) 块bootstrap 95%CI 下界>0',
                   R2='flag 成分存在 ⇔ IGPE(hybrid vs boost) 块bootstrap 95%CI 下界>0',
                   R3='share_boost=(hybrid−flag)/[(hybrid−flag)+(hybrid−boost)] 描述性',
                   R4='hybrid≈flag → flag 主导; hybrid≈boost → boost 主导; 双向成文'),
               eb_flag=dict(alpha=alpha_eb, **eb_diag),
               a_max=A_MAX, MDE_bits_context=1.1043)

    def build(gk, W, qbg, q_custom=None, cand_list=None):
        return GraphHawkes(t_all[:cut], list(dst_all[:cut]), list(src_all[:cut]), adj, nodes,
                           dst_extra=V_all, kernel='omori', gk=gk, bg='hod24', W=W, qbg=qbg,
                           cand_list=cand_list, q_custom=q_custom)

    # ---------- 模型族拟合 ----------
    print('== 模型族拟合 ==', flush=True)
    fam = {}
    # M-flag 闭式
    ll_f, mu0_f, int_mh = flag_model_ll(t_all[:cut], list(dst_all[:cut]), m_h, q_flag, cidxC)
    fam['M_flag'] = dict(n_par=2, n_par_note='μ0 + EB α', train_ll=round(ll_f, 2),
                         AIC=round(2 * 2 - 2 * ll_f, 2), mu0=mu0_f, alpha_eb=alpha_eb)
    print(f"  M-flag: ll={ll_f:.2f} μ0={mu0_f:.3e} AIC={fam['M_flag']['AIC']:.2f}", flush=True)
    # M-boost / M-hybrid: W∈{12,24,48h} × {unconstrained, constrained}
    for W in [43200.0, 86400.0, 172800.0]:
        for qbg, ctx, qkw, nm in [('pop', ctx_pop, {}, 'boost'), ('custom', ctx_flag, dict(q_custom=q_flag), 'hybrid')]:
            for constr in [False, True]:
                key = f"M_{nm}_W{int(W//3600)}h" + ('_c' if constr else '')
                m = build('hop0', W, qbg, **qkw)
                try:
                    if constr:
                        th, ll, a = fit_constrained(m, a_max=A_MAX, n_restarts=6, seed=11)
                    else:
                        m.fit(n_restarts=6, seed=11, t_end=t_cut)
                        th, ll = m.th, m.ll_max
                        a = float(np.exp(th[m.idx['log_kappa']]) * np.asarray(m.G_W(th, m.W)))
                    fam[key] = dict(n_par=m.n_par + (1 if nm == 'hybrid' else 0),
                                    n_par_note='+1 EB α' if nm == 'hybrid' else '',
                                    train_ll=round(ll, 2), AIC=round(2 * m.n_par - 2 * ll, 2),
                                    branch_a=round(a, 4), params=m.params_dict(), W=W)
                    print(f"  {key}: ll={ll:.2f} a={a:.4f} AIC={fam[key]['AIC']:.2f}", flush=True)
                    np.save(f'{R}/exp12_th_{key}.npy', th)
                except Exception as e:
                    fam[key] = dict(error=str(e)[:120])
                    print(f"  {key} ERROR {e}", flush=True)
    # 参照: 主模型 hop1 冻结 θ̂
    m1 = build('hop1', 86400.0, 'pop')
    m1.th = np.load(f'{R}/exp4_th.npy')
    fam['M_hop1_frozen'] = dict(n_par=m1.n_par, train_ll=round(m1.loglik(m1.th), 2), note='exp34 已拟合冻结 θ̂')
    out['model_family'] = fam

    # ---------- 测试段逐事件评分 ----------
    print('== 测试段评分 ==', flush=True)
    t_te, dst_te, src_te = t_all[cut:], list(dst_all[cut:]), list(src_all[cut:])
    si_h0 = None
    lam = {}
    refs = ['uniform', 'pop', 'hod_pop']
    for key, m, ctx, th in [
            ('M_boost_W24', build('hop0', 86400.0, 'pop'), ctx_pop, np.load(f'{R}/exp12_th_M_boost_W24h.npy')),
            ('M_boost_W24_c', build('hop0', 86400.0, 'pop'), ctx_pop, np.load(f'{R}/exp12_th_M_boost_W24h_c.npy')),
            ('M_hybrid_W24', build('hop0', 86400.0, 'custom', q_custom=q_flag), ctx_flag, np.load(f'{R}/exp12_th_M_hybrid_W24h.npy')),
            ('M_hybrid_W24_c', build('hop0', 86400.0, 'custom', q_custom=q_flag), ctx_flag, np.load(f'{R}/exp12_th_M_hybrid_W24h_c.npy')),
            ('M_boost_W12', build('hop0', 43200.0, 'pop'), ctx_pop, np.load(f'{R}/exp12_th_M_boost_W12h_c.npy')),
            ('M_boost_W48', build('hop0', 172800.0, 'pop'), ctx_pop, np.load(f'{R}/exp12_th_M_boost_W48h_c.npy')),
            ('M_hybrid_W12', build('hop0', 43200.0, 'custom', q_custom=q_flag), ctx_flag, np.load(f'{R}/exp12_th_M_hybrid_W12h_c.npy')),
            ('M_hybrid_W48', build('hop0', 172800.0, 'custom', q_custom=q_flag), ctx_flag, np.load(f'{R}/exp12_th_M_hybrid_W48h_c.npy')),
            ('M_hop1_frozen', m1, ctx_pop, m1.th)]:
        si0 = [m.vidx[d] for d in dst_all[:cut]]
        la, lv, ra, rv = score_Ktilde(m, th, refs, t_te, dst_te, src_te,
                                      list(t_all[:cut]), list(dst_all[:cut]), si0, ctx)
        lam[key] = la
        if key == 'M_boost_W24':
            ref_store = ra  # 任一模型参考相同（规范强度）, 存一份
    # M-flag 的 λ 解析: λ_flag(t,v)=μ0·m_h(t)·q_flag(v)
    lam['M_flag'] = np.array([mu0_f * m_h[int((t % 86400) // 3600)] * q_flag[m1.cidx[v]]
                              for t, v in zip(t_te, dst_te)])
    for key, la in lam.items():
        gu = np.log2(la / ref_store['uniform']); gp = np.log2(la / ref_store['pop'])
        gh = np.log2(la / ref_store['hod_pop'])
        st_u, st_p, st_h = igpe(gu), igpe(gp), igpe(gh)
        out.setdefault('test_IGPE', {})[key] = dict(
            vs_uniform=st_u['IGPE_bits'], p_uniform=st_u['p'],
            vs_pop=st_p['IGPE_bits'], vs_hod_pop=st_h['IGPE_bits'],
            n=len(gu))
        np.save(f'{R}/exp12_lam_{key}.npy', la)
        print(f"  {key}: vs uni={st_u['IGPE_bits']:+.3f} vs pop={st_p['IGPE_bits']:+.3f} "
              f"vs hod_pop={st_h['IGPE_bits']:+.3f}", flush=True)

    # ---------- 机制判定边际（预登记规则 R1–R4） ----------
    print('== 机制判定边际 ==', flush=True)
    margins = {}
    for a_, b_ in [('M_hybrid_W24_c', 'M_flag'), ('M_hybrid_W24_c', 'M_boost_W24_c'),
                   ('M_hybrid_W24', 'M_flag'), ('M_hybrid_W24', 'M_boost_W24'),
                   ('M_boost_W24_c', 'M_flag'), ('M_boost_W24_c', 'M_hop1_frozen'),
                   ('M_hybrid_W24_c', 'M_hop1_frozen'), ('M_boost_W24_c', 'M_boost_W24'),
                   ('M_hybrid_W24_c', 'M_hybrid_W24')]:
        g = np.log2(lam[a_] / lam[b_])
        st = igpe(g)
        bb = moving_block_bootstrap(g, block=10, n_boot=5000, seed=13)
        margins[f'{a_}_gt_{b_}'] = dict(IGPE_bits=st['IGPE_bits'], se=st['se'], t=st['t'],
                                        p_wilcoxon=st['p_wilcoxon'],
                                        boot_ci=[bb['ci_lo'], bb['ci_hi']], p_boot_two=bb['p_boot_two'])
        np.save(f'{R}/exp12_margin_{a_}_gt_{b_}.npy', g)
        print(f"  {a_} − {b_}: {st['IGPE_bits']:+.4f} bits  95%CI=[{bb['ci_lo']:+.3f},{bb['ci_hi']:+.3f}] "
              f"p_boot={bb['p_boot_two']:.3g}", flush=True)
    # R3 成分份额
    d_boost = margins['M_hybrid_W24_c_gt_M_flag']['IGPE_bits']
    d_flag = margins['M_hybrid_W24_c_gt_M_boost_W24_c']['IGPE_bits']
    share = d_boost / (d_boost + d_flag) if (d_boost + d_flag) > 0 else float('nan')
    out['R3_share_boost'] = share
    # 判定
    r1 = margins['M_hybrid_W24_c_gt_M_flag']['boot_ci'][0] > 0
    r2 = margins['M_hybrid_W24_c_gt_M_boost_W24_c']['boot_ci'][0] > 0
    verdict = ('boost+flag 并存' if (r1 and r2) else
               'flag 主导（hop0 信号≈静态异质性）' if (not r1 and r2) else
               'boost 主导' if (r1 and not r2) else 'both margins n.s.')
    out['verdict'] = verdict
    out['margins'] = margins
    print(f"  R1 boost 成分: {r1}; R2 flag 成分: {r2} → 判定: {verdict}; share_boost={share:.3f}", flush=True)

    # ---------- 辅助 A: 主模型 EM 触发质量按 hop 分解 ----------
    print('== EM hop 分解（主模型 hop1, train） ==', flush=True)
    p_bg, P = m1.trigger_probs(m1.th)
    si = np.array([m1.vidx[d] for d in dst_all[:cut]])
    HP = m1.HopPair[si][:, si]
    m0_ = float(P[HP == 0].sum()); m1_ = float(P[HP == 1].sum())
    pair = P.ravel()
    hops = HP.ravel()
    top = np.argsort(-pair)[:15]
    out['A_EM_hop_split'] = dict(trigger_mass_hop0=m0_, trigger_mass_hop1=m1_,
                                 hop0_share=m0_ / (m0_ + m1_),
                                 top15_pair_hops=[int(hops[k]) for k in top],
                                 top15_all_hop0=bool(np.all(hops[top] == 0)))
    print(f"  触发质量: hop0={m0_:.1f} hop1={m1_:.3f} → hop0 份额={m0_/(m0_+m1_):.4f}", flush=True)

    # ---------- 辅助 B: Knox δ=1 恰好（排除同主机） ----------
    print('== Knox δ=1 恰好 ==', flush=True)
    V = sorted(set(dst_all)); vidx = {v: i for i, v in enumerate(V)}
    Dm = np.full((len(V), len(V)), 4, dtype=np.int8)
    for v in V:
        Dm[vidx[v], vidx[v]] = 0
        qd = deque([(v, 0)]); seen = {v}
        while qd:
            u, dep = qd.popleft()
            if dep >= 1:
                continue
            for w in adj.get(u, ()):
                if w not in seen:
                    seen.add(w)
                    if w in vidx:
                        Dm[vidx[v], vidx[w]] = dep + 1
                    qd.append((w, dep + 1))
    Dm = np.minimum(Dm, Dm.T)
    gi = np.array([vidx[d] for d in dst_all])
    t_int = ev['t'].astype(np.int64)
    hod = ((t_int % 86400) // 3600).astype(int)
    out['B_knox_delta1_exact'] = knox_delta1(Dm, gi, t_int, hod, [60, 300, 600, 1800, 3600, 86400])

    # ---------- 辅助 C: 攻击者连续性 ----------
    print('== 攻击者连续性 ==', flush=True)
    cont = {}
    for tau in [600.0, 3600.0, 86400.0]:
        same, diff = 0, 0
        for j in range(cut, ev['n']):
            for i in range(j - 1, -1, -1):
                if t_all[j] - t_all[i] > tau:
                    break
                if dst_all[j] == dst_all[i]:
                    if src_all[j] == src_all[i]:
                        same += 1
                    else:
                        diff += 1
        tot = same + diff
        cont[f'tau{int(tau)}s'] = dict(same_src=same, diff_src=diff, frac_same_src=same / tot if tot else None)
        print(f"  τ≤{tau:.0f}s 同主机对: 同源={same} 异源={diff} (同源占比={same/tot if tot else float('nan'):.3f})", flush=True)
    out['C_attacker_continuity'] = cont

    # ---------- 辅助 D: seen/unseen 分层 ----------
    print('== seen/unseen 分层 ==', flush=True)
    train_dst = set(dst_all[:cut])
    seen = np.array([d in train_dst for d in dst_te])
    strat = {}
    for key in ['M_boost_W24_c', 'M_hybrid_W24_c', 'M_flag', 'M_hop1_frozen']:
        row = {}
        for nm, msk in [('seen', seen), ('unseen', ~seen)]:
            gu = np.log2(lam[key][msk] / ref_store['uniform'][msk])
            st = igpe(gu)
            row[nm] = dict(n=int(msk.sum()), vs_uniform=st['IGPE_bits'], p=st['p'])
        strat[key] = row
        print(f"  {key}: seen {row['seen']['vs_uniform']:+.3f} (n={row['seen']['n']}) "
              f"unseen {row['unseen']['vs_uniform']:+.3f} (n={row['unseen']['n']})", flush=True)
    out['D_stratified'] = strat

    json.dump(out, open(f'{R}/exp12_boost_flag.json', 'w'), indent=1, ensure_ascii=False, default=float)
    print(f'落盘 {R}/exp12_boost_flag.json; 总耗时 {(time.time()-t00)/60:.1f} min', flush=True)
    open(f'{R}/exp12.done', 'w').write('done\n')

if __name__ == '__main__':
    main()
