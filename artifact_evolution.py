# ============================================================
# PLOT EVOLUTION ACROSS W&B ARTIFACTS FOR A RUN
# ============================================================
# Memory-optimised version:
#   1. Batched attention forward pass (_ATTN_BATCH chunks)
#   2. Forward pass run ONCE per artifact; results shared by 4 plots
#   3. No _collect_features_and_masks — subsamples read on-demand
#   4. In-place permutation importance (no full-tensor clone)
#   5. _flush() (gc.collect + device cache clear) after every GPU op
#   6. Only layer-0 particle attention extracted (all plots use that)
#   7. Model + state_dict deleted after each artifact
# ============================================================
import os, re, gc, copy, json, shutil, hashlib, argparse
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import vector, wandb, awkward as ak

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    auc,
)
from sklearn.calibration import calibration_curve
from torch.utils.data import DataLoader
from scipy.optimize import curve_fit as _curve_fit

from parT import ParticleTransformer, to_rapidity
import train_part_data
from train_part_data import CombinedJetDataLoader, L1JetDataset, stratified_split
from data_loading_helpers import (
    apply_custom_cuts,
    load_and_prepare_data,
    select_gen_b_quarks_from_higgs,
)
from analysis_helpers import get_purity_mask_cross_matched, calculate_roc_points
from plotting_helpers import plot_roc_comparison

plt.rcParams.update(
    {
        "figure.figsize": (10, 6),
        "axes.grid": True,
        "grid.alpha": 0.6,
        "grid.linestyle": "--",
        "font.size": 14,
        "figure.dpi": 200,
    }
)


def plot_evolution(config_part_path, btag_threshold=None):
    # ── CONFIG ──────────────────────────────────────────────────
    CONFIG_PART = json.load(open(config_part_path, "r"))
    WANDB_RUN_PATH = CONFIG_PART.get("training", {}).get("wandb_run_path", None)
    if WANDB_RUN_PATH is None:
        WANDB_RUN_PATH = "adityatandon29/part-btag-analysis/tex6m5rj"

    MAX_ARTIFACTS = None
    N_IMPORTANCE_SAMPLES = 3000
    N_ATTENTION_SAMPLES = 20000
    _ATTN_BATCH = 512  # mini-batch for attention forward (memory-safe)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    with open("hh-bbbb-obj-config.json") as _f:
        _cuts_config = json.load(_f)

    INPUT_FEATURE_NAMES = [
        "Mass",
        "Pt",
        "Eta",
        "Phi",
        "Dxy",
        "Z0",
        "Charge",
        "Log Pt Rel",
        "Delta Eta",
        "Delta Phi",
        "PUPPI weight",
        "Log Delta R",
    ]

    # ── MEMORY HELPERS ──────────────────────────────────────────
    def _flush():
        """Free unused GPU / MPS memory."""
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # ── REFERENCE ROCs (once) ──────────────────────────────────
    def _build_reference_rocs(cuts_config):
        print("Computing reference ROC curves from ntuples...")
        cfg = cuts_config
        events = load_and_prepare_data(
            cfg["file_pattern"],
            cfg["tree_name"],
            [
                cfg["gen"]["collection_name"],
                cfg["offline"]["collection_name"],
                cfg["l1ng"]["collection_name"],
                cfg["l1ext"]["collection_name"],
            ],
            cfg["max_events"],
            correct_pt=True,
        )
        upart_cfg = copy.deepcopy(cfg)
        upart_cfg["offline"]["tagger_name"] = "btagUParTAK4B"
        upart_cfg["offline"]["b_tag_cut"] = 0.00496
        upart_events = load_and_prepare_data(
            upart_cfg["file_pattern"],
            upart_cfg["tree_name"],
            [upart_cfg["offline"]["collection_name"]],
            upart_cfg["max_events"],
            CONFIG=upart_cfg,
        )

        gen_b = select_gen_b_quarks_from_higgs(events)
        gen_b = gen_b[
            (gen_b.pt > cfg["gen"]["pt_cut"]) & (abs(gen_b.eta) < cfg["gen"]["eta_cut"])
        ]

        off_ev = apply_custom_cuts(
            events[cfg["offline"]["collection_name"]],
            cfg,
            "offline",
            kinematic_only=True,
        )
        off_up_ev = apply_custom_cuts(
            upart_events[upart_cfg["offline"]["collection_name"]],
            upart_cfg,
            "offline",
            kinematic_only=True,
        )
        l1ng_ev = apply_custom_cuts(
            events[cfg["l1ng"]["collection_name"]], cfg, "l1ng", kinematic_only=True
        )
        l1ext_ev = apply_custom_cuts(
            events[cfg["l1ext"]["collection_name"]], cfg, "l1ext", kinematic_only=True
        )

        off_m = get_purity_mask_cross_matched(gen_b, off_ev)
        off_up_m = get_purity_mask_cross_matched(gen_b, off_up_ev)
        l1ng_m = get_purity_mask_cross_matched(gen_b, l1ng_ev)
        l1ext_m = get_purity_mask_cross_matched(gen_b, l1ext_ev)

        off_roc = calculate_roc_points(off_ev, off_m, cfg["offline"]["tagger_name"])
        off_up_roc = calculate_roc_points(
            off_up_ev, off_up_m, upart_cfg["offline"]["tagger_name"]
        )
        l1ng_roc = calculate_roc_points(l1ng_ev, l1ng_m, cfg["l1ng"]["tagger_name"])
        l1ext_roc = calculate_roc_points(l1ext_ev, l1ext_m, cfg["l1ext"]["tagger_name"])

        del events, upart_events, gen_b
        del off_ev, off_up_ev, l1ng_ev, l1ext_ev
        gc.collect()
        print("  Reference ROC curves computed.")
        return {
            "Offline PNet": off_roc,
            "Offline UParT": off_up_roc,
            "L1NG": l1ng_roc,
            "L1ExtJet": l1ext_roc,
        }

    # ── UTILITIES ──────────────────────────────────────────────
    def _safe_name(s):
        return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)

    def _ensure_dir(p):
        os.makedirs(p, exist_ok=True)

    def _artifact_sort_key(a):
        m = a.metadata or {}
        if "epoch" in m:
            return (int(m["epoch"]), a.name or "")
        v = re.search(r"v(\d+)$", a.version or "")
        return (int(v.group(1)), a.name or "") if v else (999999, a.name or "")

    def _find_checkpoint_file(d):
        for root, _, files in os.walk(d):
            for f in files:
                if f.endswith(".pth"):
                    return os.path.join(root, f)
        raise FileNotFoundError(f"No .pth in {d}")

    def _load_checkpoint(path, dev):
        ckpt = torch.load(path, map_location=dev, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        cfg = ckpt.get("config", None)
        for k in (
            "optimiser_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
        ):
            ckpt.pop(k, None)
        return sd, cfg

    def _build_model(cfg, dev):
        cm = cfg.get("model", {})
        return ParticleTransformer(
            input_dim=cfg.get("input_dim", CONFIG_PART["input_dim"]),
            embed_dim=cm.get("embed_dim", CONFIG_PART["model"]["embed_dim"]),
            num_heads=cm.get("num_heads", CONFIG_PART["model"]["num_heads"]),
            num_layers=cm.get("num_layers", CONFIG_PART["model"]["num_layers"]),
            num_cls_layers=cm.get(
                "num_cls_layers", CONFIG_PART["model"]["num_cls_layers"]
            ),
            dropout=cm.get("dropout", CONFIG_PART["model"]["dropout"]),
            num_classes=cm.get("num_classes", CONFIG_PART["model"]["num_classes"]),
            pt_regression=cm.get("pt_regression", False),
            quantile_regression=cm.get("quantile_regression", False),
        ).to(dev)

    def _get_cfg_value(cfg, path, default=None):
        c = cfg
        for k in path:
            if not isinstance(c, dict) or k not in c:
                return default
            c = c[k]
        return c

    def _build_val_loader_from_cfg(cfg):
        try:
            data_cfg = _get_cfg_value(cfg, ["training", "data"], None)
            if data_cfg is not None:
                pf_path = data_cfg.get("pf_data_path", data_cfg.get("pf_path"))
                puppi_path = data_cfg.get("puppi_data_path", data_cfg.get("puppi_path"))
                val_split = data_cfg.get("val_split", 0.3)
                bs = _get_cfg_value(
                    cfg, ["training", "batch_size"], data_cfg.get("batch_size", 512)
                )
                nw = data_cfg.get("num_workers", 0)
                mm = data_cfg.get("match_mode", "size_and_ratio")
                use_ds = data_cfg.get("use_dataset", "pf")
                pt_reg = cfg.get("model", {}).get("pt_regression", False)
                if pf_path and puppi_path:
                    ldr = CombinedJetDataLoader(
                        pf_data_path=pf_path,
                        puppi_data_path=puppi_path,
                        val_split=val_split,
                        batch_size=bs,
                        match_mode=mm,
                        num_workers=2,
                        random_state=42,
                        pt_regression=pt_reg,
                    )
                    _, _, vl, _, _, _ = (
                        ldr.get_puppi_loaders(shuffle=False)
                        if use_ds == "puppi"
                        else ldr.get_pf_loaders(shuffle=False)
                    )
                    return vl
            dp = cfg.get("data_path")
            if dp:
                ds = L1JetDataset(filepath=dp)
                nc = cfg.get("model", {}).get("num_classes", 1)
                vs = cfg.get("training", {}).get("val_split", 0.3)
                bs = cfg.get("training", {}).get("batch_size", 512)
                nw = cfg.get("training", {}).get("num_workers", 0)
                _, vds, _, _, _ = stratified_split(
                    ds, vs, nc, random_state=42, verbose=False
                )
                return DataLoader(vds, batch_size=bs, shuffle=False, num_workers=nw)
            raise ValueError("No data path in config")
        except Exception as e:
            print(
                f"  Warning: val_loader from cfg failed ({e}), falling back to global."
            )
            raise

    # ── DATA HELPERS (memory-lean) ─────────────────────────────
    def _reconstruct_jet_kinematics(loader):
        pts, etas = [], []
        for x, *_ in loader:
            xn = x.numpy()
            cv = vector.array(
                {
                    "pt": xn[:, :, 1],
                    "eta": xn[:, :, 2],
                    "phi": xn[:, :, 3],
                    "mass": xn[:, :, 0],
                }
            )
            jv = cv.sum(axis=1)
            pts.append(jv.pt)
            etas.append(jv.eta)
        return np.concatenate(pts), np.concatenate(etas)

    def _compute_cuts_mask(pt, eta, ccfg, key="l1barrelextpf"):
        sc = ccfg[key]
        return (pt > sc.get("pt_cut", 0.0)) & (np.abs(eta) < sc.get("eta_cut", 2.4))

    def _data_cfg_hash(cfg):
        dc = _get_cfg_value(cfg, ["training", "data"], None)
        dp = cfg.get("data_path")
        return hashlib.md5(
            json.dumps({"d": dc, "p": dp}, sort_keys=True, default=str).encode()
        ).hexdigest()

    # ── INFERENCE ──────────────────────────────────────────────
    def _infer_outputs(model, loader, dev):
        """Run inference and collect classification scores, labels,
        regression corrections, quantile outputs, and jet/gen kinematics.

        Returns
        -------
        scores, labels, reg_pts, quantiles, jet_pts, gen_pts, jet_etas
        reg_pts / quantiles are None when the model has no such head.
        """
        model.eval()
        ss, ll = [], []
        rr, qq = [], []
        jpts, gpts, jetas = [], [], []
        with torch.no_grad():
            for batch in loader:
                x, y, m = batch[0], batch[1], batch[2]
                # batch may have 4 or 7 elements depending on dataset
                jet_pt_b = batch[4].squeeze() if len(batch) > 4 else None
                gen_pt_b = batch[6].squeeze() if len(batch) > 6 else None
                x, m = x.to(dev), m.to(dev)
                out = model(x, particle_mask=m)
                logits = out["classification"].view(-1)
                ss.append(torch.sigmoid(logits).cpu().numpy())
                ll.append(y.view(-1).numpy())
                if "pt" in out:
                    rr.append(out["pt"].squeeze().cpu().numpy())
                if "quantiles" in out:
                    qq.append(out["quantiles"].cpu().numpy())
                if jet_pt_b is not None:
                    jpts.append(jet_pt_b.numpy())
                if gen_pt_b is not None:
                    gpts.append(gen_pt_b.numpy())
                # reconstruct jet eta from constituents for this batch
                xn = batch[0].numpy()
                cv = vector.array(
                    {
                        "pt": xn[:, :, 1],
                        "eta": xn[:, :, 2],
                        "phi": xn[:, :, 3],
                        "mass": xn[:, :, 0],
                    }
                )
                jetas.append(cv.sum(axis=1).eta)
                del x, m, logits, out
        _flush()
        scores = np.concatenate(ss)
        labels = np.concatenate(ll)
        reg_pts = np.concatenate(rr) if rr else None
        quantiles = np.concatenate(qq) if qq else None
        jet_pts = np.concatenate(jpts) if jpts else None
        gen_pts = np.concatenate(gpts) if gpts else None
        jet_etas = np.concatenate(jetas) if jetas else None
        return scores, labels, reg_pts, quantiles, jet_pts, gen_pts, jet_etas

    def _save_fig(fig, base, plot_name, stem):
        d = os.path.join(base, plot_name)
        _ensure_dir(d)
        fig.savefig(os.path.join(d, f"{stem}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── BATCHED ATTENTION FORWARD (memory-safe) ────────────────
    def _forward_with_attention_batched(model, loader, need_attn=True, need_acts=True):
        """Process N_ATTENTION_SAMPLES through the model in _ATTN_BATCH-sized chunks.
        Only layer-0 particle attention is extracted (the only layer used by plots).

        Returns
        -------
        layer0_attn_mean : ndarray (N, P, P)  — head-averaged layer-0 attn, or None
        act_norms        : ndarray (N, P)     — L2 norm of final particle acts, or None
        x_cpu            : Tensor  (N, P, F)  — raw input features (CPU)
        mask_np          : ndarray (N, P)     — boolean mask
        labels_np        : ndarray (N,)       — 0/1 labels
        """
        model.eval()
        # Collect subsample from loader
        xs, ms, ys = [], [], []
        count = 0
        for xb, yb, mb, *_ in loader:
            xs.append(xb)
            ms.append(mb)
            ys.append(yb.view(-1))
            count += xb.shape[0]
            if count >= N_ATTENTION_SAMPLES:
                break
        x_all = torch.cat(xs)[:N_ATTENTION_SAMPLES]
        m_all = torch.cat(ms)[:N_ATTENTION_SAMPLES]
        labels_np = torch.cat(ys)[:N_ATTENTION_SAMPLES].numpy()
        del xs, ms, ys

        N = x_all.shape[0]
        attn_parts = [] if need_attn else None
        act_parts = [] if need_acts else None

        for s in range(0, N, _ATTN_BATCH):
            e = min(s + _ATTN_BATCH, N)
            x = x_all[s:e].to(device)
            mask = m_all[s:e].to(device)

            with torch.no_grad():
                xr = x
                if model.use_batch_norm:
                    xp = model.input_bn(x.transpose(1, 2)).transpose(1, 2)
                else:
                    xp = model.input_ln(x)
                u_ij = model.pair_embed(xr, mask)
                xp = model.embed(xp)

                for li, block in enumerate(model.part_atten_blocks):
                    if li == 0 and need_attn:
                        B2, N2, C2 = xp.shape
                        qkv = (
                            block.qkv(xp)
                            .reshape(B2, N2, 3, block.num_heads, C2 // block.num_heads)
                            .permute(2, 0, 3, 1, 4)
                        )
                        q, k = qkv[0], qkv[1]
                        att = (q @ k.transpose(-2, -1)) * block.scale
                        if mask is not None:
                            att = att.masked_fill(
                                ~mask.unsqueeze(1).unsqueeze(2), float("-inf")
                            )
                        att = att + block.pair_proj(u_ij).permute(0, 3, 1, 2)
                        aw = att.softmax(dim=-1)
                        attn_parts.append(aw.mean(dim=1).cpu().numpy())  # (B, P, P)
                        del qkv, q, k, att, aw
                    xp = block(xp, u_ij, mask)

                if need_acts:
                    act_parts.append(torch.norm(xp, dim=-1).cpu().numpy())  # (B, P)

            del x, mask, xr, u_ij, xp
            _flush()

        attn_mean = np.concatenate(attn_parts) if need_attn else None
        act_norms = np.concatenate(act_parts) if need_acts else None
        x_cpu = x_all  # already CPU tensor
        mask_np = m_all.numpy()
        del x_all, m_all
        _flush()
        return attn_mean, act_norms, x_cpu, mask_np, labels_np

    # ── RESOLUTION / REGRESSION HELPERS ────────────────────────
    def _gaussian(x, mu, sigma, A):
        return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def _fit_response_in_bin(response, bins=np.linspace(0, 2, 51)):
        bc = 0.5 * (bins[1:] + bins[:-1])
        counts, _ = np.histogram(response, bins=bins)
        try:
            popt, pcov = _curve_fit(
                _gaussian,
                bc,
                counts,
                absolute_sigma=True,
                p0=[1.0, 0.1, max(counts.max(), 1)],
            )
            return popt, pcov
        except RuntimeError:
            return (np.nan, np.nan, np.nan), np.zeros((3, 3))

    def _get_resolution_vs_var(gen_var, response, var_bins):
        bc = 0.5 * (var_bins[1:] + var_bins[:-1])
        mus, sigs, mu_e, sig_e = [], [], [], []
        for i in range(len(var_bins) - 1):
            mask = (gen_var > var_bins[i]) & (gen_var <= var_bins[i + 1])
            vals = response[mask]
            if len(vals) > 20:
                (mu, sigma, _A), cov = _fit_response_in_bin(vals)
                mus.append(mu if not np.isnan(mu) else np.nan)
                sigs.append(abs(sigma) if not np.isnan(sigma) else np.nan)
                mu_e.append(np.sqrt(cov[0, 0]) if cov[0, 0] > 0 else 0.0)
                sig_e.append(np.sqrt(cov[1, 1]) if cov[1, 1] > 0 else 0.0)
            else:
                mus.append(np.nan)
                sigs.append(np.nan)
                mu_e.append(0.0)
                sig_e.append(0.0)
        return bc, np.array(mus), np.array(sigs), np.array(mu_e), np.array(sig_e)

    # ── REGRESSION EVOLUTION PLOTS ─────────────────────────────
    def _plot_regression_analysis(reg_pts, labels, jet_pts, gen_pts, base, stem):
        """Regression head analysis: correction distributions, scatter, corrected pT."""
        sig = labels.squeeze() == 1
        if sig.sum() < 10:
            return
        jp, gp, rp = jet_pts[sig], gen_pts[sig], reg_pts[sig]
        true_corr = gp / (jp + 1e-9)
        pred_corr = rp
        corrected = jp * pred_corr

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # Row 0: pT distributions
        ax = axes[0, 0]
        ax.hist(
            jp, bins=60, range=(0, 500), histtype="step", density=True, label="Signal"
        )
        ax.hist(
            jet_pts[~sig],
            bins=60,
            range=(0, 500),
            histtype="step",
            density=True,
            label="Background",
        )
        ax.set_xlabel("Reco jet $p_T$ [GeV]")
        ax.set_ylabel("Density")
        ax.set_title("Reco jet $p_T$ distribution")
        ax.legend()

        ax = axes[0, 1]
        ax.hist(gp, bins=60, range=(0, 500), histtype="step", density=True, color="C0")
        ax.set_xlabel("Gen $p_T$ [GeV]")
        ax.set_ylabel("Density")
        ax.set_title("Gen $p_T$ (signal jets only)")

        ax = axes[0, 2]
        ax.hist(
            true_corr, bins=60, range=(0, 3), histtype="step", density=True, color="C2"
        )
        ax.axvline(1.0, color="gray", linestyle="--", alpha=0.7)
        ax.set_xlabel("gen $p_T$ / reco $p_T$")
        ax.set_ylabel("Density")
        ax.set_title("Target correction factor (signal)")

        # Row 1: regression outputs
        ax = axes[1, 0]
        ax.hist(rp, bins=60, histtype="step", density=True, label="Signal")
        ax.hist(
            reg_pts[~sig], bins=60, histtype="step", density=True, label="Background"
        )
        ax.set_xlabel("Regression output")
        ax.set_ylabel("Density")
        ax.set_title("Regression head output")
        ax.legend()

        ax = axes[1, 1]
        ax.scatter(true_corr, pred_corr, s=1, alpha=0.3)
        lim = max(true_corr.max(), pred_corr.max()) * 1.05
        ax.plot([0, lim], [0, lim], "r--", alpha=0.5, label="Ideal")
        ax.set_xlabel("True correction")
        ax.set_ylabel("Predicted correction")
        ax.set_title("Predicted vs true correction")
        ax.legend()

        ax = axes[1, 2]
        ax.scatter(gp, corrected, s=1, alpha=0.3, label="Corrected")
        ax.scatter(gp, jp, s=1, alpha=0.1, color="gray", label="Uncorrected")
        ax.plot([0, 500], [0, 500], "r--", alpha=0.5)
        ax.set_xlim(0, 500)
        ax.set_ylim(0, 500)
        ax.set_xlabel("Gen $p_T$ [GeV]")
        ax.set_ylabel("Jet $p_T$ [GeV]")
        ax.set_title("Corrected vs uncorrected $p_T$")
        ax.legend()

        plt.tight_layout()
        _save_fig(fig, base, "regression_head_analysis", stem)

    # ── pT RESPONSE SCALE & RESOLUTION PLOTS ──────────────────
    def _plot_resolution_vs_pt(reg_pts, labels, jet_pts, gen_pts, base, stem):
        """Scale & resolution vs gen pT, before and after correction."""
        sig = labels.squeeze() == 1
        if sig.sum() < 50:
            return
        jp, gp = jet_pts[sig], gen_pts[sig]
        raw_resp = jp / (gp + 1e-9)
        corr_resp = (
            (jp * reg_pts[sig]) / (gp + 1e-9) if reg_pts is not None else raw_resp
        )

        pt_bins = np.linspace(25, 500, 20)
        bc, mu_r, sig_r, mue_r, sige_r = _get_resolution_vs_var(gp, raw_resp, pt_bins)
        bc, mu_c, sig_c, mue_c, sige_c = _get_resolution_vs_var(gp, corr_resp, pt_bins)

        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        ax = axes[0]
        ax.errorbar(
            bc,
            mu_r,
            yerr=mue_r,
            marker="o",
            linestyle="-",
            capsize=4,
            label="Uncorrected",
            color="C0",
        )
        if reg_pts is not None:
            ax.errorbar(
                bc,
                mu_c,
                yerr=mue_c,
                marker="s",
                linestyle="-",
                capsize=4,
                label="Corrected",
                color="C1",
            )
        ax.axhline(1.0, color="black", linewidth=0.8)
        ax.set_ylabel("Jet $p_T$ Scale (Mean)")
        ax.set_title("Scale & Resolution vs Gen $p_T$")
        ax.legend()
        ax.set_ylim(0.5, 1.5)

        ax = axes[1]
        ax.errorbar(
            bc,
            sig_r,
            yerr=sige_r,
            marker="o",
            linestyle="-",
            capsize=4,
            label="Uncorrected",
            color="C0",
        )
        if reg_pts is not None:
            ax.errorbar(
                bc,
                sig_c,
                yerr=sige_c,
                marker="s",
                linestyle="-",
                capsize=4,
                label="Corrected",
                color="C1",
            )
        ax.set_xlabel("Generated Jet $p_T$ [GeV]")
        ax.set_ylabel("Resolution ($\\sigma$)")
        ax.legend()
        ax.set_ylim(0)
        plt.tight_layout()
        _save_fig(fig, base, "resolution_vs_gen_pt", stem)

    def _plot_resolution_vs_eta(
        reg_pts, labels, jet_pts, gen_pts, jet_etas, base, stem
    ):
        """Scale & resolution vs jet eta, before and after correction."""
        sig = labels.squeeze() == 1
        if sig.sum() < 50:
            return
        jp, gp, je = jet_pts[sig], gen_pts[sig], jet_etas[sig]
        raw_resp = jp / (gp + 1e-9)
        corr_resp = (
            (jp * reg_pts[sig]) / (gp + 1e-9) if reg_pts is not None else raw_resp
        )

        eta_bins = np.linspace(-2.4, 2.4, 13)
        bc, mu_r, sig_r, mue_r, sige_r = _get_resolution_vs_var(je, raw_resp, eta_bins)
        bc, mu_c, sig_c, mue_c, sige_c = _get_resolution_vs_var(je, corr_resp, eta_bins)

        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        ax = axes[0]
        ax.errorbar(
            bc,
            mu_r,
            yerr=mue_r,
            marker="o",
            linestyle="-",
            capsize=4,
            label="Uncorrected",
            color="C0",
        )
        if reg_pts is not None:
            ax.errorbar(
                bc,
                mu_c,
                yerr=mue_c,
                marker="s",
                linestyle="-",
                capsize=4,
                label="Corrected",
                color="C1",
            )
        ax.axhline(1.0, color="black", linewidth=0.8)
        ax.set_ylabel("Jet $p_T$ Scale (Mean)")
        ax.set_title("Scale & Resolution vs Jet $\\eta$")
        ax.legend()
        ax.set_ylim(0.5, 1.5)

        ax = axes[1]
        ax.errorbar(
            bc,
            sig_r,
            yerr=sige_r,
            marker="o",
            linestyle="-",
            capsize=4,
            label="Uncorrected",
            color="C0",
        )
        if reg_pts is not None:
            ax.errorbar(
                bc,
                sig_c,
                yerr=sige_c,
                marker="s",
                linestyle="-",
                capsize=4,
                label="Corrected",
                color="C1",
            )
        ax.set_xlabel("Jet $\\eta$")
        ax.set_ylabel("Resolution ($\\sigma$)")
        ax.legend()
        ax.set_ylim(0)
        plt.tight_layout()
        _save_fig(fig, base, "resolution_vs_eta", stem)

    # ── DI-HIGGS MASS PLOTS (evolution) ───────────────────────
    def _dihiggs_pair_from_4jets(j4_pt, j4_eta, j4_phi, j4_mass):
        """D_HH minimisation pairing for (N, 4) arrays. Returns lead, sub, hh masses."""
        j = [
            vector.array(
                {
                    "pt": j4_pt[:, i],
                    "eta": j4_eta[:, i],
                    "phi": j4_phi[:, i],
                    "mass": j4_mass[:, i],
                }
            )
            for i in range(4)
        ]
        perm_pairs = [([0, 1], [2, 3]), ([0, 2], [1, 3]), ([0, 3], [1, 2])]
        m1s, m2s = [], []
        h_vecs_list = []
        for (a, b), (c, d) in perm_pairs:
            h1 = j[a] + j[b]
            h2 = j[c] + j[d]
            m1s.append(h1.mass)
            m2s.append(h2.mass)
            h_vecs_list.append((h1, h2))
        m1 = np.stack(m1s, axis=1)  # (N, 3)
        m2 = np.stack(m2s, axis=1)
        d_hh = np.abs(m1 - (125.0 / 120.0) * m2) / np.sqrt(1 + (125.0 / 120.0) ** 2)
        best = np.argmin(d_hh, axis=1)
        N = len(best)
        lead_m, sub_m, hh_m = np.zeros(N), np.zeros(N), np.zeros(N)
        for i in range(N):
            h1, h2 = h_vecs_list[best[i]]
            v1 = vector.obj(pt=h1.pt[i], eta=h1.eta[i], phi=h1.phi[i], mass=h1.mass[i])
            v2 = vector.obj(pt=h2.pt[i], eta=h2.eta[i], phi=h2.phi[i], mass=h2.mass[i])
            if v1.pt >= v2.pt:
                lead, sub = v1, v2
            else:
                lead, sub = v2, v1
            hh = lead + sub
            lead_m[i] = lead.mass
            sub_m[i] = sub.mass
            hh_m[i] = hh.mass
        return lead_m, sub_m, hh_m

    def _plot_dihiggs_mass(
        model,
        loader,
        cuts_cfg,
        dev,
        collection_key,
        apply_correction,
        btag_thr,
        base,
        stem,
    ):
        """Run di-Higgs mass reconstruction and plot signal vs QCD."""
        from make_dataset import cluster_candidates
        from data_loading_helpers import one_hot_encode_l1_puppi

        cfg = cuts_cfg
        # Determine collection
        dset_used = (
            CONFIG_PART.get("training", {}).get("data", {}).get("use_dataset", "pf")
        )
        if dset_used == "pf":
            ck = "l1extpf"
        elif dset_used == "puppi":
            ck = "l1extpuppi"
        else:
            ck = "l1barrelextpf"
        col = cfg[ck]["collection_name"]

        # Load events
        events = load_and_prepare_data(
            cfg["file_pattern"],
            cfg["tree_name"],
            [col, "GenPart"],
            max_events=cfg["max_events"],
            correct_pt=False,
            CONFIG=cfg,
        )
        clustered = cluster_candidates(events, cfg, ck, dist_param=0.4)
        si = ak.argsort(clustered.pt, axis=1, ascending=False)
        cl = clustered[si]
        mc = cl.constituents
        mc = mc[ak.argsort(mc.pt, axis=2, ascending=False)]

        # Determine n_const from loader
        n_const = 16
        try:
            for xb, *_ in loader:
                n_const = xb.shape[1]
                break
        except Exception:
            pass

        # Build features
        j_pt = cl.pt[:, :, None]
        j_eta = cl.eta[:, :, None]
        j_phi = cl.phi[:, :, None]
        m_pt = mc.vector.pt
        m_eta = mc.vector.eta
        m_phi = mc.vector.phi
        m_mass = mc.vector.mass
        m_dxy = mc.dxy
        m_z0 = mc.z0
        m_charge = mc.charge
        m_w = mc.puppiWeight
        m_id = mc.id
        log_pt_rel = np.log(np.maximum(m_pt, 1e-3) / np.maximum(j_pt, 1e-3))
        deta = m_eta - j_eta
        dphi = (m_phi - j_phi + np.pi) % (2 * np.pi) - np.pi
        log_dr = np.log(np.maximum(np.sqrt(deta**2 + dphi**2), 1e-3))

        def pf(arr):
            return ak.fill_none(ak.pad_none(arr, n_const, axis=2, clip=True), 0.0)

        flist = [
            pf(m_mass),
            pf(m_pt),
            pf(m_eta),
            pf(m_phi),
            pf(m_dxy),
            pf(m_z0),
            pf(m_charge),
            pf(log_pt_rel),
            pf(deta),
            pf(dphi),
            pf(m_w),
            pf(log_dr),
            pf(m_id),
        ]
        n_jets_per_ev = ak.num(cl, axis=1)
        n_actual = ak.num(mc, axis=2)
        n_actual_flat = ak.to_numpy(ak.flatten(n_actual, axis=1))
        x_ini = np.stack([ak.to_numpy(ak.flatten(f, axis=1)) for f in flist], axis=-1)
        flat_ids = x_ini[..., -1]
        oh = one_hot_encode_l1_puppi(flat_ids, n_classes=5)
        X_all = np.concatenate([x_ini[..., :-1], oh], axis=-1)
        pmask = np.zeros((X_all.shape[0], n_const), dtype=bool)
        for i in range(X_all.shape[0]):
            nr = min(n_actual_flat[i], n_const)
            pmask[i, :nr] = True
        cv = vector.array(
            {
                "pt": x_ini[:, :, 1],
                "eta": x_ini[:, :, 2],
                "phi": x_ini[:, :, 3],
                "mass": x_ini[:, :, 0],
            }
        )
        jv = cv.sum(axis=1)
        fpt, feta, fphi, fmass = jv.pt, jv.eta, jv.phi, jv.mass

        # Inference
        infer_bs = 512
        all_sc, all_rc = [], []
        model.eval()
        with torch.no_grad():
            for s in range(0, len(X_all), infer_bs):
                e = min(s + infer_bs, len(X_all))
                xb = torch.tensor(X_all[s:e], dtype=torch.float32).to(dev)
                mb = torch.tensor(pmask[s:e], dtype=torch.bool).to(dev)
                out = model(xb, particle_mask=mb)
                all_sc.append(
                    torch.sigmoid(out["classification"]).squeeze().cpu().numpy()
                )
                if "pt" in out:
                    all_rc.append(out["pt"].squeeze().cpu().numpy())
                del xb, mb, out
        _flush()
        all_sc = np.concatenate(all_sc)
        has_r = len(all_rc) > 0
        all_rc = np.concatenate(all_rc) if has_r else np.ones_like(all_sc)
        cpt = fpt * all_rc if (has_r and apply_correction) else fpt

        # Unflatten
        njnp = ak.to_numpy(n_jets_per_ev)
        cum = np.concatenate([[0], np.cumsum(njnp)])
        ev_pts, ev_etas, ev_phis, ev_masses, ev_scores = [], [], [], [], []
        for i in range(len(njnp)):
            s, e = cum[i], cum[i + 1]
            ev_pts.append(cpt[s:e])
            ev_etas.append(feta[s:e])
            ev_phis.append(fphi[s:e])
            ev_masses.append(fmass[s:e] * (cpt[s:e] / (fpt[s:e] + 1e-9)))
            ev_scores.append(all_sc[s:e])
        scored = ak.zip(
            {
                "pt": ak.Array(ev_pts),
                "eta": ak.Array(ev_etas),
                "phi": ak.Array(ev_phis),
                "mass": ak.Array(ev_masses),
                "btag_score": ak.Array(ev_scores),
            }
        )
        scored["vector"] = ak.zip(
            {
                "pt": scored.pt,
                "eta": scored.eta,
                "phi": scored.phi,
                "mass": scored.mass,
            },
            with_name="Momentum4D",
        )

        # B-tag cut and pairing
        jbs = scored[ak.argsort(scored.btag_score, ascending=False)]
        jt = jbs[jbs.btag_score > btag_thr]
        has4 = ak.num(jt) >= 4
        j4 = jt[has4][:, :4]
        if ak.sum(has4) < 5:
            print(
                f"    Not enough events ({ak.sum(has4)}) with >=4 tagged jets for di-Higgs."
            )
            del events, clustered, cl, mc, scored, jbs, jt
            gc.collect()
            _flush()
            return

        # Purity matching
        gen_b = select_gen_b_quarks_from_higgs(events)
        gen_b = gen_b[
            (gen_b.pt > cfg["gen"]["pt_cut"]) & (abs(gen_b.eta) < cfg["gen"]["eta_cut"])
        ]
        gb4 = gen_b[has4]
        dr_r = j4[:, :, None].vector.deltaR(gb4[:, None, :].vector)
        idx_g = ak.argmin(dr_r, axis=2)
        min_dr = ak.fill_none(ak.min(dr_r, axis=2), np.inf)
        dr_g = gb4[:, :, None].vector.deltaR(j4[:, None, :].vector)
        idx_r = ak.argmin(dr_g, axis=2)
        bc2 = idx_r[idx_g]
        ri = ak.local_index(j4, axis=1)
        pm = (ak.fill_none(bc2, -1) == ri) & (min_dr < cfg["matching_cone_size"])
        sig_evt = ak.sum(pm, axis=1) == 4
        qcd_evt = ~sig_evt

        # Reconstruct masses
        def _to_np4(jets):
            return (
                ak.to_numpy(jets.pt),
                ak.to_numpy(jets.eta),
                ak.to_numpy(jets.phi),
                ak.to_numpy(jets.mass),
            )

        sig_lead_m = sig_sub_m = sig_hh_m = np.array([])
        qcd_lead_m = qcd_sub_m = qcd_hh_m = np.array([])
        n_sig = int(ak.sum(sig_evt))
        n_qcd = int(ak.sum(qcd_evt))
        if n_sig > 0:
            sp, se, sphi, sm = _to_np4(j4[sig_evt])
            sig_lead_m, sig_sub_m, sig_hh_m = _dihiggs_pair_from_4jets(sp, se, sphi, sm)
        if n_qcd > 0:
            qp, qe, qphi, qm = _to_np4(j4[qcd_evt])
            qcd_lead_m, qcd_sub_m, qcd_hh_m = _dihiggs_pair_from_4jets(qp, qe, qphi, qm)

        # Plot 1D mass distributions
        color = "purple"
        bins_h = np.linspace(0, 300, 61)
        bins_hh = np.linspace(200, 800, 61)
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        for ax, (s_m, q_m, xlabel, title) in zip(
            axes,
            [
                (sig_lead_m, qcd_lead_m, "Leading $m_H$ [GeV]", "Leading Higgs"),
                (sig_sub_m, qcd_sub_m, "Subleading $m_H$ [GeV]", "Subleading Higgs"),
                (sig_hh_m, qcd_hh_m, "$m_{HH}$ [GeV]", "$m_{HH}$"),
            ],
        ):
            bns = bins_hh if "HH" in title else bins_h
            if len(s_m) > 0:
                ax.hist(
                    s_m,
                    bins=bns,
                    histtype="stepfilled",
                    alpha=0.3,
                    color=color,
                    label=f"Signal ({n_sig})",
                    density=True,
                )
                ax.hist(
                    s_m,
                    bins=bns,
                    histtype="step",
                    linewidth=2,
                    color=color,
                    density=True,
                )
            if len(q_m) > 0:
                ax.hist(
                    q_m,
                    bins=bns,
                    histtype="step",
                    linewidth=2,
                    color="grey",
                    linestyle="--",
                    label=f"QCD bkg ({n_qcd})",
                    density=True,
                )
            if "HH" not in title:
                ax.axvline(125, color="green", linestyle=":", linewidth=1.5)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Density")
            ax.set_title(title)
            ax.legend(fontsize=10)

        reg_tag = "pT-corrected" if (has_r and apply_correction) else "uncorrected"
        fig.suptitle(f"Di-Higgs Reconstruction ({reg_tag})", fontsize=14, y=1.01)
        plt.tight_layout()
        _save_fig(fig, base, "dihiggs_mass_1d", stem)

        # 2D mH1 vs mH2
        if n_sig > 5 or n_qcd > 5:
            bins_2d = np.linspace(0, 300, 61)
            fig, axes2 = plt.subplots(1, 2, figsize=(18, 7))
            for ax, (lm, sm2, n_ev, cat) in zip(
                axes2,
                [
                    (sig_lead_m, sig_sub_m, n_sig, "Signal"),
                    (qcd_lead_m, qcd_sub_m, n_qcd, "QCD Background"),
                ],
            ):
                if n_ev > 0:
                    ax.hist2d(lm, sm2, bins=[bins_2d, bins_2d], cmap="viridis", cmin=1)
                    ax.axvline(125, color="red", linestyle="--", linewidth=1.5)
                    ax.axhline(120, color="red", linestyle="--", linewidth=1.5)
                ax.set_xlabel("Leading $m_H$ [GeV]")
                ax.set_ylabel("Subleading $m_H$ [GeV]")
                ax.set_title(f"{cat} ({n_ev} events)")
            fig.suptitle(
                f"2D $m_{{H1}}$ vs $m_{{H2}}$ ({reg_tag})", fontsize=14, y=1.01
            )
            plt.tight_layout()
            _save_fig(fig, base, "dihiggs_mass_2d", stem)

        # Cleanup
        del events, clustered, cl, mc, scored, jbs, jt, gen_b
        gc.collect()
        _flush()

    # ── PLOT FUNCTIONS ─────────────────────────────────────────
    def _plot_model_output_distribution(scores, labels, base, stem):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(
            scores[labels == 0], bins=50, range=(0, 1), histtype="step", label="Bkg"
        )
        ax.hist(
            scores[labels == 1], bins=50, range=(0, 1), histtype="step", label="Sig"
        )
        ax.set_xlabel("Model Output Score")
        ax.set_ylabel("Density")
        ax.set_title("Model Output Distribution")
        ax.legend()
        _save_fig(fig, base, "model_output_distribution", stem)

    def _plot_roc(scores, labels, base, stem):
        fpr, tpr, _ = roc_curve(labels, scores)
        a = roc_auc_score(labels, scores)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, label=f"AUC={a:.3f}")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("ROC Curve")
        ax.legend()
        _save_fig(fig, base, "roc_curve", stem)

    def _plot_roc_comparison(scores, labels, ref_rocs, base, stem):
        pfpr, ptpr, _ = roc_curve(labels, scores)
        pa = roc_auc_score(labels, scores)
        rr = [(l, d) for l, d in ref_rocs.items()]
        rr.append(("Trained ParT", (pfpr, ptpr, pa, _)))
        fig = plot_roc_comparison(rr, working_point=0.1, return_fig=True)
        if fig:
            _save_fig(fig, base, "roc_comparison", stem)

    def _plot_pr(scores, labels, base, stem):
        prec, rec, _ = precision_recall_curve(labels, scores)
        pa = average_precision_score(labels, scores)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(rec, prec, label=f"PR-AUC={pa:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall")
        ax.legend()
        _save_fig(fig, base, "precision_recall_curve", stem)

    def _plot_calibration(scores, labels, base, stem):
        pt, pp = calibration_curve(labels, scores, n_bins=10)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(pp, pt, "o-", label="Model")
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Ideal")
        ax.set_xlabel("Mean Predicted Prob")
        ax.set_ylabel("Fraction Positives")
        ax.set_title("Calibration Curve")
        ax.legend()
        _save_fig(fig, base, "calibration_curve", stem)

    def _plot_significance(scores, labels, base, stem):
        thr = np.linspace(0, 1, 200)
        sig = [
            np.sum((scores >= t) & (labels == 1))
            / np.sqrt(np.sum((scores >= t) & (labels == 0)) + 1e-9)
            for t in thr
        ]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(thr, sig, color="darkgreen")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("S/√B")
        ax.set_title("Significance vs Threshold")
        _save_fig(fig, base, "significance_curve", stem)

    def _plot_auc_heatmap(scores, labels, jpt, jeta, cmask, base, stem):
        pt_r = [
            (25, 100),
            (100, 200),
            (200, 300),
            (300, 400),
            (400, 500),
            (500, np.inf),
            (25, np.inf),
        ]
        eta_r = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.4), (0, 1.5), (0, 2.4)]
        jp = jpt[cmask]
        je = jeta[cmask]
        lb = labels[cmask]
        sc = scores[cmask]
        am = np.zeros((len(pt_r), len(eta_r)))
        cm2 = np.zeros_like(am)
        for j, (pl, ph) in enumerate(pt_r):
            for i, (el, eh) in enumerate(eta_r):
                bm = (jp >= pl) & (jp < ph) & (np.abs(je) >= el) & (np.abs(je) < eh)
                bl, bo = lb[bm], sc[bm]
                cm2[j, i] = len(bl)
                if len(np.unique(bl)) < 2 or len(bl) < 10:
                    am[j, i] = np.nan
                else:
                    try:
                        am[j, i] = roc_auc_score(bl, bo)
                    except Exception:
                        am[j, i] = np.nan
        ptl = [f"[{l},{h})" for l, h in pt_r]
        etl = [f"[{l},{h})" for l, h in eta_r]
        ann = np.empty_like(am, dtype=object)
        for i in range(am.shape[0]):
            for j in range(am.shape[1]):
                ann[i, j] = (
                    f"N={int(cm2[i,j])}\n(N/A)"
                    if np.isnan(am[i, j])
                    else f"{am[i,j]:.3f}\n(N={int(cm2[i,j])})"
                )
        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(
            pd.DataFrame(am, index=ptl, columns=etl),
            annot=ann,
            fmt="",
            cmap="viridis",
            vmin=0.5,
            vmax=1.0,
            ax=ax,
            cbar_kws={"label": "AUC"},
        )
        ax.set_xlabel(r"$|\eta|$")
        ax.set_ylabel("Jet pT [GeV]")
        ax.set_title(r"AUC in pT vs $|\eta|$")
        _save_fig(fig, base, "auc_heatmap_pt_eta", stem)

    # ── FEATURE IMPORTANCE ─────────────────────────────────────
    def _gradient_feature_importance(model, x, mask):
        x = x.detach().clone().requires_grad_(True)
        logits = model(x, particle_mask=mask)["classification"].view(-1)
        torch.sigmoid(logits).mean().backward()
        g = x.grad.detach().abs().cpu().numpy()
        mn = mask.detach().cpu().numpy()
        me = np.broadcast_to(mn[..., None], g.shape)
        gv = g[me].reshape(-1, g.shape[-1])
        del x, logits, g
        _flush()
        if gv.size == 0:
            return None
        imp = gv.mean(axis=0)
        return imp / (imp.sum() + 1e-9)

    def _permutation_importance(model, x, mask, labels):
        """Permutation importance — permutes in-place to save memory."""
        model.eval()
        with torch.no_grad():
            bs = (
                torch.sigmoid(model(x, particle_mask=mask)["classification"].view(-1))
                .cpu()
                .numpy()
            )
        ba = roc_auc_score(labels, bs)
        del bs
        _flush()
        nf = x.shape[-1]
        drops = np.zeros(nf)
        for f in range(nf):
            orig = x[:, :, f].clone()
            vals = x[:, :, f].reshape(-1)
            x[:, :, f] = vals[torch.randperm(vals.shape[0], device=x.device)].reshape(
                x[:, :, f].shape
            )
            with torch.no_grad():
                ps = (
                    torch.sigmoid(
                        model(x, particle_mask=mask)["classification"].view(-1)
                    )
                    .cpu()
                    .numpy()
                )
            x[:, :, f] = orig
            drops[f] = ba - roc_auc_score(labels, ps)
            del orig, ps
        drops = np.maximum(drops, 0)
        _flush()
        return drops / (drops.sum() + 1e-9)

    def _plot_feature_importance(model, loader, base, stem):
        """Load a subsample directly from DataLoader — no cached full tensor."""
        xs, ms, ys = [], [], []
        c = 0
        for xb, yb, mb, *_ in loader:
            xs.append(xb)
            ms.append(mb)
            ys.append(yb.view(-1))
            c += xb.shape[0]
            if c >= N_IMPORTANCE_SAMPLES:
                break
        x = torch.cat(xs)[:N_IMPORTANCE_SAMPLES].to(device)
        m = torch.cat(ms)[:N_IMPORTANCE_SAMPLES].to(device)
        lb = torch.cat(ys)[:N_IMPORTANCE_SAMPLES].numpy()
        del xs, ms, ys

        names = list(INPUT_FEATURE_NAMES)
        nf = x.shape[-1]
        if len(names) > nf:
            names = names[:nf]
        elif len(names) < nf:
            names += [f"f{i}" for i in range(len(names), nf)]

        gi = _gradient_feature_importance(model, x, m)
        if gi is not None:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(range(len(gi)), gi)
            ax.set_xticks(range(len(gi)))
            ax.set_xticklabels(names[: len(gi)], rotation=45, ha="right")
            ax.set_ylabel("Importance")
            ax.set_title("Gradient-based Feature Importance")
            _save_fig(fig, base, "feature_importance_grad", stem)

        pi = _permutation_importance(model, x, m, lb)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(pi)), pi)
        ax.set_xticks(range(len(pi)))
        ax.set_xticklabels(names[: len(pi)], rotation=45, ha="right")
        ax.set_ylabel("Importance")
        ax.set_title("Permutation Feature Importance")
        _save_fig(fig, base, "feature_importance_perm", stem)
        del x, m, lb
        _flush()

    # ── ATTENTION / ACTIVATION PLOTS (use pre-computed data) ───
    def _plot_constituent_activation_maps(act_norms, mask_np, base, stem):
        """Constituent activation bar chart — uses pre-computed act_norms."""
        mean_act = np.sum(act_norms * mask_np, axis=0) / (
            np.sum(mask_np, axis=0) + 1e-9
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(np.arange(len(mean_act)), mean_act)
        ax.set_xlabel("Constituent Index")
        ax.set_ylabel("Mean Activation Magnitude")
        ax.set_title("Constituent Activation Map (Mean)")
        _save_fig(fig, base, "constituent_activation_map", stem)

    def _compute_dr_attn(attn_mean, x_cpu, mask_np, labels_np=None):
        """Vectorised ΔR and attention extraction — no Python loops over pairs."""
        rap = to_rapidity(x_cpu).numpy()
        phi = x_cpu[:, :, 3].numpy()
        N = mask_np.shape[0]
        drs, atts, lbls = [], [], ([] if labels_np is not None else None)
        for i in range(N):
            idx = np.where(mask_np[i])[0]
            if len(idx) < 2:
                continue
            ii, jj = np.triu_indices(len(idx), k=1)
            ra, pa = rap[i, idx], phi[i, idx]
            dr = np.sqrt((ra[ii] - ra[jj]) ** 2 + (pa[ii] - pa[jj]) ** 2)
            at = attn_mean[i, idx[ii], idx[jj]]
            drs.append(dr)
            atts.append(at)
            if lbls is not None:
                lbls.append(np.full(len(dr), labels_np[i]))
        if not drs:
            e = np.array([])
            return (e, e, e) if labels_np is not None else (e, e)
        do, ao = np.concatenate(drs), np.concatenate(atts)
        return (do, ao, np.concatenate(lbls)) if labels_np is not None else (do, ao)

    def _plot_attention_vs_distance(attn_mean, x_cpu, mask_np, base, stem):
        """Attention magnitude vs ΔR (2D histogram) — uses pre-computed attn."""
        dr, at = _compute_dr_attn(attn_mean, x_cpu, mask_np)
        if len(dr) == 0:
            return
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist2d(dr, at, bins=[50, 50], cmap="viridis")
        ax.set_xlabel("ΔR")
        ax.set_ylabel("Attention Weight")
        ax.set_title("Attention Magnitude vs Distance")
        _save_fig(fig, base, "attention_vs_distance", stem)

    def _plot_attention_vs_distance_by_jet_type(
        attn_mean, x_cpu, mask_np, labels_np, base, stem
    ):
        """Hexbin + binned attention vs ΔR by signal/background."""
        dr, at, lb = _compute_dr_attn(attn_mean, x_cpu, mask_np, labels_np)
        if len(dr) == 0:
            return
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax = axes[0]
        hb = ax.hexbin(
            dr,
            at,
            gridsize=50,
            cmap="viridis",
            extent=[0, 2, 0, np.percentile(at, 99) if len(at) else 1],
            mincnt=1,
        )
        ax.set_xlabel(r"$\Delta R$")
        ax.set_ylabel("Attention Weight")
        ax.set_title("Attention Weight vs Angular Distance")
        plt.colorbar(hb, ax=ax, label="Count")
        ax = axes[1]
        sm = lb == 1
        bins = np.linspace(0, 2, 20)
        smn, bmn = [], []
        for j in range(len(bins) - 1):
            bm = (dr >= bins[j]) & (dr < bins[j + 1])
            smn.append(np.mean(at[bm & sm]) if (bm & sm).sum() else np.nan)
            bmn.append(np.mean(at[bm & ~sm]) if (bm & ~sm).sum() else np.nan)
        ct = (bins[:-1] + bins[1:]) / 2
        ax.plot(ct, smn, "b-o", label="Signal", markersize=5)
        ax.plot(ct, bmn, "r-o", label="Background", markersize=5)
        ax.set_xlabel(r"$\Delta R$")
        ax.set_ylabel("Mean Attention Weight")
        ax.set_title("Attention vs Distance by Jet Type")
        ax.legend()
        plt.tight_layout()
        _save_fig(fig, base, "attention_vs_delta_r_by_jet_type", stem)

    def _plot_constituent_activation_vs_pt(
        act_norms, x_cpu, mask_np, labels_np, base, stem
    ):
        """Constituent activation vs pT scatter + distribution — pre-computed data."""
        cpt = x_cpu[:, :, 1].numpy()
        N = act_norms.shape[0]
        pf, af, lf = [], [], []
        for i in range(N):
            nv = int(mask_np[i].sum())
            pf.append(cpt[i, :nv])
            af.append(act_norms[i, :nv])
            lf.append(np.full(nv, labels_np[i]))
        if not pf:
            return
        pf, af, lf = np.concatenate(pf), np.concatenate(af), np.concatenate(lf)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sm = lf == 1
        ax = axes[0]
        ax.scatter(pf[~sm], af[~sm], alpha=0.1, s=2, c="red", label="Background")
        ax.scatter(pf[sm], af[sm], alpha=0.1, s=2, c="blue", label="Signal")
        ax.set_xlabel("Constituent $p_T$ [GeV]")
        ax.set_ylabel("Activation Magnitude (L2 norm)")
        ax.set_title("Constituent Activation vs $p_T$")
        ax.legend()
        if len(pf):
            ax.set_xlim(0, np.percentile(pf, 99))
        ax = axes[1]
        ax.hist(
            af[sm], bins=50, histtype="step", label="Signal", color="blue", density=True
        )
        ax.hist(
            af[~sm],
            bins=50,
            histtype="step",
            label="Background",
            color="red",
            density=True,
        )
        ax.set_xlabel("Activation Magnitude")
        ax.set_ylabel("Density")
        ax.set_title("Per-Constituent Activation Distribution")
        ax.legend()
        plt.tight_layout()
        _save_fig(fig, base, "constituent_activation_vs_pt", stem)

    # ================================================================
    # MAIN LOOP
    # ================================================================
    _dihiggs_btag_thr = btag_threshold if btag_threshold is not None else 0.5
    print(f"Di-Higgs b-tag threshold: {_dihiggs_btag_thr:.4f}")

    api = wandb.Api()
    run = api.run(WANDB_RUN_PATH)
    run_name = _safe_name(run.name or run.id or "wandb_run")
    base_out_dir = os.path.join("Updates", run_name, "plot_evolution")
    _ensure_dir(base_out_dir)

    artifacts = sorted(
        [a for a in run.logged_artifacts() if a.type == "model"], key=_artifact_sort_key
    )
    if MAX_ARTIFACTS is not None:
        artifacts = artifacts[:MAX_ARTIFACTS]

    print(f"Found {len(artifacts)} model artifacts for run {run_name}")
    for i, a in enumerate(artifacts):
        m = a.metadata or {}
        ep = m.get("epoch", "?")
        av = m.get("val_auc")
        avs = f", AUC={av:.4f}" if av else ""
        print(f"  {i+1:3d}. {a.name}:{a.version}  epoch={ep}{avs}")

    reference_rocs = _build_reference_rocs(_cuts_config)

    _last_hash = None
    _c_loader = _c_jpt = _c_jeta = _c_cmask = None

    for artifact in artifacts:
        meta = artifact.metadata or {}
        epoch_num = meta.get("epoch", "unknown")
        val_auc_meta = meta.get("val_auc")
        file_stem = _safe_name(f"epoch_{epoch_num}_{artifact.name}")
        avs = f", val_auc={val_auc_meta:.4f}" if val_auc_meta else ""
        print(f"\n{'='*60}")
        print(
            f"Processing: {artifact.name}:{artifact.version}  (epoch {epoch_num}{avs})"
        )
        print(f"{'='*60}")
        local_dir = artifact.download()

        try:
            ckpt_path = _find_checkpoint_file(local_dir)
            sd, cfg = _load_checkpoint(ckpt_path, device)
            cfg = cfg if cfg is not None else CONFIG_PART

            model = _build_model(cfg, device)
            model.load_state_dict(sd)
            del sd
            _flush()

            # Build / reuse val data
            ch = _data_cfg_hash(cfg)
            if ch != _last_hash:
                print("  Building val loader from checkpoint config...")
                _c_loader = _build_val_loader_from_cfg(cfg)
                print("  Reconstructing jet kinematics...")
                _c_jpt, _c_jeta = _reconstruct_jet_kinematics(_c_loader)
                print("  Computing cuts mask...")
                _c_cmask = _compute_cuts_mask(_c_jpt, _c_jeta, _cuts_config)
                _last_hash = ch
            else:
                print("  Reusing cached validation data.")

            print("  Running inference...")
            scores, labels, reg_pts, quantiles, inf_jpts, inf_gpts, inf_jetas = (
                _infer_outputs(model, _c_loader, device)
            )
            _has_reg = reg_pts is not None
            _has_kin = inf_jpts is not None and inf_gpts is not None

            # ── Lightweight plots ──
            print("  Generating lightweight plots...")
            _plot_model_output_distribution(scores, labels, base_out_dir, file_stem)
            _plot_roc(scores, labels, base_out_dir, file_stem)
            _plot_roc_comparison(
                scores, labels, reference_rocs, base_out_dir, file_stem
            )
            _plot_pr(scores, labels, base_out_dir, file_stem)
            _plot_calibration(scores, labels, base_out_dir, file_stem)
            _plot_significance(scores, labels, base_out_dir, file_stem)
            _plot_auc_heatmap(
                scores, labels, _c_jpt, _c_jeta, _c_cmask, base_out_dir, file_stem
            )

            # ── Regression & resolution plots ──
            if _has_reg and _has_kin:
                print("  Generating regression analysis plots...")
                _plot_regression_analysis(
                    reg_pts, labels, inf_jpts, inf_gpts, base_out_dir, file_stem
                )
                print("  Generating resolution vs pT plots...")
                _plot_resolution_vs_pt(
                    reg_pts, labels, inf_jpts, inf_gpts, base_out_dir, file_stem
                )
                if inf_jetas is not None:
                    print("  Generating resolution vs eta plots...")
                    _plot_resolution_vs_eta(
                        reg_pts,
                        labels,
                        inf_jpts,
                        inf_gpts,
                        inf_jetas,
                        base_out_dir,
                        file_stem,
                    )
            elif _has_kin:
                # No regression but we can still plot uncorrected resolution
                print("  Generating resolution plots (no regression)...")
                _plot_resolution_vs_pt(
                    None, labels, inf_jpts, inf_gpts, base_out_dir, file_stem
                )
                if inf_jetas is not None:
                    _plot_resolution_vs_eta(
                        None,
                        labels,
                        inf_jpts,
                        inf_gpts,
                        inf_jetas,
                        base_out_dir,
                        file_stem,
                    )

            # ── Di-Higgs mass reconstruction ──
            print("  Running di-Higgs mass reconstruction...")
            try:
                _plot_dihiggs_mass(
                    model,
                    _c_loader,
                    _cuts_config,
                    device,
                    collection_key=CONFIG_PART.get("training", {})
                    .get("data", {})
                    .get("use_dataset", "pf"),
                    apply_correction=_has_reg,
                    btag_thr=_dihiggs_btag_thr,
                    base=base_out_dir,
                    stem=file_stem,
                )
            except Exception as e:
                print(f"    Di-Higgs reconstruction failed: {e}")

            # ── Feature importance (small subsample, on-demand from loader) ──
            print("  Computing feature importance...")
            _plot_feature_importance(model, _c_loader, base_out_dir, file_stem)

            # ── SINGLE batched attention forward → feeds ALL 4 attn/activation plots ──
            print("  Batched attention forward pass...")
            attn_mean, act_norms, x_cpu, mask_np, labels_np = (
                _forward_with_attention_batched(
                    model, _c_loader, need_attn=True, need_acts=True
                )
            )

            print("  Generating attention & activation plots...")
            _plot_constituent_activation_maps(
                act_norms, mask_np, base_out_dir, file_stem
            )
            _plot_attention_vs_distance(
                attn_mean, x_cpu, mask_np, base_out_dir, file_stem
            )
            _plot_attention_vs_distance_by_jet_type(
                attn_mean, x_cpu, mask_np, labels_np, base_out_dir, file_stem
            )
            _plot_constituent_activation_vs_pt(
                act_norms, x_cpu, mask_np, labels_np, base_out_dir, file_stem
            )

            # Free heavy data for this artifact
            del attn_mean, act_norms, x_cpu, mask_np, labels_np
            del scores, labels, reg_pts, quantiles, inf_jpts, inf_gpts, inf_jetas
            del model
            _flush()
            print(f"  ✓ All plots saved for epoch {epoch_num}")

        finally:
            if os.path.exists(local_dir):
                shutil.rmtree(local_dir, ignore_errors=True)

    print(f"\n{'='*60}")
    print(f"Done!  Plot evolution saved to: {base_out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate plot evolution across W&B training artifacts"
    )
    parser.add_argument(
        "--config_part_path",
        type=str,
        default="config_part.json",
        help="Path to the ParT config JSON file",
    )
    parser.add_argument(
        "--btag_threshold",
        type=float,
        default=None,
        help="B-tag score threshold for di-Higgs reconstruction (default: 0.5)",
    )
    args = parser.parse_args()
    plot_evolution(
        config_part_path=args.config_part_path, btag_threshold=args.btag_threshold
    )
