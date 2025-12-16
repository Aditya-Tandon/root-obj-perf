import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import awkward as ak

from analysis_helpers import calculate_pur_eff_vs_variable

def plot_signal_background_histogram(reco_jets, is_pure_mask, bins, variable, xlabel, title):
    """
    Plots histograms of signal and background vs. a kinematic variable.
    """
    plt.figure(figsize=(10, 6))
    
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    signal_data = ak.to_numpy(ak.flatten(getattr(reco_jets[is_pure_mask], variable)))
    background_data = ak.to_numpy(ak.flatten(getattr(reco_jets[~is_pure_mask], variable)))

    h_signal, _ = np.histogram(signal_data, bins=bins)
    h_background, _ = np.histogram(background_data, bins=bins)

    plt.hist(signal_data, bins=bins, histtype="step", label='Signal', color='blue')
    plt.hist(background_data, bins=bins, histtype="step", label='Background', color='red')

    plt.xlabel(xlabel)
    plt.ylabel("Counts")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

def plot_kinematic_comparison(bins, variable, xlabel, title,
                              gen_particles,
                              objects, 
                              is_purity_plot=False,
                              fmt="o-", new_fig=True,
                              legend_postfix=""):
    """
    Plots efficiency or purity vs. a kinematic variable for objects input.
    Objects is a list of tuples: [(label, object_collection, mask)]
    """
    if new_fig:
        plt.figure(figsize=(10, 6))
    
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    for obj in objects:
        obj_label, obj_collection, obj_mask = obj
        y_values, y_errors = calculate_pur_eff_vs_variable(
            gen_particles, obj_collection, obj_mask, variable, bins, is_purity_plot=is_purity_plot
        )
        plt.errorbar(
            bin_centers, y_values, yerr=y_errors, fmt=fmt, label=f"{obj_label}{legend_postfix}"
        )
    plt.xlabel(xlabel)
    plt.ylabel("Purity" if is_purity_plot else "Efficiency")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, 1.05)
    plt.legend()
    if new_fig:
        plt.show()

def plot_roc_comparison(roc_results, working_point=None):
    """
    Plots multiple ROC curves on the same axes.
    roc_results should be a list of tuples:
    [(label, (mistag_points, eff_points, auc_score)), ...]
    """
    plt.figure(figsize=(8, 8))
    
    for label, (mistag_pts, eff_pts, auc_score, _) in roc_results:
        valid_mask = mistag_pts > 0
        plt.plot( mistag_pts[valid_mask], eff_pts[valid_mask], 'o-', 
                 label=f'{label} (AUC = {auc_score:.3f})', 
                 markersize=2)
    
    if working_point != None:
        plt.vlines(working_point, ymin=0, ymax=1, colors="black", linestyles="dashed", label=f"WP = {working_point*100}% Mistag rate")

    plt.ylabel("B-Tagging Efficiency")
    plt.xlabel("Mistag Rate")
    plt.xscale('log') 
    plt.xlim(1e-4, 1.0) 
    plt.ylim(1e-4, 1.05)
    plt.title("ROC Curve Comparison: Offline vs. L1 B-Tagging")
    plt.grid(True, linestyle='--', which='both', alpha=0.6)
    plt.legend(fontsize="small")
    plt.show()

def plot_btag_map(jets, tagger_name, pt_bins, eta_bins):
    """
    Plots a 2D heatmap of the average b-tag score vs. eta and jet pT.
    """
    print(f"Plotting 2D b-tag map for {tagger_name}...")
    
    # Flatten the jet properties into simple numpy arrays
    jet_pt = ak.to_numpy(ak.flatten(jets.vector.pt))
    jet_eta = ak.to_numpy(ak.flatten(jets.vector.eta))
    jet_btag = ak.to_numpy(ak.flatten(getattr(jets, tagger_name)))

    # --- Create the 2D Profile ---
    
    # 1. Create a 2D histogram of the SUM of b-tag scores in each bin
    #    We use the 'weights' argument to sum the b-tag scores
    h_sum_btag, xedges, yedges = np.histogram2d(
        jet_eta, jet_pt, bins=[eta_bins, pt_bins], weights=jet_btag
    )
    
    # 2. Create a 2D histogram of the COUNT of jets in each bin
    h_count_jets, _, _ = np.histogram2d(
        jet_eta, jet_pt, bins=[eta_bins, pt_bins]
    )
    
    # 3. Calculate the average score per bin
    #    Use np.divide to safely handle division by zero (for empty bins)
    h_avg_btag = np.divide(
        h_sum_btag, 
        h_count_jets, 
        out=np.zeros_like(h_sum_btag), 
        where=(h_count_jets != 0)
    )

    # --- Plotting ---
    # Use pcolormesh to plot the 2D array. Transpose (T) is needed
    # because numpy histogram and pcolormesh have different axis conventions.
    im = plt.pcolormesh(
        xedges, 
        yedges, 
        h_avg_btag.T, 
        cmap="jet",         # A common colormap for this
        norm=colors.Normalize(vmin=0.0, vmax=1.0) # Keep color scale 0-1
    )
    
    plt.ylabel(r"Corrected Jet $p_T$ [GeV]")
    plt.xlabel("Jet $\\eta$")
    plt.title(f"Average b-tag score ({tagger_name}) vs. $p_T$ and $\\eta$")
    
    # Add a color bar, which represents your z-axis
    cbar = plt.colorbar(im)
    cbar.set_label("Average b-tag score")
    
    plt.show()

def plot_cvb_map(jets, tagger_name, pt_bins, eta_bins):
    """
    Plots a 2D heatmap of the average CvB score vs. eta and jet pT.
    """
    print(f"Plotting 2D b-tag map for {tagger_name}...")
    
    # Flatten the jet properties into simple numpy arrays
    jet_pt = ak.to_numpy(ak.flatten(jets.vector.pt))
    jet_eta = ak.to_numpy(ak.flatten(jets.vector.eta))
    jet_cvb_tag = ak.to_numpy(ak.flatten(getattr(jets, tagger_name)))

    # --- Create the 2D Profile ---
    
    # 1. Create a 2D histogram of the SUM of b-tag scores in each bin
    #    We use the 'weights' argument to sum the b-tag scores
    h_sum_cvb_tag, xedges, yedges = np.histogram2d(
        jet_eta, jet_pt, bins=[eta_bins, pt_bins], weights=jet_cvb_tag
    )
    
    # 2. Create a 2D histogram of the COUNT of jets in each bin
    h_count_jets, _, _ = np.histogram2d(
        jet_eta, jet_pt, bins=[eta_bins, pt_bins]
    )
    
    # 3. Calculate the average score per bin
    #    Use np.divide to safely handle division by zero (for empty bins)
    h_avg_cvb_tag = np.divide(
        h_sum_cvb_tag, 
        h_count_jets, 
        out=np.zeros_like(h_sum_cvb_tag), 
        where=(h_count_jets != 0)
    )

    # --- Plotting ---
    # Use pcolormesh to plot the 2D array. Transpose (T) is needed
    # because numpy histogram and pcolormesh have different axis conventions.
    im = plt.pcolormesh(
        xedges, 
        yedges, 
        h_avg_cvb_tag.T, 
        cmap="jet",         # A common colormap for this
        norm=colors.Normalize(vmin=0.0, vmax=1.0) # Keep color scale 0-1
    )
    
    plt.ylabel(r"Corrected Jet $p_T$ [GeV]")
    plt.xlabel(r"Jet $\eta$")
    plt.title(f"Average CvB score ({tagger_name}) vs. $p_T$ and $\eta$")
    
    # Add a color bar, which represents your z-axis
    cbar = plt.colorbar(im)
    cbar.set_label("Average b-tag score")
    
    plt.show()

def plot_matching_criteria(gen_particles, reco_objects, CONFIG=None):
    """
    Plots a 2D heatmap of (reco_pT / gen_pT) vs. dR
    for the closest reco_object to each gen_particle.
    """
    if CONFIG is None:
        with open("hh-bbbb-obj-config.json", "r") as config_file:
            CONFIG = json.load(config_file)
    
    print("Plotting pT response vs. dR matching criteria...")
    
    # 1. Create the all-to-all deltaR matrix
    # gen_expanded shape: (events, n_gen, 1)
    # reco_expanded shape: (events, 1, n_reco)
    gen_expanded = gen_particles.vector[:, :, None]
    reco_expanded = reco_objects.vector[:, None, :]
    delta_r_matrix = gen_expanded.deltaR(reco_expanded)

    # 2. Find the index of the closest reco_object for each gen_particle
    #    shape: (events, n_gen)
    closest_reco_idx = ak.argmin(delta_r_matrix, axis=2)
    
    # 3. Get the dR value for this closest match
    #    shape: (events, n_gen)
    min_delta_r = ak.min(delta_r_matrix, axis=2)

    # 4. Get the pT of the gen particles
    #    shape: (events, n_gen)
    gen_pt = gen_particles.vector.pt
    
    # 5. Get the pT of all reco jets in each event
    #    shape: (events, n_reco)
    reco_pt = reco_objects.vector.pt
    
    # 6. Use the 'closest_reco_idx' to "pick" the pT of the matched jet
    #    This is the "fancy indexing" that matches gen to reco
    matched_reco_pt = reco_pt[closest_reco_idx]
    
    # 7. Calculate the pT ratio (reco / gen)
    #    We must use ak.where to prevent division by zero
    pt_ratio = ak.where(gen_pt > 0, matched_reco_pt / gen_pt, np.nan)

    # 8. Flatten everything into 1D numpy arrays for plotting
    flat_delta_r = ak.to_numpy(ak.flatten(min_delta_r))
    flat_pt_ratio = ak.to_numpy(ak.flatten(pt_ratio))
    
    # 9. Remove any invalid 'nan' values
    valid_mask = ~np.isnan(flat_pt_ratio)
    flat_delta_r = flat_delta_r[valid_mask]
    flat_pt_ratio = flat_pt_ratio[valid_mask]

    # --- Plotting ---
    plt.figure(figsize=(10, 8))
    
    # We use a 2D histogram (hist2d) to create the heatmap
    plt.hist2d(
        flat_delta_r, 
        flat_pt_ratio, 
        bins=[np.linspace(0, 2, 50), np.linspace(0, 2, 50)], 
        cmap='viridis', 
        norm=colors.LogNorm()  # Use a log scale for color to see faint spots
    )
    
    # Draw a red line at dR = 0.4 to show our matching cut
    plt.axvline(x=CONFIG["matching_cone_size"], color='red', linestyle='--', label=f'Matching Cut (ΔR={CONFIG["matching_cone_size"]})')
    
    plt.xlabel("ΔR (gen b-quark, closest reco jet)")
    plt.ylabel(r"p$_T$ Response (reco p$_T$ / gen p$_T$)")
    plt.title("p$_T$ Response vs. ΔR for b-quark to Jet Matching")
    plt.legend()
    
    # Add a color bar
    cbar = plt.colorbar()
    cbar.set_label("Number of Gen b-quarks")
    
    plt.show()

def plot_attr_vs_var(events, obj_collection, attr, variable, bins_attr, bins_var, xlabel, ylabel, title, mask=None):
    """
    Plots a 2D histogram of a given attribute vs. a variable for objects in a specified collection.
    """
    print(f"Plotting {attr} vs. {variable} for {obj_collection}...")

    objs = events[obj_collection]
    attr_values = getattr(objs, attr)
    var_values = getattr(objs, variable)

    if mask is not None:
        attr_values = attr_values[mask]
        var_values = var_values[mask]

    attr_values = ak.to_numpy(ak.flatten(attr_values))
    var_values = ak.to_numpy(ak.flatten(var_values))

    plt.figure(figsize=(10, 8))
    plt.hist2d(
        var_values,
        attr_values,
        bins=[bins_var, bins_attr],
        cmap='viridis',
        norm=colors.LogNorm()
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    cbar = plt.colorbar()
    cbar.set_label("Counts")

    plt.show()

def plot_attr_vs_var_proj(events, obj_collection, attr, variable, bins_attr, bins_var, xlabel, ylabel, title, mask=None):
    """
    Plots the counts for the input collection against the variable on the x-axis adn the attribute on the y-axis.
    Accompanied by panels showing the projection of the counts onto each axis.
    """
    
    
    objs = events[obj_collection]
    attr_values = getattr(objs, attr)
    var_values = getattr(objs, variable)

    if mask is not None:
        attr_values = attr_values[mask]
        var_values = var_values[mask]

    attr_values = ak.to_numpy(ak.flatten(attr_values))
    var_values = ak.to_numpy(ak.flatten(var_values))

    # --- Plotting ---
    plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], hspace=0.05, width_ratios=[3, 1], wspace=0.05)
    
    # Top Panel: Average B-Tag Score
    ax0 = plt.subplot(gs[0, 0])
    ax0.hist2d(
        var_values, attr_values, 
        bins=[bins_var, bins_attr], 
        cmap='viridis',
        norm=colors.LogNorm()
    )
    ax0.set_ylabel(ylabel)
    ax0.set_title(title)
    ax0.set_xticklabels([]) # Hide x-labels for top plot
    ax0.set_yticks(np.linspace(min(bins_attr), max(bins_attr), 11)) # Reduce number of y-ticks for clarity
    ax0.set_xticks(np.linspace(min(bins_var), max(bins_var), 11)) # Reduce number of x-ticks for clarity
    ax0.grid(True, linestyle='--', alpha=0.6)
    
    # Bottom Panel: Projection on the x-axis 
    ax1 = plt.subplot(gs[1, 0])
    var_counts, _ = np.histogram(var_values, bins=bins_var)
    bin_centres = 0.5 * (bins_var[1:] + bins_var[:-1])
    ax1.step(bin_centres, var_counts, label="Counts", color="black")
    ax1.set_ylabel("Counts")
    ax1.set_xlabel(xlabel)
    ax1.set_xticks(np.linspace(min(bins_var), max(bins_var), 11)) 
    ax1.set_yticks(np.linspace(0, max(var_counts), 3)) 
    # ax1.set_yscale("log") # Counts often span orders of magnitude
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_xlim(min(bins_var), max(bins_var))
    ax1.legend(fontsize='small')
    
    # Side Panel: Projection on the y-axis
    ax2 = plt.subplot(gs[0, 1])
    attr_counts, _ = np.histogram(attr_values, bins=bins_attr)
    bin_centres = 0.5 * (bins_attr[1:] + bins_attr[:-1])
    ax2.step(attr_counts, bin_centres, label="Counts", color='black')
    ax2.set_xlabel("Counts")
    # ax2.set_yscale("log") # Counts often span orders of magnitude
    ax2.set_yticklabels([]) # Hide y-labels for side plot
    ax2.set_xticks(np.linspace(0, max(attr_counts), 2)) # Reduce number of x-ticks for clarity
    ax2.set_yticks(np.linspace(min(bins_attr), max(bins_attr), 11)) # Reduce number of y-ticks for clarity
    ax2.legend(fontsize='small')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_ylim(min(bins_attr), max(bins_attr))
    cbar = plt.colorbar(ax0.collections[0], ax=ax2, pad=0.15)
    cbar.set_label("Counts")
    
    plt.show()

def plot_avg_attr_vs_var(reco_jets, is_signal_mask, attr, variable, bins, xlabel, ylabel, title):
    """
    Plots the average of a given attribute vs. a variable for signal and background jets.
    """
    attr_signal_flattened = ak.to_numpy(ak.flatten(getattr(reco_jets[is_signal_mask], attr)))
    variable_signal_flattened = ak.to_numpy(ak.flatten(getattr(reco_jets[is_signal_mask], variable)))
    attr_background_flattened = ak.to_numpy(ak.flatten(getattr(reco_jets[~is_signal_mask], attr)))
    variable_background_flattened = ak.to_numpy(ak.flatten(getattr(reco_jets[~is_signal_mask], variable)))
    avg_signal_attr_per_var_bin = []
    std_signal_attr_per_var_bin = []
    avg_background_attr_per_var_bin = []
    std_background_attr_per_var_bin = []
    for i in range(len(bins)-1):
        bin_mask_signal = (variable_signal_flattened >= bins[i]) & (variable_signal_flattened < bins[i+1])
        bin_mask_background = (variable_background_flattened >= bins[i]) & (variable_background_flattened < bins[i+1])
        if np.sum(bin_mask_signal) > 0:
            avg_signal_attr_per_var_bin.append(np.mean(attr_signal_flattened[bin_mask_signal]))
            std_signal_attr_per_var_bin.append(np.std(attr_signal_flattened[bin_mask_signal]) / np.sqrt(np.sum(bin_mask_signal)))
        else:
            avg_signal_attr_per_var_bin.append(0)
            std_signal_attr_per_var_bin.append(0)
        if np.sum(bin_mask_background) > 0:
            avg_background_attr_per_var_bin.append(np.mean(attr_background_flattened[bin_mask_background]))
            std_background_attr_per_var_bin.append(np.std(attr_background_flattened[bin_mask_background]) / np.sqrt(np.sum(bin_mask_background)))
        else:
            avg_background_attr_per_var_bin.append(0)
            std_background_attr_per_var_bin.append(0)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    plt.errorbar(bin_centers, avg_signal_attr_per_var_bin, yerr=std_signal_attr_per_var_bin, marker='o', label="Signal Avg")
    plt.errorbar(bin_centers, avg_background_attr_per_var_bin, yerr=std_background_attr_per_var_bin, marker='o', label="Background Avg")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

    return bin_centers, avg_signal_attr_per_var_bin, std_signal_attr_per_var_bin, avg_background_attr_per_var_bin, std_background_attr_per_var_bin
  
def plot_resolution_vs_var(gen_var, resolution_values, bins, y_label, x_label, title):
    """
    Bins the data by Gen pT and calculates the Mean and Width (StdDev) 
    of the resolution in each bin.
    """
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    
    means = []
    widths = []
    errors = [] # Error on the width calculation
    
    # Digitize: Find which bin each event belongs to
    # indices 1 to len(bins)-1 are valid bins
    bin_indices = np.digitize(gen_var, bins)
    
    for i in range(1, len(bins)):
        # Select data for this specific bin
        data_in_bin = resolution_values[bin_indices == i]
        
        if len(data_in_bin) > 10: # Require minimum stats
            # Calculate Mean (Bias)
            mu = np.mean(data_in_bin)
            
            # Calculate Width (Resolution)
            # Standard Deviation is simple, but IQR/2 is more robust against tails
            sigma = np.std(data_in_bin) 
            
            # Error on std dev estimate approx: sigma / sqrt(2N)
            err = sigma / np.sqrt(2 * len(data_in_bin))
            
            means.append(mu)
            widths.append(sigma)
            errors.append(err)
        else:
            means.append(np.nan)
            widths.append(np.nan)
            errors.append(0)

    # --- Plotting ---
    plt.figure(figsize=(10, 8))
    
    # Top Panel: Resolution (Width)
    plt.subplot(2, 1, 1)
    plt.errorbar(bin_centers, widths, yerr=errors, fmt='o-', capsize=5, label='Resolution ($\sigma$)')
    plt.ylabel(f"{y_label} Resolution")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Bottom Panel: Scale/Bias (Mean)
    plt.subplot(2, 1, 2)
    plt.errorbar(bin_centers, means, fmt='s--', color='red', capsize=5, label='Scale (Mean)')
    plt.axhline(0, color='black', linestyle='-', linewidth=1)
    plt.xlabel(x_label)
    plt.ylabel(f"{y_label} Scale (Bias)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
