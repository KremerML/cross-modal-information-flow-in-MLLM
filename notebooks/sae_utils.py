"""
sae_utils.py — Helper utilities for the GPT-2 SAE seminar notebook.

Import with:
    import sys; sys.path.insert(0, ".")
    from sae_utils import collect_activations, make_activation_dataloader, ...
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display, HTML


# ── Activation collection ─────────────────────────────────────────────────────

def collect_activations(
    model,
    dataset,
    hook_name: str = "blocks.6.hook_mlp_out",
    n_sequences: int = 2000,
    seq_len: int = 128,
    batch_size: int = 16,
    device: str = "cuda",
    return_token_ids: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    """
    Run ``n_sequences`` text documents through ``model`` and collect activations
    at ``hook_name`` for every token position.

    Parameters
    ----------
    model          : TransformerLens HookedTransformer
    dataset        : HuggingFace dataset with a "text" column
    hook_name      : TransformerLens hook point name
    n_sequences    : how many documents to process
    seq_len        : truncate each document to this many tokens
    batch_size     : forward-pass batch size
    device         : "cuda" or "cpu"
    return_token_ids : if True, also return per-sequence token id tensors

    Returns
    -------
    acts       : FloatTensor (n_tokens, d_model)
    token_ids  : list of (seq_len,) LongTensors  [only if return_token_ids=True]
    """
    from tqdm.auto import tqdm

    all_acts: list[torch.Tensor] = []
    all_token_ids: list[torch.Tensor] = []
    n_done = 0

    model.eval()
    pbar = tqdm(total=n_sequences, desc="Collecting activations")

    for i in range(0, len(dataset), batch_size):
        if n_done >= n_sequences:
            break
        texts  = dataset[i : i + batch_size]["text"]
        tokens = model.to_tokens(texts, truncate=True)
        tokens = tokens[:, :seq_len].to(device)

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)

        acts = cache[hook_name]           # (batch, seq_len, d_model)
        all_acts.append(acts.reshape(-1, acts.shape[-1]).cpu())

        if return_token_ids:
            for j in range(tokens.shape[0]):
                all_token_ids.append(tokens[j].cpu())

        n_done += tokens.shape[0]
        pbar.update(tokens.shape[0])

    pbar.close()
    acts_tensor = torch.cat(all_acts, dim=0)
    token_ids   = all_token_ids if return_token_ids else None
    return acts_tensor, token_ids


# ── DataLoader construction ──────────────────────────────────────────────────

def make_activation_dataloader(
    acts: torch.Tensor,
    batch_size: int = 4096,
    subtract_mean: bool = True,
    shuffle: bool = True,
) -> tuple[DataLoader, torch.Tensor]:
    """
    Build a DataLoader from a flat activation tensor.

    Parameters
    ----------
    acts          : (n_tokens, d_model) FloatTensor
    batch_size    : DataLoader batch size
    subtract_mean : if True, subtract the column-wise mean (geometric centering)
    shuffle       : whether to shuffle

    Returns
    -------
    loader : DataLoader yielding (batch, d_model) batches
    mean   : (d_model,) mean vector used for centering (zeros if subtract_mean=False)
    """
    mean = acts.mean(dim=0) if subtract_mean else torch.zeros(acts.shape[-1])
    acts_c = acts - mean
    loader = DataLoader(TensorDataset(acts_c), batch_size=batch_size,
                        shuffle=shuffle, drop_last=True, num_workers=0)
    return loader, mean


# ── Feature retrieval ─────────────────────────────────────────────────────────

def get_top_activating_examples(
    sae,
    acts: torch.Tensor,
    feature_idx: int,
    k: int = 10,
    batch_size: int = 4096,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Find the ``k`` token indices that maximally activate ``feature_idx``.

    Parameters
    ----------
    sae         : SparseAutoencoder (already sent to device)
    acts        : (n_tokens, d_model) — *centered* activation tensor (CPU)
    feature_idx : which feature to query
    k           : how many examples to return
    batch_size  : mini-batch size for encoding
    device      : device string

    Returns
    -------
    top_indices : LongTensor (k,) — indices into ``acts``
    top_values  : FloatTensor (k,) — activation values
    """
    sae.eval()
    all_vals: list[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, acts.shape[0], batch_size):
            xb = acts[start : start + batch_size].to(device)
            z  = sae.encode(xb)
            all_vals.append(z[:, feature_idx].cpu())

    all_vals_cat = torch.cat(all_vals, dim=0)
    result = torch.topk(all_vals_cat, k=k)
    return result.indices, result.values


def get_feature_frequencies(
    sae,
    acts: torch.Tensor,
    batch_size: int = 4096,
    device: str = "cuda",
) -> np.ndarray:
    """
    Compute the fraction of tokens where each feature activates (freq > 0).

    Returns
    -------
    feature_freqs : np.ndarray (d_dict,)
    """
    sae.eval()
    all_bits: list[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, acts.shape[0], batch_size):
            xb = acts[start : start + batch_size].to(device)
            z  = sae.encode(xb)
            all_bits.append((z > 0).float().cpu())

    bits = torch.cat(all_bits, dim=0)
    return bits.mean(dim=0).numpy()


# ── Context display ──────────────────────────────────────────────────────────

def show_top_activating_contexts(
    feature_idx: int,
    z_flat: torch.Tensor,
    index_map: list[tuple[int, int]],
    token_ids: list[torch.Tensor],
    model,
    k: int = 5,
    context_window: int = 8,
) -> None:
    """
    Display the top-k text contexts where ``feature_idx`` fires most strongly.

    Parameters
    ----------
    feature_idx    : which SAE feature to inspect
    z_flat         : (n_tokens, d_dict) SAE activations for the analysis pool
    index_map      : list mapping flat token index → (seq_idx, pos_in_seq)
    token_ids      : list of per-sequence token id tensors
    model          : TransformerLens model (for decoding tokens)
    k              : number of examples to show
    context_window : tokens of context on each side of the highlighted token
    """
    scores = z_flat[:, feature_idx]
    top_k  = torch.topk(scores, k=k)
    freq   = float((z_flat[:, feature_idx] > 0).float().mean())

    html = [
        f"<b>Feature {feature_idx}</b> &nbsp; "
        f"<small>freq={freq:.4f}</small><br><br>"
    ]

    for rank, flat_idx in enumerate(top_k.indices):
        flat_idx = int(flat_idx)
        val      = float(top_k.values[rank])
        if flat_idx >= len(index_map):
            continue
        seq_idx, pos = index_map[flat_idx]
        tids = token_ids[seq_idx]

        start = max(0, pos - context_window)
        end   = min(len(tids), pos + context_window + 1)

        tok_parts = []
        for tok_pos in range(start, end):
            tok_str = model.to_single_str_token(tids[tok_pos].item())
            tok_esc = (tok_str.replace("&", "&amp;")
                               .replace("<", "&lt;")
                               .replace(">", "&gt;"))
            if tok_pos == pos:
                tok_parts.append(
                    f'<mark style="background:#ffeb3b;padding:1px 3px;'
                    f'border-radius:3px;font-weight:bold">{tok_esc}</mark>')
            else:
                tok_parts.append(f'<span style="color:#666">{tok_esc}</span>')

        html.append(
            f'<b>#{rank+1}</b> <small>(act={val:.3f})</small>: '
            f'<code>{"".join(tok_parts)}</code><br>'
        )

    display(HTML("".join(html) + "<br>"))


# ── Plotly figures ────────────────────────────────────────────────────────────

def plot_training_curves(logs: list[dict]) -> go.Figure:
    """
    Plot total loss, MSE, L1, mean L0, and dead features over training steps.

    Parameters
    ----------
    logs : list of dicts with keys 'step', 'loss', 'mse_loss', 'l1_loss',
           'mean_l0', 'dead_features'
    """
    steps    = [d["step"]          for d in logs]
    loss     = [d["loss"]          for d in logs]
    mse      = [d["mse_loss"]      for d in logs]
    l1       = [d["l1_loss"]       for d in logs]
    l0       = [d["mean_l0"]       for d in logs]
    dead     = [d["dead_features"] for d in logs]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Total Loss", "MSE Loss",
            "Mean L0 (active features / token)", "Dead Features",
        ],
    )

    for row, col, y, name, color in [
        (1, 1, loss,  "Total",  "#1f77b4"),
        (1, 2, mse,   "MSE",    "#ff7f0e"),
        (2, 1, l0,    "L0",     "#9467bd"),
        (2, 2, dead,  "Dead",   "#d62728"),
    ]:
        fig.add_trace(
            go.Scatter(x=steps, y=y, name=name,
                       line=dict(color=color, width=1.5), showlegend=False),
            row=row, col=col,
        )

    fig.update_xaxes(title_text="Step")
    fig.update_layout(height=580, title_text="SAE Training Diagnostics",
                      margin=dict(t=70))
    return fig


def plot_feature_frequency_histogram(
    feature_freqs: np.ndarray,
    d_dict: Optional[int] = None,
) -> go.Figure:
    """
    Log-scale histogram of per-feature activation frequencies.

    Parameters
    ----------
    feature_freqs : (d_dict,) array of frequencies in [0, 1]
    d_dict        : optional dict size for title annotation
    """
    n = len(feature_freqs) if d_dict is None else d_dict
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=np.log10(feature_freqs + 1e-10),
        nbinsx=80,
        marker_color="#1f77b4",
    ))
    fig.add_vline(x=np.log10(0.01), line_dash="dash", line_color="orange",
                  annotation_text="1%", annotation_position="top right")
    fig.add_vline(x=np.log10(0.1),  line_dash="dash", line_color="red",
                  annotation_text="10%", annotation_position="top right")
    fig.update_layout(
        title=f"Feature activation frequency — {n} features",
        xaxis_title="log₁₀(fraction of tokens where feature fires)",
        yaxis_title="Number of features",
        bargap=0.05, height=380,
    )
    return fig


def plot_token_feature_heatmap(
    tokens: torch.Tensor,
    feature_acts: torch.Tensor,
    model,
    top_n: int = 20,
    sentence: str = "",
) -> go.Figure:
    """
    Token × feature activation heatmap for a single input sentence.

    Parameters
    ----------
    tokens       : (seq_len,) LongTensor of token ids
    feature_acts : (seq_len, d_dict) FloatTensor of SAE activations
    model        : TransformerLens model (for decoding tokens)
    top_n        : how many top features to show on the y-axis
    sentence     : optional original sentence string for the title
    """
    max_per_feature = feature_acts.max(dim=0).values
    top_fids        = torch.topk(max_per_feature, k=top_n).indices.cpu().numpy()

    tok_strs = [
        model.to_single_str_token(t.item())
              .replace("Ġ", " ")
              .replace("Ċ", "\\n")
        for t in tokens
    ]

    z_data = feature_acts[:, top_fids].T.cpu().float().numpy()   # (top_n, seq_len)

    title = "Token × Feature heatmap"
    if sentence:
        title += f'<br><i>"{sentence[:70]}{"…" if len(sentence) > 70 else ""}"</i>'

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=tok_strs,
        y=[f"F{fid}" for fid in top_fids],
        colorscale="Viridis",
        colorbar_title="Activation",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Token",
        yaxis_title="SAE Feature",
        height=560, xaxis_tickangle=-45, margin=dict(t=90),
    )
    return fig


def plot_decoder_umap(
    W_dec: np.ndarray,
    feature_freqs: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> go.Figure:
    """
    UMAP projection of decoder weight vectors, coloured by log-frequency.

    Parameters
    ----------
    W_dec         : (d_dict, d_model) decoder weight matrix
    feature_freqs : (d_dict,) activation frequency per feature
    """
    try:
        from umap import UMAP
    except ImportError:
        raise ImportError("Install umap-learn:  pip install umap-learn")

    reducer   = UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                     random_state=random_state, verbose=False)
    embedding = reducer.fit_transform(W_dec)   # (d_dict, 2)
    log_freq  = np.log10(feature_freqs + 1e-10)

    fig = go.Figure(data=go.Scatter(
        x=embedding[:, 0], y=embedding[:, 1],
        mode="markers",
        marker=dict(
            size=3,
            color=log_freq,
            colorscale="Viridis",
            colorbar=dict(title="log₁₀(freq)"),
            opacity=0.6,
        ),
        text=[f"Feature {i}<br>freq={feature_freqs[i]:.4f}"
              for i in range(len(feature_freqs))],
        hoverinfo="text",
    ))
    fig.update_layout(
        title="UMAP of SAE decoder directions (coloured by activation frequency)",
        xaxis_title="UMAP 1", yaxis_title="UMAP 2",
        height=620, margin=dict(t=60),
    )
    return fig


def feature_dashboard(
    feature_idx: int,
    z_flat: torch.Tensor,
    sae,
    model,
) -> go.Figure:
    """
    Side-by-side: activation value histogram + top vocabulary projections.

    Parameters
    ----------
    feature_idx : feature to inspect
    z_flat      : (n_tokens, d_dict) SAE activations for the analysis pool
    sae         : SparseAutoencoder instance
    model       : TransformerLens model (needs W_U attribute)
    """
    act_vals    = z_flat[:, feature_idx].numpy()
    active_vals = act_vals[act_vals > 0]
    freq        = float((act_vals > 0).mean())

    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        f"Feature {feature_idx} — activation values (active tokens only)",
        f"Feature {feature_idx} — top vocabulary projections",
    ])

    # Activation histogram
    fig.add_trace(go.Histogram(x=active_vals, nbinsx=40,
                               marker_color="#1f77b4", name="Activations"),
                  row=1, col=1)

    # Decoder direction → vocabulary logits
    W_dec_vec = sae.W_dec.data[feature_idx].cpu().float()
    W_U       = model.W_U.data.cpu().float()         # (d_model, vocab)
    logits    = W_dec_vec @ W_U                       # (vocab,)

    top_vals, top_idx = torch.topk(logits, k=15)
    top_tokens = [model.to_single_str_token(i.item()).strip() for i in top_idx]

    fig.add_trace(go.Bar(
        x=top_vals.numpy()[::-1], y=top_tokens[::-1],
        orientation="h", marker_color="#ff7f0e", name="Vocab projection"),
        row=1, col=2)

    mean_act = float(active_vals.mean()) if len(active_vals) > 0 else 0

    fig.update_layout(
        height=340,
        title_text=(f"Feature {feature_idx}  |  freq={freq:.4f}  |  "
                    f"mean_act={mean_act:.3f}  |  n_active={len(active_vals):,}"),
        showlegend=False, margin=dict(t=65),
    )
    return fig


def plot_decoder_cosine_heatmap(
    W_dec: torch.Tensor,
    feature_ids: np.ndarray,
) -> go.Figure:
    """
    Ward-clustered cosine similarity heatmap for a subset of decoder directions.

    Parameters
    ----------
    W_dec       : (d_dict, d_model) decoder weight tensor (on any device)
    feature_ids : 1-D array of feature indices to include
    """
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance  import pdist

    W_sub  = W_dec[feature_ids].cpu().float().numpy()
    norms  = np.linalg.norm(W_sub, axis=1, keepdims=True)
    W_norm = W_sub / (norms + 1e-8)

    cos_sim = W_norm @ W_norm.T
    dists   = pdist(W_norm, metric="cosine")
    Z       = linkage(dists, method="ward")
    order   = leaves_list(Z)

    cos_ord = cos_sim[order][:, order]
    labels  = [f"F{feature_ids[i]}" for i in order]

    fig = go.Figure(data=go.Heatmap(
        z=cos_ord, x=labels, y=labels,
        colorscale="RdBu", zmin=-1, zmax=1,
        colorbar_title="cos sim",
    ))
    fig.update_layout(
        title=f"Decoder direction cosine similarity — {len(feature_ids)} features (ward-clustered)",
        height=620, width=680, xaxis_tickangle=-45,
    )
    return fig
