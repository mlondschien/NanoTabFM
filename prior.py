import torch
from torch.nn import functional as F

ACTIVATIONS = (lambda x: x, torch.tanh, torch.sin, torch.abs, torch.square, F.softplus)


@torch.no_grad()
def make_batch(
    batch_size=32,
    num_context=100,
    num_queries=28,
    num_features=10,
    device="cpu",
):
    """Sample binary classification tasks from random structural causal models.

    Based on ModdedNanoPrior from modded-nanotabpfn. Each table is a random DAG with
    num_features + 1 nodes. One random non-root node is the target, split at a random quantile;
    the other nodes are the features, in random column order. Some features are observed as
    integer-coded categories.
    """
    num_rows = num_context + num_queries
    num_nodes = num_features + 1
    tables = torch.arange(batch_size, device=device)

    # Each node draws three candidate parents among earlier nodes. With probability 0.5, a
    # candidate that has parents of its own is replaced by one of them.
    parents = torch.zeros(batch_size, num_nodes, num_nodes, dtype=torch.bool, device=device)
    for node in range(1, num_nodes):
        for _ in range(3):
            candidates = torch.randint(node, (batch_size,), device=device)
            candidate_parents = parents[tables, candidates]
            parent_scores = torch.rand(batch_size, num_nodes, device=device) * candidate_parents
            redirect = candidate_parents.any(dim=1) & (torch.rand(batch_size, device=device) < 0.5)
            candidates = torch.where(redirect, parent_scores.argmax(dim=1), candidates)
            parents[tables, node, candidates] = True

    # Each node applies a random activation to a random linear combination of its parents,
    # adds noise with a per-table scale between 0.01 and 10, and is standardized.
    weights = torch.randn(batch_size, num_nodes, num_nodes, device=device) * parents
    activations = torch.randint(len(ACTIVATIONS), (batch_size, num_nodes), device=device)
    noise_scales = 10 ** (1 - 3 * torch.rand(batch_size, 1, device=device))
    values = torch.randn(batch_size, num_rows, num_nodes, device=device)
    for node in range(1, num_nodes):
        inputs = values @ weights[:, node].unsqueeze(-1)
        outputs = torch.cat([activation(inputs) for activation in ACTIVATIONS], dim=-1)
        outputs = outputs[tables, :, activations[:, node]] + noise_scales * values[:, :, node]
        mean, std = outputs.mean(dim=1, keepdim=True), outputs.std(dim=1, keepdim=True)
        values[:, :, node] = (outputs - mean) / (std + 1e-6)

    # Split the target at a random quantile, so that 5-95% of the rows are positive.
    targets = torch.randint(1, num_nodes, (batch_size,), device=device)
    scores = values[tables, :, targets]
    quantiles = 0.05 + 0.9 * torch.rand(batch_size, 1, device=device)
    thresholds = scores.sort(dim=1).values.gather(1, (quantiles * (num_rows - 1)).long())
    labels = (scores > thresholds).long()

    # Sorting random keys shuffles the columns; the target's key sorts it last, out of the features.
    keys = torch.rand(batch_size, num_nodes, device=device)
    keys[tables, targets] = 2
    feature_nodes = keys.argsort(dim=1)[:, :num_features]
    features = values.gather(2, feature_nodes.unsqueeze(1).expand(-1, num_rows, -1))

    # Observe a random fraction of the columns as categories with 2-10 levels: cut each column
    # at the values of random rows, then code the levels 0, 1, ... in order or shuffled.
    max_levels = 10
    level_indices = torch.arange(max_levels, device=device).view(1, -1, 1)
    cut_rows = torch.randint(num_rows, (batch_size, max_levels - 1, num_features), device=device)
    num_cuts = torch.randint(1, max_levels, (batch_size, 1, num_features), device=device)
    used_cuts = (level_indices[:, :-1] < num_cuts).unsqueeze(1)
    above_cuts = features.unsqueeze(2) > features.gather(1, cut_rows).unsqueeze(1)
    levels = (above_cuts & used_cuts).sum(dim=2)
    ordered = torch.rand(batch_size, 1, num_features, device=device) < 0.5
    shuffled_keys = torch.rand(batch_size, max_levels, num_features, device=device)
    level_keys = torch.where(ordered, level_indices.float(), shuffled_keys)
    level_keys = level_keys + max_levels * (level_indices > num_cuts)  # unused levels sort last
    codes = level_keys.argsort(dim=1).argsort(dim=1).gather(1, levels).float()
    fraction = torch.rand(batch_size, 1, 1, device=device)
    categorical = torch.rand(batch_size, 1, num_features, device=device) < fraction
    features = torch.where(categorical, codes, features)
    return (
        features[:, :num_context],
        labels[:, :num_context],
        features[:, num_context:],
        labels[:, num_context:],
    )
