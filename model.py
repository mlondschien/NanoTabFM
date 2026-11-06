import torch
from torch import nn


class FourierFeatureEncoder(nn.Module):
    def __init__(self, embedding_size, num_frequencies=16):
        super().__init__()
        self.frequencies = nn.Parameter(0.1 * torch.randn(num_frequencies))
        # Each cell encodes 3 grouped features (see NanoTabFM.forward).
        self.projection = nn.Linear(3 * 2 * num_frequencies, embedding_size)

    def forward(self, features):
        angles = 2 * torch.pi * features.unsqueeze(-1) * self.frequencies
        fourier_features = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return self.projection(fourier_features.flatten(-2))


class AttentionBlock(nn.Module):
    def __init__(self, embedding_size, num_heads, blowup_factor=2):
        super().__init__()
        self.attention = nn.MultiheadAttention(embedding_size, num_heads, batch_first=True)
        self.attention_norm = nn.RMSNorm(embedding_size)
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_size, blowup_factor * embedding_size),
            nn.GELU(),
            nn.Linear(blowup_factor * embedding_size, embedding_size),
        )
        self.feedforward_norm = nn.RMSNorm(embedding_size)

    def forward(self, queries, context):
        normalized_queries = self.attention_norm(queries)
        normalized_context = self.attention_norm(context)
        attention_output, _ = self.attention(
            normalized_queries,
            normalized_context,
            normalized_context,
            need_weights=False,
        )
        queries = queries + attention_output
        return queries + self.feedforward(self.feedforward_norm(queries))


class InducedSelfAttentionBlock(nn.Module):
    """Two attention passes through a small set of learned inducing tokens."""

    def __init__(self, embedding_size, num_heads, num_inducing_tokens=16, blowup_factor=2):
        super().__init__()
        self.inducing_tokens = nn.Parameter(0.02 * torch.randn(num_inducing_tokens, embedding_size))
        self.summarize = AttentionBlock(embedding_size, num_heads, blowup_factor)
        self.broadcast = AttentionBlock(embedding_size, num_heads, blowup_factor)

    def forward(self, inputs, context):
        # Gather the context into the inducing tokens, then let every input read them.
        inducing_tokens = self.inducing_tokens.expand(inputs.shape[0], -1, -1)
        return self.broadcast(inputs, self.summarize(inducing_tokens, context))


class NanoTabFM(nn.Module):
    """In-context binary classifier for tables with 1 to max_features features.

    1. Standardize features.
    2. Apply learned Fourier encoding to all entries of X_train and X_test. The cell of
       feature j encodes features j, j + 1, and j + 3 (modulo F) together.
       Apply first label embedding to y_train and add this to the encoded cells.
    3. Apply induced self-attention (ISAB) to each column of the encoded feature matrix.
       This first "summarizes" the column's distribution with the inducing tokens, then
       relates each cell to the summarized column information.
    4. Apply self-attention to each row of the encoded feature matrix.
    5. Another ISAB.
    6. Compress rows into a single embedding per row by having a learned summary token
       attend to the cells of its row.
    7. Transformer.
    8. Prediction head.

    For defaults (embedding_size=96, num_heads=4, num_inducing_tokens=16,
    num_context_layers=3, max_features=10, num_frequencies=16), approx. parameter counts
    are 1. 10k, 2. 151k, 3. 75k, 4. 151k, 5. 75k, 6. 224k, 7. 19k, total 705k.
    """

    def __init__(
        self,
        embedding_size=96,
        num_heads=4,
        num_inducing_tokens=16,
        num_context_layers=3,
        max_features=10,
        num_frequencies=16,
    ):
        super().__init__()
        self.feature_encoder = FourierFeatureEncoder(embedding_size, num_frequencies)
        self.first_label_embedding = nn.Embedding(2, embedding_size)
        self.second_label_embedding = nn.Embedding(2, embedding_size)
        self.feature_position_embeddings = nn.Parameter(
            0.02 * torch.randn(max_features, embedding_size)
        )
        self.first_column_attention = InducedSelfAttentionBlock(
            embedding_size, num_heads, num_inducing_tokens
        )
        self.row_attention = AttentionBlock(embedding_size, num_heads)
        self.second_column_attention = InducedSelfAttentionBlock(
            embedding_size, num_heads, num_inducing_tokens
        )
        self.summary_token = nn.Parameter(0.02 * torch.randn(1, embedding_size))
        self.summarize_row = AttentionBlock(embedding_size, num_heads)
        self.row_norm = nn.LayerNorm(embedding_size)
        self.context_layers = nn.ModuleList(
            [AttentionBlock(embedding_size, num_heads) for _ in range(num_context_layers)]
        )
        self.predictor = nn.Sequential(
            nn.LayerNorm(embedding_size),
            nn.Linear(embedding_size, 2 * embedding_size),
            nn.GELU(),
            nn.Linear(2 * embedding_size, 2),
        )

    def forward(self, X_train, y_train, X_test):
        """
        Forward pass.

        Here B: batch size, C: number of context (training) rows, Q: number of query (test) rows,
        F: number of columns (features).

        Parameters:
        -----------
        X_train: array-like, (B, C, F)
            Training features.
        y_train: array-like, (B, C)
            Training labels.
        X_test: array-like, (B, Q, F)
            Test features.

        Returns:
        --------
        logits: torch.Tensor, (B, Q, 2)
            Predicted logits for the query (test) rows.
        """
        batch_size, num_context_rows, num_features = X_train.shape
        X = torch.cat([X_train, X_test], dim=1)  # [B, N, F]
        num_rows = X.shape[1]

        # Standardize inputs
        column_mean = X_train.mean(dim=1, keepdim=True)  # [B, 1, F]
        column_scale = X_train.std(dim=1, keepdim=True, unbiased=False)  # [B, 1, F]
        X = (X - column_mean) / column_scale.clamp_min(1e-5)  # [B, N, F]
        X = X.clamp(-10, 10)  # [B, N, F]

        # Encode X and y and sum. The feature-position embeddings tell row attention which
        # column each cell belongs to.
        # Jointly encode feature-groups (j, j+1, j+3) modulo F, as in TabICLv2.
        groups = torch.stack([X.roll(-offset, dims=2) for offset in (0, 1, 3)], dim=-1)
        cells = self.feature_encoder(groups)  # [B, N, F, E]
        cells = cells + self.feature_position_embeddings[:num_features]  # [B, N, F, E]
        cell_labels = self.first_label_embedding(y_train).unsqueeze(2)  # [B, C, 1, E]
        cells_train = cells[:, :num_context_rows] + cell_labels  # [B, C, F, E]
        cells = torch.cat([cells_train, cells[:, num_context_rows:]], dim=1)  # [B, N, F, E]

        # Column attention is an ISAB. It uses the train rows to build a "summary" of each
        # column's distribution and then relates all train+test rows to this summary.
        # We pass subset :num_context_rows for the context so the train rows don't attend to the
        # test rows.
        columns = cells.transpose(1, 2).flatten(0, 1)  # [B * F, N, E]
        columns = self.first_column_attention(columns, columns[:, :num_context_rows])
        cells = columns.reshape(batch_size, num_features, num_rows, -1)  # [B, F, N, E]

        # Row attention: within each row, every cell attends to all cells of that row.
        row_cells = cells.transpose(1, 2).flatten(0, 1)  # [B * N, F, E]
        row_cells = self.row_attention(row_cells, row_cells)  # [B * N, F, E]
        cells = row_cells.reshape(batch_size, num_rows, num_features, -1)  # [B, N, F, E]

        # Second column attention.
        columns = cells.transpose(1, 2).flatten(0, 1)  # [B * F, N, E]
        columns = self.second_column_attention(columns, columns[:, :num_context_rows])
        cells = columns.reshape(batch_size, num_features, num_rows, -1)  # [B, F, N, E]

        # Next, we compress each row's features representation into a single summary token.
        # Here, for each row, the summary token attends to all feature representations of that row
        # (summary token as the query and feature representations as the keys and values).
        row_cells = cells.transpose(1, 2).flatten(0, 1)  # [B * N, F, E]
        summary_token = self.summary_token.expand(batch_size * num_rows, -1, -1)  # [B * N, 1, E]
        row_summaries = self.summarize_row(summary_token, row_cells)  # [B * N, 1, E]
        rows = self.row_norm(row_summaries).reshape(batch_size, num_rows, -1)  # [B, N, E]

        row_labels = self.second_label_embedding(y_train)  # [B, C, E]
        context_rows = rows[:, :num_context_rows] + row_labels  # [B, C, E]
        rows = torch.cat([context_rows, rows[:, num_context_rows:]], dim=1)  # [B, N, E]
        for layer in self.context_layers:
            rows = layer(rows, rows[:, :num_context_rows])  # [B, N, E]
        return self.predictor(rows[:, num_context_rows:])  # [B, Q, 2]
