import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """
    A Text-based Convolutional Neural Network (TextCNN) for classification.

    This architecture uses an embedding layer followed by parallel convolutional
    layers with different kernel sizes. The outputs are passed through a max-over-time
    pooling layer, concatenated, and fed into a final fully-connected layer.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        filter_sizes=None,
        num_filters: int = 100,
        dropout: float = 0.5,
        padding_idx: int = 0,
    ):
        """
        Args:
            vocab_size: The size of the vocabulary.
            embed_dim: The dimensionality of the word embeddings.
            num_classes: The number of output classes.
            filter_sizes: A list of kernel sizes for the convolutional layers.
            num_filters: The number of filters for each convolutional layer.
            dropout: The dropout rate.
            padding_idx: The index of the padding token in the vocabulary.
        """
        super().__init__()
        if filter_sizes is None:
            filter_sizes = [3, 4, 5]
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k) for k in filter_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the TextCNN.

        Args:
            x: Input tensor of shape (batch, seq_len).

        Returns:
            Output logits tensor of shape (batch, num_classes).
        """
        # x: (batch, seq_len)
        emb = self.embedding(x)  # (batch, seq_len, embed_dim)
        emb = emb.transpose(1, 2)  # (batch, embed_dim, seq_len)
        conv_outs = []
        for conv in self.convs:
            h = conv(emb)  # (batch, num_filters, seq_len - k + 1)
            h = F.relu(h)
            # Max-over-time pooling
            pooled = F.max_pool1d(h, kernel_size=h.size(2)).squeeze(2)  # (batch, num_filters)
            conv_outs.append(pooled)
        h_cat = torch.cat(conv_outs, dim=1)  # (batch, num_filters * len(filter_sizes))
        h_drop = self.dropout(h_cat)
        logits = self.fc(h_drop)
        return logits

