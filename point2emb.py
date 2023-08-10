import torch
from torch import Tensor, nn
import torch.nn.functional as F

POSITION_INPUT_DIMS = 3


class SinusoidalPositionalEmbeddings(nn.Module):
    def __init__(self, num_embs: int, learned: bool = False) -> None:
        super().__init__()

        self.num_embs = num_embs
        self.freqs = nn.Parameter(2.0 ** torch.arange(0, num_embs), requires_grad=learned)

    def out_dims(self) -> int:
        return 2 * self.num_embs + 1

    def forward(self, pos: Tensor) -> Tensor:
        """Applies sinusoidal embeddings to input positions.

        Args:
            pos: Tensor with shape (..., N)

        Returns:
            Embedded positions, with shape (..., N * (num_embs * 2) + 1)
        """

        pos = pos.unsqueeze(-1)
        freq_pos = self.freqs * pos
        sin_embs, cos_embs = torch.sin(freq_pos), torch.cos(freq_pos)
        return torch.cat([pos, sin_embs, cos_embs], dim=-1).flatten(-2)


def init_weights(layer: nn.Module) -> None:
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


#num_layers: int = ml.conf_field(MISSING, help="Number of MLP layers for encoding position")
#hidden_dims: int = ml.conf_field(MISSING, help="Number of hidden layer dimensions")
#num_pos_embs: int = ml.conf_field(6, help="Number of positional embedding frequencies")
#output_dims: int = ml.conf_field(MISSING, help="Number of output dimensions")
#norm: str = ml.conf_field("no_norm", help="Per-layer normalization to apply")
#act: str = ml.conf_field("relu", help="Activation function to use")

def get_norm_linear(norm = 'no_norm', dim = 0):
    if norm == 'no_norm':
        return nn.Identity()
    if norm == 'layer':
        return nn.LayerNorm(dim)
    if norm == 'batch':
        return nn.BactchNorm1d(dim)
    
def get_activation(act = 'relu'):
    if act == 'no_act':
        return nn.Identity()
    if act == 'relu':
        return nn.ReLU()
    if act == 'softmax':
        return nn.Softmax(dim = -1)

class Point2EmbModel(nn.Module):
    def __init__(self, 
                 num_layers: int, 
                 hidden_dims: int, 
                 image_rep_size: int, 
                 text_rep_size: int, 
                 num_pos_embs = 6,
                 norm = 'no_norm', 
                 act = 'relu', ) -> None:
        super().__init__()

        assert num_layers > 0

        # Gets the position embedding MLP.
        self.image_rep_size = image_rep_size
        self.text_rep_size = text_rep_size
        self.pos_embs = SinusoidalPositionalEmbeddings(num_pos_embs)
        pos_mlp_in_dims = POSITION_INPUT_DIMS * self.pos_embs.out_dims()
        output_dims = image_rep_size + text_rep_size
        layers: list[nn.Module] = []
        self.temperature = nn.Parameter(torch.log(torch.tensor(1.0 / 0.07)))
        layers += [
            nn.Sequential(
                nn.Linear(
                    pos_mlp_in_dims if i == 0 else hidden_dims,
                    output_dims if i == num_layers - 1 else hidden_dims,
                ),
                get_norm_linear(
                    "no_norm" if i == num_layers - 1 else norm,
                    dim=output_dims if i == num_layers - 1 else hidden_dims,
                ),
                get_activation(
                    "no_act" if i == num_layers - 1 else act,
                ),
            )
            for i in range(num_layers)
        ]
        self.position_mlp = nn.Sequential(*layers)

        self.apply(init_weights)

    def forward(self, points: Tensor) -> Tensor:
        """Simple model mapping a viewing angle to an embedding vector.

        Args:
            points: The point cloud, with shape (B, N, 3)

        Returns:
            The output embedding for the viden views, with shape (B, N, E)
        """

        # Embeds the (X, Y, Z) coordinates.
        pos_embs = self.pos_embs(points)
        preds = self.position_mlp(pos_embs)
        
        return preds[:, :self.text_rep_size], preds[:, -self.image_rep_size:]

    def compute_loss(
        self, predicted_latents, actual_latents, label_mask=None, weights=None
    ):
        normalized_predicted_latents = F.normalize(predicted_latents, p=2, dim=-1)
        normalized_actual_latents = F.normalize(actual_latents, p=2, dim=-1)
        temp = torch.exp(self.temperature)
        sim = (
            torch.einsum(
                "i d, j d -> i j",
                normalized_predicted_latents,
                normalized_actual_latents,
            )
            * temp
        )
        # Zero out the cells where the labels are same.
        if label_mask is not None:
            sim = sim * label_mask
            del label_mask
        labels = torch.arange(len(predicted_latents), device=predicted_latents.device)
        if weights is None:
            loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
        else:
            loss = (
                F.cross_entropy(sim, labels, reduction="none")
                + F.cross_entropy(sim.t(), labels, reduction="none")
            ) / 2
            loss = (loss * weights).mean()
        return loss