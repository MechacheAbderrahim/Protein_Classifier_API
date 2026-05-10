import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout, Sequential, ReLU, BatchNorm1d

from models.pointnets.pointnet_cls import get_model as pointnet_model

# === Utilitaires ===
def count_parameters(model):
    """Compte le nombre total de paramètres du modèle."""
    return sum(p.numel() for p in model.parameters())

def build_mlp_nnconv(in_dim, dims, out_dim):
    layers = []
    prev = in_dim
    for d in dims:
        layers += [nn.Linear(prev, d), nn.ReLU()]
        prev = d
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)

  
class ProteinSurfaceGCNN(nn.Module):
    """
    Version plus robuste du GCNN :
    - projection d'entrée optionnelle
    - BatchNorm optionnelle
    - connexions résiduelles
    - dropout dans les blocs GCN
    - graph embedding conservé après mean+max pooling
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 99,
        physioc: bool = False,
        params: dict = {},
    ):
        super().__init__()
        from torch_geometric.nn import GCNConv

        if physioc:
            in_channels = 4

        emb_dim = params["EMBEDDING_SIZE"]
        n_layers = params["GCN_LAYERS"]
        mlp_dims = list(params["MLP_DIMS"])
        mlp_dropout = params["DROP_OUT"]

        use_input_proj = params["USE_INPUT_PROJ"]
        use_residual = params["USE_RESIDUAL"]
        use_bn = params["USE_BN"]
        conv_dropout = params["CONV_DROPOUT"]

        self.use_input_proj = use_input_proj
        self.use_residual = use_residual
        self.use_bn = use_bn

        self.relu = nn.ReLU()
        self.conv_dropout = nn.Dropout(conv_dropout)

        # ----- Optional input projection -----
        if use_input_proj:
            self.input_proj = nn.Sequential(
                nn.Linear(in_channels, emb_dim),
                nn.BatchNorm1d(emb_dim),
                nn.ReLU(),
                nn.Dropout(conv_dropout),
            )
            first_in_dim = emb_dim
        else:
            self.input_proj = None
            first_in_dim = in_channels

        # ----- GCN blocks -----
        self.gcn_layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.skip_projs = nn.ModuleList()

        for i in range(n_layers):
            in_dim = first_in_dim if i == 0 else emb_dim
            out_dim = emb_dim

            self.gcn_layers.append(GCNConv(in_dim, out_dim))

            if use_bn:
                self.bns.append(nn.BatchNorm1d(out_dim))
            else:
                self.bns.append(nn.Identity())

            if use_residual:
                if in_dim == out_dim:
                    self.skip_projs.append(nn.Identity())
                else:
                    self.skip_projs.append(nn.Linear(in_dim, out_dim))
            else:
                self.skip_projs.append(nn.Identity())

        # ----- MLP classifier -----
        self.mlp_layers = nn.ModuleList()
        in_dim = emb_dim * 2  # mean + max pooling

        for dim in mlp_dims:
            self.mlp_layers.append(nn.Linear(in_dim, dim))
            self.mlp_layers.append(nn.BatchNorm1d(dim))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(mlp_dropout))
            in_dim = dim

        self.mlp_out = nn.Linear(in_dim, num_classes)

    def forward(self, x, edge_index, batch):
        from torch_geometric.nn import global_max_pool, global_mean_pool

        # ----- Optional projection -----
        if self.input_proj is not None:
            x = self.input_proj(x)

        # ----- Message passing -----
        for gcn, bn, skip_proj in zip(self.gcn_layers, self.bns, self.skip_projs):
            identity = x
            x = gcn(x, edge_index)
            x = bn(x)

            if self.use_residual:
                identity = skip_proj(identity)
                x = x + identity

            x = self.relu(x)
            x = self.conv_dropout(x)

        # ----- Global pooling -----
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        graph_emb = x.clone()

        # ----- MLP -----
        for layer in self.mlp_layers:
            x = layer(x)

        logits = self.mlp_out(x)
        return logits, graph_emb

class NNConvClassifier(torch.nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 99,
        physioc: bool = False,
        params: dict = {},
    ):
        super().__init__()
        from torch_geometric.nn import NNConv

        if physioc:
            in_channels = 4

        edge_dim = params["EDGE_DIM"]
        edge_mlp_dim = params["EDGE_MLP_DIM"]
        embedding_size = params["EMBEDDING_SIZE"]
        dropout = params["DROPOUT"]

        mlp_dims = params.get("MLP_DIMS", [256, 128, 64])

        # -------- Edge MLP --------
        self.edge_nn = Sequential(
            Linear(edge_dim, edge_mlp_dim),
            ReLU(),
            Linear(edge_mlp_dim, in_channels * embedding_size)
        )

        self.conv = NNConv(
            in_channels,
            embedding_size,
            nn=self.edge_nn,
            aggr=params["AGGR"]
        )

        self.bn_conv = BatchNorm1d(embedding_size)

        # -------- Graph MLP dynamique --------
        self.mlp = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.drops = torch.nn.ModuleList()

        input_dim = embedding_size * 2  # max_pool + mean_pool

        for hidden_dim in mlp_dims:
            self.mlp.append(Linear(input_dim, hidden_dim))
            self.bns.append(BatchNorm1d(hidden_dim))
            self.drops.append(Dropout(p=dropout))
            input_dim = hidden_dim

        self.out = Linear(input_dim, num_classes)

    def forward(self, x, edge_index, edge_attr, batch_index):
        from torch_geometric.nn import global_max_pool, global_mean_pool

        x = self.conv(x, edge_index, edge_attr)
        x = self.bn_conv(x)
        x = F.relu(x)

        graphs_embs = torch.cat([
            global_max_pool(x, batch_index),
            global_mean_pool(x, batch_index)
        ], dim=1)

        x = graphs_embs

        for layer, bn, drop in zip(self.mlp, self.bns, self.drops):
            x = F.relu(bn(layer(x)))
            x = drop(x)

        return self.out(x), graphs_embs

def get_model_architecture(name: str, num_classes: int, physioc: bool, model_params: dict):
    name = name.lower()
    if name.startswith("sfm"):
        model = ProteinSurfaceGCNN(num_classes=num_classes, physioc=physioc, params=model_params)
    elif name.startswith("ee"):
        model = NNConvClassifier(num_classes=num_classes, physioc=physioc, params=model_params)
    elif name.startswith("pointnet"):
        model = pointnet_model(k=num_classes, normal_channel=physioc)

    else:
        raise ValueError(f"Modèle inconnu ou non supporté : {name}. "
                         "Essayez parmi : gcnn1 gcnn2 gcnn3 pointnet")

    n_params = count_parameters(model)
    return model, n_params
