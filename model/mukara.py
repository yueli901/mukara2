import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from model.dataloader import load_dgl_graph
from config import MODEL

class Mukara(nn.Module):
    def __init__(self):
        super(Mukara, self).__init__()

        self.g, self.edge_ids = load_dgl_graph()
        print(f'Converting to line graph...')
        self.lg = dgl.line_graph(self.g, backtracking=False)
        self.num_edges = self.g.number_of_edges()

        self.edge_proj = nn.Sequential(
            nn.Linear(self.g.edata['feat'].shape[1], 128),
            nn.ReLU(),
            nn.Linear(128, MODEL['embedding_dim'])
        )

        self.gnn_blocks = nn.ModuleList()
        for i in range(MODEL['gnn_layers']):
            update_net = nn.Sequential(
                nn.Linear(2 * MODEL['embedding_dim'], 128),
                nn.ReLU(),
                nn.Linear(128, MODEL['embedding_dim'])
            )
            norm_layer = nn.LayerNorm(MODEL['embedding_dim'])
            self.gnn_blocks.append(nn.ModuleDict({
                'update_net': update_net,
                'norm': norm_layer
            }))

        self.output_mlp = nn.Sequential(
            nn.Linear(MODEL['embedding_dim'], 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, edge_features):
        # Initial edge embedding
        self.g.edata['h'] = self.edge_proj(edge_features)

        for block in self.gnn_blocks:
            lg = self.lg
            lg.ndata['h'] = self.g.edata['h']
            lg.update_all(
                message_func=fn.copy_u('h', 'm'),
                reduce_func=fn.mean('m', 'agg_msg')
            )

            h_old = lg.ndata['h']
            h_new = block['update_net'](torch.cat([h_old, lg.ndata['agg_msg']], dim=-1))
            h_updated = block['norm'](h_old + h_new)

            self.g.edata['h'] = h_updated

        return self.output_mlp(self.g.edata['h'])
