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

        self.attention_heads = MODEL['attention_heads']
        self.head_dim = MODEL['embedding_dim'] // self.attention_heads

        self.gnn_blocks = nn.ModuleList()
        for i in range(MODEL['gnn_layers']):
            query_proj = nn.Linear(MODEL['embedding_dim'], MODEL['embedding_dim'], bias=False)
            key_proj = nn.Linear(MODEL['embedding_dim'], MODEL['embedding_dim'], bias=False)
            value_proj = nn.Linear(MODEL['embedding_dim'], MODEL['embedding_dim'], bias=False)

            update_net = nn.Sequential(
                nn.Linear(MODEL['embedding_dim'] * 2, 128),
                nn.ReLU(),
                nn.Linear(128, MODEL['embedding_dim'])
            )

            norm_layer = nn.LayerNorm(MODEL['embedding_dim'])

            self.gnn_blocks.append(nn.ModuleDict({
                'query_proj': query_proj,
                'key_proj': key_proj,
                'value_proj': value_proj,
                'update_net': update_net,
                'norm': norm_layer
            }))

        self.output_mlp = nn.Sequential(
            nn.Linear(MODEL['embedding_dim'], 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, edge_features):
        self.g.edata['h'] = self.edge_proj(edge_features)

        for block in self.gnn_blocks:
            h = self.g.edata['h']
            lg = self.lg

            # Multi-head projections
            Q = block['query_proj'](h).view(-1, self.attention_heads, self.head_dim)
            K = block['key_proj'](h).view(-1, self.attention_heads, self.head_dim)
            V = block['value_proj'](h).view(-1, self.attention_heads, self.head_dim)

            lg.ndata['Q'] = Q
            lg.ndata['K'] = K
            lg.ndata['V'] = V

            def apply_attention(edges):
                q = edges.src['Q']     # [E, H, Dk]
                k = edges.dst['K']     # [E, H, Dk]
                score = (q * k).sum(dim=-1, keepdim=True) / (self.head_dim ** 0.5)  # [E, H, 1]
                return {'score': score, 'V': edges.src['V']}

            lg.apply_edges(apply_attention)

            # Use edge_softmax to normalize attention scores per target node
            from dgl.nn.functional import edge_softmax
            attn_weights = edge_softmax(lg, lg.edata['score'])  # [E, H, 1]
            lg.edata['attn'] = attn_weights

            def message_func(edges):
                return {'m': edges.data['attn'] * edges.data['V']}  # [E, H, D]

            def reduce_func(nodes):
                return {'agg_msg': torch.sum(nodes.mailbox['m'], dim=1)}  # [N, H, D] -> [N, D]

            lg.update_all(message_func, reduce_func)
            agg_msg = lg.ndata['agg_msg'].view(-1, MODEL['embedding_dim'])

            concat_input = torch.cat([h, agg_msg], dim=-1)
            h_new = block['update_net'](concat_input)
            h_updated = block['norm'](h + h_new)
            self.g.edata['h'] = h_updated

        return self.output_mlp(self.g.edata['h'])
