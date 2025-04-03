import tensorflow as tf
import dgl
import dgl.function as fn
from model.dataloader import load_dgl_graph
from config import MODEL

class Mukara(tf.keras.Model):
    def __init__(self):
        super(Mukara, self).__init__()

        self.g, self.edge_ids = load_dgl_graph()
        self.num_edges = self.g.number_of_edges()

        self.edge_proj = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', name='edge_proj_dense1'),
            tf.keras.layers.Dense(MODEL['embedding_dim'], name='edge_proj_dense2')
        ], name='edge_proj')

        self.gnn_blocks = []
        for i in range(MODEL['gnn_layers']):
            update_net = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', name=f'update_dense1_{i}'),
                tf.keras.layers.Dense(MODEL['embedding_dim'], name=f'update_dense2_{i}')
            ], name=f'update_net_{i}')

            norm_layer = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f'norm_{i}')

            self.gnn_blocks.append((update_net, norm_layer))

        self.output_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', name='output_dense1'),
            tf.keras.layers.Dense(1, name='output_dense2')
        ], name='output_mlp')

    def call(self, edge_features):
        # Initial edge embedding
        self.g.edata['h'] = self.edge_proj(edge_features)

        for i, (update_net, norm_layer) in enumerate(self.gnn_blocks):
            lg = dgl.line_graph(self.g, backtracking=False)
            lg.ndata['h'] = self.g.edata['h']
            lg.ndata['h'] = tf.identity(lg.ndata['h'])

            lg.update_all(
                message_func=fn.copy_u('h', 'm'),
                reduce_func=fn.mean('m', 'agg_msg')
            )

            h_old = lg.ndata['h']
            h_new = update_net(tf.concat([h_old, lg.ndata['agg_msg']], axis=-1))
            h_updated = norm_layer(h_old + h_new)

            self.g.edata['h'] = h_updated

        return self.output_mlp(self.g.edata['h'])
