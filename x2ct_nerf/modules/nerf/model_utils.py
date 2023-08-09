import torch
from x2ct_nerf.modules.nerf import nerf_helpers
import pdb

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn

    def ret(inputs, axis=0):
        if axis == 1:
            output_dict = {}
            for i in range(0, inputs.shape[1], chunk):
                for k, v in fn(inputs[:, i: i + chunk]).items():
                    if i == 0:
                        output_dict[k] = v
                    else:
                        output_dict[k] = torch.cat([output_dict[k], v], 0)
            return output_dict
        else:
            output_dict = {}
            for i in range(0, inputs.shape[0], chunk):
                for k, v in fn(inputs[:, i: i + chunk]).items():
                    if i == 0:
                        output_dict[k] = v
                    else:
                        output_dict[k] = torch.cat([output_dict[k], v], 0)
            return output_dict
            # return torch.cat([fn(inputs[i: i + chunk])['outputs'] for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64, axis=0):
    """Prepares inputs and applies network 'fn'."""
    if axis == 1:
        inputs_flat = torch.reshape(inputs, [len(inputs), -1, inputs.shape[-1]])
    else:
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])

    if hasattr(fn, "encoder") or hasattr(fn, "cond_encoder"):
        inputs_flat, features = inputs_flat[..., :3], inputs_flat[..., 3:]
        embedded = embed_fn(inputs_flat)
        embedded = torch.cat((embedded, features), dim=-1)
    else:
        embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[..., None].expand(inputs.shape)
        if axis == 1:
            input_dirs_flat = torch.reshape(input_dirs, [len(inputs), -1, input_dirs.shape[-1]])
        else:
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_dict = batchify(fn, netchunk)(embedded, axis=axis)
    for k in outputs_dict:
        outputs_dict[k] = torch.reshape(outputs_dict[k], list(inputs.shape[:-1]) + [outputs_dict[k].shape[-1]])

    assert isinstance(outputs_dict, dict)
    return outputs_dict

def update_nerf_params(cfg):
    embed_fn, cfg['input_ch'] = nerf_helpers.get_embedder(cfg['multires'], cfg['i_embed'])

    if cfg['use_viewdirs']:
        embeddirs_fn, cfg['input_ch_views'] = nerf_helpers.get_embedder(cfg['multires_views'], cfg['i_embed'])
    else:
        embeddirs_fn = None
        cfg['input_ch_views'] = 0

    cfg['output_ch'] = cfg['output_color_ch']

    def network_query_fn(inputs, viewdirs, network_fn): return run_network(
        inputs, viewdirs, network_fn, embed_fn=embed_fn,
        embeddirs_fn=embeddirs_fn, netchunk=cfg['netchunk'], axis=cfg['batchify_axis']
    )
    return cfg, network_query_fn