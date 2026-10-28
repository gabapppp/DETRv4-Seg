"""P2–P5 detector configuration and stage-aware P3–P5 checkpoint transfer."""

import re


def p2_config():
    return {
        "HGNetv2": {"return_idx": [0, 1, 2, 3]},
        "HybridEncoder": {
            "in_channels": [64, 256, 512, 1024],
            "feat_strides": [4, 8, 16, 32],
            "use_encoder_idx": [3],
        },
        "DFINETransformer": {
            "feat_channels": [256] * 4,
            "feat_strides": [4, 8, 16, 32],
            "num_levels": 4,
            "num_points": [3, 3, 6, 3],
        },
    }


def transfer_p2(weights, current, heads=8):
    """Keep existing feature levels aligned; initialize only the additional P2 path."""
    result = {}
    for name, value in weights.items():
        shifted = re.match(
            r"(encoder\.(?:input_proj|downsample_convs|pan_blocks)\.)(\d+)(\..*)", name
        )
        target = f"{shifted[1]}{int(shifted[2]) + 1}{shifted[3]}" if shifted else name
        if target not in current:
            continue
        if name.endswith((".num_points_scale", "decoder.anchors", "decoder.valid_mask")):
            continue
        if ".cross_attn." in name and ("sampling_offsets" in name or "attention_weights" in name):
            multiplier = 2 if "sampling_offsets" in name else 1
            old = value.reshape(heads, 12, multiplier, *value.shape[1:])
            expanded = current[target].clone().reshape(heads, 15, multiplier, *value.shape[1:])
            expanded[:, 3:] = old
            expanded[:, :3] = old[:, :3]
            if "attention_weights.bias" in name:
                expanded[:, :3] -= 2.0
            result[target] = expanded.reshape(current[target].shape)
        elif value.shape == current[target].shape:
            result[target] = value
    # Geometry is a deterministic buffer, not a learned parameter to transfer.
    for name, value in current.items():
        if name.endswith((".num_points_scale", "decoder.anchors", "decoder.valid_mask")):
            result[name] = value
    return result
