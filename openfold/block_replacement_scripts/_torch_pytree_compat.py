"""
Compatibility shim for environments where `transformers` expects
`torch.utils._pytree.register_pytree_node` but the installed Torch exposes only
`_register_pytree_node`.

This is needed because some transitive imports (e.g. PyTorch Lightning ->
torchmetrics -> torchvision -> torch.onnx -> transformers) can crash at import
time otherwise, even though we do not use transformers directly.
"""

import torch


def _ensure_register_pytree_node() -> None:
    pytree = getattr(getattr(torch, "utils", None), "_pytree", None)
    if pytree is None:
        return
    if hasattr(pytree, "register_pytree_node"):
        return
    if hasattr(pytree, "_register_pytree_node"):
        def register_pytree_node(
            typ,
            flatten_fn,
            unflatten_fn,
            *,
            to_dumpable_context=None,
            from_dumpable_context=None,
            serialized_type_name=None,
            **_ignored_kwargs,
        ) -> None:
            # Newer `transformers` may pass `serialized_type_name` and other kwargs
            # that older Torch `_register_pytree_node` does not accept.
            return pytree._register_pytree_node(
                typ,
                flatten_fn,
                unflatten_fn,
                to_dumpable_context=to_dumpable_context,
                from_dumpable_context=from_dumpable_context,
            )

        pytree.register_pytree_node = register_pytree_node


_ensure_register_pytree_node()

