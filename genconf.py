#!/usr/bin/env python
"""Generate python config with multiple instances of the ghostbuster algorithm

"""

import ast
from dataclasses import asdict, replace
from pathlib import Path

import sh

from scanprops import git_root, jobopts_t


def edit_options(src: str, opts: jobopts_t) -> str:
    tree = ast.parse(src)
    # contextmanager
    with_block, *_ = [node for node in tree.body if isinstance(node, ast.With)]
    with_item, *_ = with_block.items
    with_item.context_expr.keywords = [
        ast.keyword(prop, ast.Constant(getattr(opts, prop)))
        for prop in asdict(opts)
        if prop not in opts.not_props
    ]

    # line 1: ghostbuster copies
    seq_assign, *_ = [node for node in with_block.body if isinstance(node, ast.Assign)]
    seq_assign.value.keywords = [
        ast.keyword("with_ghostbuster", ast.Constant(True)),
        ast.keyword("ghostbuster_copies", ast.Constant(opts.copies)),
    ]
    return ast.unparse(tree)


def write_config_py(builddir: str, sequence: str, opts: jobopts_t) -> list[Path]:
    files = []
    with sh.cd(builddir):
        pyconfig = git_root() / f"configuration/python/AllenSequences/{sequence}.py"
        for i in range(1, opts.copies + 1):
            config = edit_options(pyconfig.read_text(), replace(opts, copies=i))
            pyconfig_edited = pyconfig.with_stem(f"{sequence}_n{i}")
            pyconfig_edited.write_text(config)
            files.append(pyconfig_edited)
    return files


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "builddir",
        help="Repo or build directory, a subdir of the repo",
    )
    parser.add_argument(
        "--max-copies", type=int, default=1, help="Max number of algo instances"
    )
    parser.add_argument("--sequence", default="ghostbuster_test")

    opts = parser.parse_args()
    jobopts = jobopts_t(
        max_batch_size=1024,
        no_infer=False,
        use_fp16=False,
        onnx_input="/project/bfys/suvayua/codebaby/Allen/input/ghost_nn.onnx",
        input_name="dense_input",
        copies=opts.max_copies,
    )

    for f in write_config_py(opts.builddir, opts.sequence, jobopts):
        print(f.stem)
