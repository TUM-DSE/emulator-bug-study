# QEMU user-mode error types

Error types of the 551 QEMU user-mode bug reports in `../../user-mode-bugs/`, as defined in the
Focaccia paper:

- **mistranslation**: guest instructions are incorrectly translated to host instructions.
- **syscall**: a system call is emulated or passed through incorrectly.
- **runtime**: all remaining execution failures that occur despite a correct translation.

Each classified report is stored as `<category>/<bug-id>`.

## `paper/`

The 119 mistranslation bugs identified by manual review, from which the paper reports that
mistranslations account for 22% of QEMU user-mode bugs. This is a copy of
`../../semantic-bugs/`. The manual split of the remaining bugs into runtime and syscall errors
was not preserved.

All 16 QEMU bugs cited in the paper are among them: 364, 508, 1370, 1371, 1372, 1374, 1375,
1376, 1377, 2175, 2248, 2419, 2495, 1828867, 1832422 and 1861404.

## `gpt-6-luna/`

A classification of all 551 reports by a language model. `output/` holds the reports and
`reasoning/` holds the model's answer for each: the category, followed by one sentence of
evidence from the report.

| Category       | Bugs | Share |
| -------------- | ---: | ----: |
| runtime        |  281 |   51% |
| mistranslation |  182 |   33% |
| syscall        |   88 |   16% |

The model labels 112 of the 119 manually identified mistranslations as mistranslation, including
all 16 bugs cited in the paper. It labels the other 7 (361, 422, 824, 2302, 1751494, 1809546 and
1862167) as runtime. It also finds 70 further mistranslations, so the paper's 22% is a lower
bound.

The classification was produced on 2026-11-10 with:

- model `openai-codex/gpt-6-luna` with thinking level `low`, through pi 1.4.0;
- prompt `data/prompts/qemu-error-type.txt` and categories `data/configs/qemu-error-type.toml`.

To reproduce it, run from the repository root:

```sh
nix run .#bug-classifier -- --backend pi --model openai-codex/gpt-6-luna --thinking low \
  --config data/configs/qemu-error-type.toml --preamble data/prompts/qemu-error-type.txt \
  -i results/classifier/user-mode-bugs -o results/classifier/error-types/qemu/gpt-6-luna/output
```

Model output is not deterministic, so a new run may differ for borderline reports.
