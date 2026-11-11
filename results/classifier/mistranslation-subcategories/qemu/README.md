# QEMU mistranslation subcategories

Subcategories of the QEMU user-mode mistranslations, as defined in the Focaccia paper:

- **crash**: the instruction raises a fault or signal it should not, or fails to raise one it
  should. This includes unimplemented or misdecoded instructions, and QEMU aborting while
  translating them.
- **optimization**: the error comes from an optimisation across several instructions, rather than
  from the translation of a single instruction.
- **memory**: a load or store has the wrong address, size, alignment, ordering or atomicity.
- **vector**: wrong results in vector, SIMD or floating-point registers.
- **flag**: the result is correct, but condition codes or status flags are wrong.
- **computation**: a scalar instruction writes a wrong result to general-purpose registers.

When several fit, the first in this order applies: crash, optimization, memory, vector, flag,
computation.

Each classified report is stored as `<category>/<bug-id>`.

## `gpt-6-luna/`

A classification by a language model of the 182 reports it labelled mistranslation in
`../../error-types/qemu/gpt-6-luna/`. `output/` holds the reports and `reasoning/` holds the
model's answer for each: the category, followed by one sentence of evidence from the report.

The paper reports shares for the manually identified mistranslations. The table compares them with
the model's labels for all 182 reports, and for the 112 of them that are also among the manually
identified mistranslations in `../../error-types/qemu/paper/`.

| Category     | All | Share | Manual subset | Share | Paper |
| ------------ | --: | ----: | ------------: | ----: | ----: |
| crash        |  71 |   39% |            36 |   32% |   32% |
| vector       |  34 |   19% |            22 |   20% |   27% |
| computation  |  30 |   16% |            20 |   18% |   19% |
| memory       |  28 |   15% |            20 |   18% |    9% |
| flag         |  17 |    9% |            14 |   13% |   12% |
| optimization |   2 |    1% |             0 |    0% |    1% |

The classification was produced on 2026-11-11 with:

- model `openai-codex/gpt-6-luna` with thinking level `low`, through pi 1.4.0;
- prompt `data/prompts/mistranslation-subcategory.txt` and categories
  `data/configs/mistranslation-subcategory.toml`.

To reproduce it, run from the repository root:

```sh
nix run .#bug-classifier -- --backend pi --model openai-codex/gpt-6-luna --thinking low \
  --config data/configs/mistranslation-subcategory.toml \
  --preamble data/prompts/mistranslation-subcategory.txt \
  -i results/classifier/error-types/qemu/gpt-6-luna/output/mistranslation \
  -o results/classifier/mistranslation-subcategories/qemu/gpt-6-luna/output
```

Model output is not deterministic, so a new run may differ for borderline reports.
