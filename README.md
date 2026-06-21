# Reaction ECFPs

Graph-to-vector reaction application in molecular fingerprint space.

This repository contains tools to represent chemical reactions as signed transformations between counted molecular ECFP vectors. It focuses on the link between graph-level reaction templates and vector-space reaction operators.

## Scope

The project provides utilities to:

- compute molecular ECFPs;
- compute reaction ECFPs as product-minus-substrate fingerprint differences;
- compute reaction-center ECFPs for fast applicability filtering;
- build ECFP-compatible reaction templates from mapped reactions;
- analyze and compare reaction representations across datasets such as MetaNetX and USPTO;
- run benchmark tasks such as reaction-class and EC-number prediction.

## Concept

For a reaction `S -> P`, the reaction ECFP is represented as:

```text
reaction ECFP = ECFP(P) - ECFP(S)
```

Reaction-center ECFPs encode the local molecular environments required for a reaction to be applicable. They can be used as fast coordinate-wise filters before graph-level template validation.

## Status

This repository is under active development and accompanies ongoing research on reaction representations in molecular fingerprint space.

## Citation

Citation information will be added with the associated manuscript.
