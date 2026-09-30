# Plugin Guide

!!! warning "Experimental plugin interfaces"
    Plugin discovery and custom-component registration are Experimental.
    Deep registry imports are Internal contributor interfaces. Use the Stable
    `make_problem` route when you only need to define an objective function. See
    [Stability and versioning](../project/stability-and-versioning.md).

Use [Extending VAMOS](extending.md) as the source of truth.

This page exists only as a redirect for contributors who search for “plugin guide”.

## Canonical paths

- Custom algorithms: register through `get_algorithms_registry().register(...)` for in-process extensions, or expose a `vamos.algorithms` entry point from an installed package.
- Custom operators: register through `get_operator_registry().register(...)` in `src/vamos/engine/operators/impl/registry.py`.
- Custom problems: implement `ProblemProtocol` or use `make_problem(...)`, then register the family/spec if the problem should be discoverable by name.

## Working rule

If this page and [Extending VAMOS](extending.md) ever disagree, follow [Extending VAMOS](extending.md) and update this page to point back there.
