# OSSM Tasks

`ossm-tasks` implements the **Sensorimotor Task and Environment Format (STEF)** and provides a catalogue of ecologically valid tasks for goal-driven sensorimotor neuroscience.  
It is one of the core packages of the [OSSM ecosystem](https://github.com/ossm-team), supported by the [NWO Open Science Fund](https://www.nwo.nl/en/researchprogrammes/open-science/open-science-fund).

## Vision

This package is part of the effort to build an **open, community-driven ecosystem** for computational sensorimotor neuroscience.  
By standardizing task descriptions, OSSM enables tasks to be **shared, reproduced, and reused** across different labs and embodiments.  
The long-term goal is a growing catalogue of tasks that bridge sensory, motor, and cognitive capacities.

## Features

- **Schema-driven Standards**: Task definitions are validated against [`STEF.xsd`](./STEF.xsd).  
- **Ecologically valid**: Supports embodied, enactive tasks.

#### Features under Development

- **Python API**: Programmatic interfaces to create tasks.
- **Python STEF Loader**: Load STEF tasks into executable Gymnasium environments.

#### Planned features

- **Integration**: Interfaces to existing embodied tasks.
- **Catalogue**: A curated collection of reusable, well-documented tasks.  

## Install

Requires **Python 3.11+**.

```bash
pip install -e .
```

## Quickstart

Explore the [suite](./ossm-tasks/suite/) for ready-to-use task definitions.

Detailed documentation is under development.

## Related Packages

- [`ossm-base`](https://github.com/ossm-team/ossm-base) – shared types and utilities  
- [`ossm-models`](https://github.com/ossm-team/ossm-models) – model catalogue & SMS standard  
- [`ossm-analysis`](https://github.com/ossm-team/ossm-analysis) – analysis methods & benchmarks  

## Contribution

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md).  

## License

GPL-3.0. See [LICENSE](./LICENSE).
