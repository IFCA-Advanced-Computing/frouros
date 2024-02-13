# Batch

The {mod}`frouros.detectors.data_drift.batch` module contains batch data drift detection algorithms.

```{eval-rst}
.. automodule:: frouros.detectors.data_drift.batch
    :no-members:
    :no-inherited-members:
```

```{currentmodule} frouros.detectors.data_drift.batch
```

## Distance Based

```{eval-rst}
.. automodule:: frouros.detectors.data_drift.batch.distance_based
    :no-members:
    :no-inherited-members:
```

```{eval-rst}
.. autosummary::
    :toctree: auto_generated/
    :template: class.md

    BhattacharyyaDistance
    EMD
    EnergyDistance
    HellingerDistance
    HINormalizedComplement
    JS
    KL
    MMD
    PSI
```

## Statistical Test

```{eval-rst}
.. automodule:: frouros.detectors.data_drift.batch.statistical_test
    :no-members:
    :no-inherited-members:
```

```{eval-rst}
.. autosummary::
    :toctree: auto_generated/
    :template: class.md

    AndersonDarlingTest
    BWSTest
    ChiSquareTest
    CVMTest
    KSTest
    MannWhitneyUTest
    WelchTTest
```