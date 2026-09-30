# Research status

| Component | Current status | Main limitation |
|---|---|---|
| Media review | Local photos/video and manually reviewed timeline events work | Automatic video understanding is not validated |
| Recipe representation | Ordered graph with branching and merging | Free-text event matching is limited |
| Process simulation | Inspectable state transitions for common cooking operations | Physical rules are simplified placeholders |
| Sensory fingerprint | Eleven dimensions; ingredient-model taste estimates where supported, illustrative demo proxies, and separate user observations | Sparse composition data and uncalibrated response curves; demo proxies are not measured outcomes |
| Mamba residual | Integrated CPU runtime with artifact checks | Checkpoint trained on synthetic examples only |
| Evaluation | Contract and regression tests run locally | No controlled sensory dataset or accuracy benchmark yet |

## Claims this prototype supports

It can demonstrate how reviewed cooking events change an inspectable recipe-state trace and how an experimental estimate is produced from that trace.

## Claims it does not yet support

It cannot establish that its flavor scores match human perception, that Mamba improves accuracy, or that recipe optimization will reliably hit a desired sensory target.

## Data and publication boundary

Some underlying source-derived datasets have noncommercial or unclear redistribution terms. No raw or derived data tables, trained artifacts, application source, credentials, or Git history from the private repository belong in this public documentation repository without a separate review.
