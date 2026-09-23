---
title: >
  Supplementary Material: Implementation Details of the
  Agent-Based Psychological Resilience Simulation
author: Aly Lamuri
format:
  pdf:
    keep-tex: true
    number-sections: true
    code-overflow: wrap
    include-in-header:
      text: |
        \renewcommand{\thesection}{S\arabic{section}}
        \renewcommand{\thesubsection}{S\arabic{section}.\arabic{subsection}}
        \usepackage{tabularx}
        \newcommand{\trow}[3]{%
          \texttt{\allowbreak #1} & #2 & #3 \\
        }
        \newcommand{\trowfour}[4]{%
          \texttt{\allowbreak #1} & #2 & #3 & #4 \\
        }
---

{{< include design/_protocol.md >}}

{{< include design/_schema.md >}}

{{< include design/_orchestration.md >}}

# Phase Implementations

{{< include design/_stress-perception.md >}}

{{< include design/_resilience-activation.md >}}

{{< include design/_interaction.md >}}

{{< include design/_resource-allocation.md >}}

{{< include design/_stress-buffering.md >}}

{{< include design/_affect-dynamics.md >}}

{{< include design/_pss10-consolidation.md >}}

{{< include design/_daily-reset.md >}}

{{< include design/_cumulative-rationale.md >}}

{{< include design/_network.md >}}

{{< include design/_data-collection.md >}}
