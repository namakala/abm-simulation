# Python Modules Import Organization

The codebase follows strict import organization standards to ensure maintainability and avoid circular dependencies:

## **Module-Level Import Pattern (Required):**

- All imports MUST be declared at the module level (top of file)
- Function-level imports are strictly prohibited
- This ensures clear dependency visibility and prevents import-related runtime errors

## **Import Structure:**

```python
# 1. Standard library imports (alphabetical)
import hashlib
import random
from typing import Dict, List, Tuple

# 2. Third-party imports (alphabetical)
import mesa
import networkx as nx
import numpy as np
import pandas as pd

# 3. Local imports (organized by module)
from src.python.config import get_config
from src.python.affect_utils import (
    process_interaction, compute_stress_impact_on_affect,
    InteractionConfig, AffectDynamicsConfig
)
from src.python.stress_utils import (
    generate_stress_event, process_stress_event,
    StressEvent, AppraisalWeights, ThresholdParams
)
```

## **Import Guidelines:**

- One import per line for better readability and easier maintenance
- Use import aliasing consistently (e.g., `import numpy as np`)
- Group related imports together with blank lines for separation
- Import specific functions/classes rather than entire modules when possible
- Avoid relative imports in favor of absolute imports from package root
