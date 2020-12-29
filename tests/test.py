import pandas as pd
from MetalnessCalculator import MetalnessCalculator

metal_data = pd.read_csv('../mpi_py_example/data/dark_lyrics.csv.4', sep=',', escapechar='\\')
control_data = pd.read_csv('../mpi_py_example/data/light_lyrics.csv.1', encoding='utf-8', sep=',', dtype=str,escapechar='\\')

calc = MetalnessCalculator(metal_data,control_data)

test_str = """
But the Prince Prospero was happy and dauntless and sagacious. When his
dominions were half depopulated, he summoned to his presence a thousand hale
and light-hearted friends from among the knights and dames of his court, and
with these retired to the deep seclusion of one of his castellated abbeys. This
was an extensive and magnificent structure, the creation of the prince’s
own eccentric yet august taste. A strong and lofty wall girdled it in. This
wall had gates of iron. The courtiers, having entered, brought furnaces and
massy hammers and welded the bolts. They resolved to leave means neither of
ingress nor egress to the sudden impulses of despair or of frenzy from within.
The abbey was amply provisioned. With such precautions the courtiers might bid
defiance to contagion. The external world could take care of itself. In the
meantime it was folly to grieve, or to think. The prince had provided all the
appliances of pleasure. There were buffoons, there were improvisatori, there
were ballet-dancers, there were musicians, there was Beauty, there was wine.
All these and security were within. Without was the “Red Death”.
"""

print(calc.calculate_metalness_score(test_str))
