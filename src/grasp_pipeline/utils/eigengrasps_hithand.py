import numpy as np

# Index, Little, Middle, Ring, Thumb
########## Type 1: thumb abduction min

THUMB_ABD_MIN = np.array([ 0.0, 0.0, 0.0, 0.0, \
                    0.0, 0.0, 0.0, 0.0, \
                    0.0, 0.0, 0.0, 0.0, \
                    0.0, 0.0, 0.0, 0.0, \
                    -0.26179, 0.0, 0.0, 0.0 ])

# max

THUMB_ABD_MAX = np.array([ 0.0, 0.0, 0.0, 0.0, \
                    0.0, 0.0, 0.0, 0.0, \
                    0.0, 0.0, 0.0, 0.0, \
                    0.0, 0.0, 0.0, 0.0, \
                    0.1, 0.0, 0.0, 0.0 ])

# ########## Type 2: Finger spread min

SPREAD_MAX = np.array([ -0.26179, 0.0, 0.0, 0.0, \
                    0.26179, 0.0, 0.0, 0.0, \
                    -0.13, 0.0, 0.0, 0.0, \
                    0.13, 0.0, 0.0, 0.0, \
                    0.0, 0.0, 0.0, 0.0 ])

# max

SPREAD_MIN = np.array([ 0.09, 0.0, 0.0, 0.0, \
                    -0.09, 0.0, 0.0, 0.0, \
                    0.03, 0.0, 0.0, 0.0, \
                    -0.03, 0.0, 0.0, 0.0, \
                    0.0, 0.0, 0.0, 0.0 ])

# ########## Type 3: MCP min

MCP_MIN = np.array([ 0.0, 0.087266, 0.0, 0.0, \
                    0.0, 0.087266, 0.0, 0.0, \
                    0.0, 0.087266, 0.0, 0.0, \
                    0.0, 0.087266, 0.0, 0.0, \
                    0.0, 0.087266, 0.0, 0.0 ])

# max

MCP_MAX = np.array([ 0.0, 1, 0.0, 0.0, \
                    0.0, 1, 0.0, 0.0, \
                    0.0, 1, 0.0, 0.0, \
                    0.0, 1, 0.0, 0.0, \
                    0.0, 0.7, 0.0, 0.0 ])

# ########## Type 3: PIP min

PIP_MIN = np.array([ 0.0, 0.0, 0.087266, 0.087266, \
                    0.0, 0.0, 0.087266, 0.087266, \
                    0.0, 0.0, 0.087266, 0.087266, \
                    0.0, 0.0, 0.087266, 0.087266, \
                    0.0, 0.0, 0.087266, 0.087266 ])

# max

PIP_MAX = np.array([ 0.0, 0.0, 1, 1, \
                    0.0, 0.0, 1, 1, \
                    0.0, 0.0, 1, 1, \
                    0.0, 0.0, 1, 1, \
                    0.0, 0.0, 1, 1 ])