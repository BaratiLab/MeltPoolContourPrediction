# power_velocity_map.py
# Mapping for sets and beads to associated power and velocity combination.

def get_power_velocity(set_number, bead_number):
    mapping = [
        # Set 1
        [
            [50, 0.3],  # Bead 1
            [50, 0.6],  # Bead 2
            [50, 0.9],  # Bead 3
            [50, 1.2]   # Bead 4
        ],
        # Set 2
        [
            [100, 0.3], # Bead 1
            [100, 0.6], # Bead 2
            [100, 0.9], # Bead 3
            [100, 1.2]  # Bead 4
        ],
        # Set 3
        [
            [150, 0.3], # Bead 1
            [150, 0.6], # Bead 2
            [150, 0.9], # Bead 3
            [150, 1.2]  # Bead 4
        ],
        # Set 4
        [
            [200, 0.3], # Bead 1
            [200, 0.6], # Bead 2
            [200, 0.9], # Bead 3
            [200, 1.2]  # Bead 4
        ],
        # Set 5
        [
            [250, 0.3], # Bead 1
            [250, 0.6], # Bead 2
            [250, 0.9], # Bead 3
            [250, 1.2]  # Bead 4
        ],
        # Set 6
        [
            [300, 0.3], # Bead 1
            [300, 0.6], # Bead 2
            [300, 0.9], # Bead 3
            [300, 1.2]  # Bead 4
        ],
        # Set 7
        [
            [350, 0.3], # Bead 1
            [350, 0.6], # Bead 2
            [350, 0.9], # Bead 3
            [350, 1.2]  # Bead 4
        ],
        # Set 8
        [
            [50, 1.5],  # Bead 1
            [50, 1.8],  # Bead 2
            [50, 2.1],  # Bead 3
            [50, 2.4]   # Bead 4
        ],
        # Set 9
        [
            [100, 1.5],  # Bead 1
            [100, 1.8],  # Bead 2
            [100, 2.1],  # Bead 3
            [100, 2.4]   # Bead 4
        ],
        # Set 10
        [
            [150, 1.5],  # Bead 1
            [150, 1.8],  # Bead 2
            [150, 2.1],  # Bead 3
            [150, 2.4]   # Bead 4
        ],
        # Set 11
        [
            [200, 1.5],  # Bead 1
            [200, 1.8],  # Bead 2
            [200, 2.1],  # Bead 3
            [200, 2.4]   # Bead 4
        ],
        # Set 12
        [
            [250, 1.5],  # Bead 1
            [250, 1.8],  # Bead 2
            [250, 2.1],  # Bead 3
            [250, 2.4]   # Bead 4
        ],
        # Set 13
        [
            [300, 1.5],  # Bead 1
            [300, 1.8],  # Bead 2
            [300, 2.1],  # Bead 3
            [300, 2.4]   # Bead 4
        ],
        # Set 14
        [
            [350, 1.5],  # Bead 1
            [350, 1.8],  # Bead 2
            [350, 2.1],  # Bead 3
            [350, 2.4]   # Bead 4
        ],
        # Set 15
        [
            [400, 1.5],  # Bead 1
            [400, 1.8],  # Bead 2
            [400, 2.1],  # Bead 3
            [400, 2.4]   # Bead 4
        ],
        # Set 16
        [
            [400, 0.3],  # Bead 1
            [400, 0.6],  # Bead 2
            [400, 0.9],  # Bead 3
            [400, 1.2]   # Bead 4
        ],
    ]

    return mapping[set_number - 1][bead_number -1]
