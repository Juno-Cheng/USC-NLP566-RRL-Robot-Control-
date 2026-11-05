# Chelroy & Juno
"""
               +------------------+
               |  Reacher-v5      |
               |  (same simulator)|
               +--------+---------+
                        |
               10 sensor readings
                        |
         +--------------+--------------+
         |                             |
C0 / C3: leave them alone     C1 / C2: corruptions.py
         |                             |
         +--------------+--------------+
                        |
               readings go into
                        |
         +--------------+--------------+
         |                             |
C0 / C1 / C2: normal NN         C3: rolldrop.py
         |                             |  (drops some inputs
         |                             |   while practicing)
         +--------------+--------------+
                        |
                  2 joint torques
                        |
                  back to simulator
"""