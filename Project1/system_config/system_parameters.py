import numpy as np
from typing import Dict, Set, List


class SystemParameters:
    """
    System Parameters from Section 3.1.2.
    
    Pre-defined and fixed parameters for the retail simulation universe.
    """
    
    def __init__(
        self,
        K: int,  # number of items displayed on the website
    ):
        self.K = K  # items displayed on website
