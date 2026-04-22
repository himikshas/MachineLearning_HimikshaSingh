#!/usr/bin/env python3

""" Implement twitter sentiment prediction using SVM -
    Try different kernel functions and compare the results. """

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text