import unittest
import numpy as np
from copy import deepcopy

from RIM_weightings import rim, prepare_data


class RIMWeightingsTest(unittest.TestCase):

    def setUp(self) -> None:
        self.panelists, self.targets = prepare_data()
        self.rim, _ = rim(deepcopy(self.panelists), deepcopy(self.targets), 6)
        self.rim_dd, _ = rim(deepcopy(self.panelists), deepcopy(self.targets), 20, dd=True)

    def test_gender_weights(self):
        for cond_value in [0,1]:
            self.check_target_diff(self.rim, 'Gender', cond_value)

    def test_age_groups_weights(self):
        for cond_value in [0, 1, 2]:
            self.check_target_diff(self.rim, 'Age_group', cond_value)

    def test_gender_weights_dd(self):
        for cond_value in [0,1]:
            self.check_target_diff(self.rim_dd, 'Gender', cond_value)

    def test_age_groups_weights_dd(self):
        for cond_value in [0, 1, 2]:
            self.check_target_diff(self.rim_dd, 'Age_group', cond_value)

    def check_target_diff(self, df_rim, factor, cond_value):
        self.assertTrue(
            np.allclose(
                df_rim[df_rim[factor] == cond_value]['weights'].sum(),
                self.targets[(self.targets['Factor'] == factor) &
                             (self.targets['Condition_value'] == cond_value)]['Target_value'].values,
                atol=1e-5
            )
        )
