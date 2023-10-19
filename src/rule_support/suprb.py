# An extension of suprb for if-then rule extraction.
#
# Copyright (C) 2023 David Pätzel
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import numpy as np  # type: ignore
import suprb  # type: ignore
from sklearn.utils.validation import check_is_fitted  # type: ignore

from .utils import clamp_transform


class SupRB(suprb.SupRB):
    @property
    def rules_(self):
        return self.elitist_.subpopulation

    def bounds_(self, X_min, X_max, transformer_X=None):
        rules = self.rules_
        lowers = [rule.match.bounds[:, 0] for rule in self.rules_]
        uppers = [rule.match.bounds[:, 1] for rule in self.rules_]

        return clamp_transform(lowers, uppers, X_min, X_max, transformer_X)
