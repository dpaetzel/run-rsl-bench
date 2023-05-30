# Utilities to interact with datasets created by `syn-rsl-benchs`.
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

import hashlib


def file_digest(fname):
    with open(fname, 'rb') as f:
        hash_object = hashlib.sha256()
        # Avoid loading large files into memory by reading in chunks.
        for chunk in iter(lambda: f.read(4096), b''):
            hash_object.update(chunk)
    return hash_object.hexdigest()


# TODO Consider moving to syn_rsl_benchs
def get_train(data):
    X, y = data["X"], data["y"]
    y = y.reshape(len(X), -1)
    return X, y


# TODO Consider moving to syn_rsl_benchs
def get_test(data):
    X_test = data["X_test"]
    try:
        y_test = data["y_test_true"]
    except KeyError:
        y_test = data["y_test"]
    y_test = y_test.reshape(len(X_test), -1)
    return X_test, y_test
