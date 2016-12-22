#!/usr/bin/python
# -*- coding: utf8 -*-

"""
Class to manage data of MIL problems

:author: Manuel Tuschen
:date: 23.06.2016
:license: GPL3
"""

from itertools import compress

import numpy as np
import pandas as pd
from miscpy import prepareSaving, prepareLoading

__all__ = ["milData"]


class milData():
    """
    Class to efficiently handel data for multiple instance learning (MIL).

    Attributes
    ----------
    name : string
        The name of the dataset.
    keys : list
        The sorted names of individual bags. For sorting pythons internal sort
        function is used.
    N_X : int
        The number of instances stored.
    N_B : int
        The number of bags stored.
    N_D : int
        The dimension of the feature vectors.
    N_b : ndarray
        The number of instances per bag.
    X : ndarray
        The instance features for all bags.
    z : ndarray
        The labels for all sorted bags.
    y : ndarray
        The instance labels for all sorted bags.
    pos_b : dict
        The (start, stop) positions during array slicing for each bag.
    pos_x : dict
        The keys and bag position for one global instance.

    Methods
    -------
    get_B(key) : ndarray
        Get Bag with name 'key'.
    get_x(n ,key=None) : ndarray
        Get one instance.
    add_x(self, key, x, y, z=None,  UPDATE=True) :
        Add one instance.
    del_x(self, n, key=None, UPDATE=True) :
        Delete one instance.
    del_B(self, key) :
        Delete a whole bag.
    del_singleEntires():
        Delete all bags which do not at least have 2 instances.
    save(self, path) :
        Save all MIL data.
    load(self, path) :
        Load all three  MIL data. files.
    """

    def __init__(self, name, CACHE=True):
        """
        Initialize.

        Parameters
        ----------
        name : string
            The name of the dataset.
        CACHE : bool, optional
            Set if data shall be cached for faster access.
        """
        self.name = name
        self._keys = []  # List of sorted keys of the dicts
        self._N_X = 0  # The number of instances stored
        self._N_B = 0  # The number of bags stored
        self._N_D = 0  # The dimension of the feature vectors
        self._N_b = None
        self._X = {}  # Dictionary with data per bags
        self._X_arr = None
        self._z = {}  # Dictionary with bag labels
        self._z_arr = None
        self._y = {}  # Dictionary with instance labels per bag
        self._y_arr = None
        self._pos_b = {}  # Store start and stop position of one bag in X_arr (start, stop)
        self._pos_x = {}  # Store instance positions for fast access (key, n)
        self._CACHE = CACHE

    def __getattr__(self, item):
        if item == "keys":
            return self._keys
        elif item == "N_X":
            return self._N_X
        elif item == "N_B":
            return self._N_B
        elif item == "N_D":
            return self._N_D
        elif item == "N_b":
            if self._CACHE:
                if self._N_b is None:
                    self._cache()
                return np.array(self._N_b, dtype=np.int)
            else:
                N_b_tmp = []
                for key in self._keys:
                    N_b_tmp.append(len(self._X[key]))
                return np.array(N_b_tmp, dtype=np.int)
        elif item == "X":
            if self._CACHE:
                if self._X_arr is None:
                    self._cache()
                return self._X_arr
            else:
                return dict_to_arr(self._X, self._keys)[0]
        elif item == "z":
            if self._CACHE:
                if self._z_arr is None:
                    self._cache()
                return self._z_arr
            else:
                return dict_to_arr(self._z, self._keys)[0]
        elif item == "y":
            if self._CACHE:
                if self._y_arr is None:
                    self._cache()
                return self._y_arr
            else:
                return dict_to_arr(self._y, self._keys)[0]
        elif item == "pos_b":
            return self._pos_b
        elif item == "pos_x":
            return self._pos_x
        else:
            raise AttributeError("Attribute not found!")

    def get_B(self, key):
        """
        Get Bag with name 'key'.

        Parameters
        ----------
        key : key
            The name of which bag to return.

        Returns
        --------
        ndarray
            The MilArray of one bag.
        """
        return np.array(self._X[key]), np.array(self._z[key]), np.array(
            self._y[key])

    def get_x(self, n, key=None):
        """
        Return one instance of given position.

        Parameters
        ----------
        n : int
            The position of the wanted instance.
        key : key, optional
            If given, x is looked for at the nth position in that bag. Otherwise
            n is interpreted as the global position
        Returns
        --------
        tuple(ndarray, int)
            The features and label of one instance.
        """
        if key is None:
            return self._X[self._pos_x[n][0]][self._pos_x[n][1]], \
            self._y[self._pos_x[n][0]][self._pos_x[n][1]]
        else:
            return self._X[key][n], self._y[key][n]

    def add_x(self, key, x, z, y=None, UPDATE=True):
        """
        Add one instance.

        Parameters
        ----------
        key : key
            The key of which bag to add to.
        x : array_like
            Instance to add. The shape must mach the shape of other instances.
        z : int
            Bag label to add. If non is given, the instance label will be used
            for new bags.
        y : int, optional
            Instance label to add.
        UPDATE : bool, optional
            If True, bag labels will be recalculated to fit MIL constraints.
        """
        # make data to have correct dimensions
        x = np.atleast_2d(np.array(x))
        z = np.atleast_1d(np.array(z))
        if y is None:
            y = np.nan
        y = np.atleast_1d(np.array(y))

        if len(x.shape) > 2:
            raise ValueError(
                "The instance features to add have inappropriate shape.")
        if len(z.shape) > 1:
            raise ValueError("The bag label to add has inappropriate shape.")
        if len(y.shape) > 1:
            raise ValueError(
                "The instance label to add has inappropriate shape.")
        if len(y) != len(x):
            raise ValueError(
                "Number of labels does not match number of features.")

        self._N_X += 1
        self._N_D = x.shape[-1]

        if key in self._keys:  # We add to an existing bag
            self._X[key] = np.concatenate((self._X[key], x), axis=0)
            self._y[key] = np.concatenate((self._y[key], y), axis=0)
            self._z[key] = np.atleast_1d(np.array(z))
        else:  # we create a new bag
            self._N_B += 1
            self._keys.append(key)
            self._X[key] = x
            self._y[key] = y
            self._z[key] = z
        if UPDATE:
            if not np.any(np.isnan(self._y[key])):
                self._z[key] = np.atleast_1d(np.max(self._y[key]))
            else:
                raise ValueError(
                    "Can not update due to unknown instance label.")

        self._clear()
        self._sort_keys()
        self._map()

    def del_x(self, n, key=None, UPDATE=True):
        """
        Delete one instance.

        Parameters
        ----------
        n : int
            The overall postion of the instance to delete.
        key : key, optional
            If given, x is deleted at the nth position in that bag. Otherwise
            n is interpreted as the global position.
        UPDATE : bool, optional
            If True, bag labels will be recalculated to fit MIL constraints.
        """

        if key is None:
            key = self._pos_x[n][0]
            n = self._pos_x[n][1]

        if len(self._X[key]) == 1:  # We have to delete a bag
            self.del_B(key)
        else:
            self._X[key] = np.delete(self._X[key], n, 0)
            self._y[key] = np.delete(self._y[key], n, 0)
            if UPDATE:
                self._z[key] = np.atleast_1d(np.max(self._y[key]))

            self._N_X -= 1
            self._clear()
            self._sort_keys()
            self._map()

    def del_B(self, key):
        """
        Delete a whole bag.

        Parameters
        ----------
        key : key
            The key of which bag to delete.
        """
        self._N_X -= len(self._X[key])

        del self._X[key]
        del self._y[key]
        del self._z[key]
        self._N_B -= 1
        self._keys.remove(key)

        self._clear()
        self._sort_keys()
        self._map()

    def del_feat(self, l):
        """
        Delete the lth feature in all bags:
        """
        if self.N_D < l:
            raise ValueError("Feature does nor exist!")

        for key in self.keys:  # got through all bags and delete feature
            self._X[key] = np.delete(self._X[key], l, axis=1)

    def del_singleEntires(self):
        """
        Delete all bags which do not at least have 2 instances.
        """
        to_delete = list(compress(self._keys, self.N_b == 1))
        for key in to_delete:
            self.del_B(key)
        to_delete = list(compress(self._keys, self.N_b == 0))
        for key in to_delete:
            self.del_B(key)

    def save(self, path):
        """
        Save all MIL data. Three .json files will be generated: One for the
        data, one for the instance labels and one for the bag labels.

        Parameters
        ----------
        path : string
            The path where to save.
        """
        save_dict(self._X, self.name + "_x", path)
        save_dict(self._y, self.name + "_y", path)
        save_dict(self._z, self.name + "_z", path)

    def load(self, path):
        """
        Load all three  MIL data. files.

        Parameters
        ----------
        path : string
            The path where to find the files.
        """

        self._X = load_dict(self.name + "_x", path)
        self._y = load_dict(self.name + "_y", path)
        self._z = load_dict(self.name + "_z", path)

        self._clear()
        self._sort_keys()
        self._N_B = len(self._keys)
        self._N_D = self._X[self._keys[0]].shape[-1]
        self._N_X = 0
        for k, v in self._X.items():
            self._N_X += len(v)
        self._map()

    def _sort_keys(self):
        """
        Sort all keys
        """
        if self._X.keys() != self._z.keys() and self._z.keys() != self._y.keys():
            raise KeyError("Key Error")
        self._keys = sorted(self._X.keys())

    def _cache(self):
        """
        Cache intermediate results for easy acces
        """
        self._clear()
        self._X_arr = dict_to_arr(self._X, self._keys)[0]
        self._y_arr = dict_to_arr(self._y, self._keys)[0]
        self._z_arr = dict_to_arr(self._z, self._keys)[0]
        self._N_b = []
        for key in self._keys:
            self._N_b.append(len(self._X[key]))

    def _clear(self):
        """
        Clear cached data
        """
        self._N_b = None
        self._X_arr = None
        self._y_arr = None
        self._z_arr = None

    def _map(self):
        """
        Map global position of one instance to bag key and bag position.
        Map bag to global start and stop position.
        """
        self._pos_x = {}
        l = 0  # count along instance in bag
        b = 0  # count along bags
        for i in range(self._N_X):
            if l < len(self._X[self._keys[b]]):
                self._pos_x[i] = [self._keys[b], l]
                l += 1
            else:
                b += 1
                l = 0
                self._pos_x[i] = [self._keys[b], l]
                l += 1

        self._pos_b = {}
        counter = 0
        for key in self._keys:
            n = len(self._X[key])
            self._pos_b[key] = (counter, counter + n)
            counter += n


def dict_to_arr(mydict, keys=None):
    """
    Convert a MilDictionary into a 2D ndarray.

    Parameters
    ----------
    mydict : dict
        A MilDictionary has the bag names as keys and for each key an
        instance dataset as a 2D ndarray.
    keys : list, optional
        Only entries of the given keys will be included into the output array. The
        row ordering will correspond to the key ordering. If None is given, all
        keys will be sorted using python's sort functionality.

    Returns
    -------
    myarr : ndarray
        The MilArray. Each row corresponds to one instance.
    N_b : ndarray
        The number of instances corresponding to each bag.
    keys : list
        Only entries of the given keys will be included into the output array. The
        row ordering will correspond to the key ordering. If None is given, all
        keys will be sorted using python's sort functionality.

    Exceptions
    ----------
    MilError :
        Raised if the MilDictionary has instances with number of dimensions > 2.
    """
    myarr = None
    N_b = None

    if keys is None:  # sort keys if not given
        keys = sorted(mydict.keys())

    for k, key in enumerate(keys):
        if len(mydict[key].shape) > 2:
            raise ValueError(
                "The MilDictionary has instances of inappropriate shape")
        if k == 0:  # we initialize the output array
            myarr = np.array(mydict[key])
            N_b = np.atleast_1d(np.array([len(mydict[key])], dtype=np.int))
        else:
            myarr = np.concatenate((myarr, mydict[key]))
            N_b = np.concatenate(
                (N_b, np.array([len(mydict[key])], dtype=np.int)), axis=0)
    return myarr, N_b, keys


def arr_to_dict(myarr, keys, N_b):
    """
    Convert an MilArray into an MilDictionary.

    Parameters
    ----------
    myarr : ndarray
        The MilArray. Each row corresponds to one instance.
    keys : list
        keys for each bag to construct the dictionary from. The ordering of the
         keys will define which array entry will belong to which bag.
    N_b : ndarray
        The number of instances corresponding to each bag.

    Returns
    ----------
    mydict : dict
        A MilDictionary has the bag names as keys and for each key an
        instance dataset as a 2D ndarray.

    Exceptions
    ----------
    MilError :
        Raised if the number of keys and entries in bag instances do not match.
        Raised if the number of instances in the given array do not match the
        number of instances in the instance number list.
    """

    if len(keys) != len(N_b):
        raise KeyError("Unequal number of keys and bag instance.")
    if np.sum(N_b) != len(myarr):
        raise ValueError(
            "Unequal number of instances in array and bag instance number.")

    mydict = dict.fromkeys(keys)

    n = 0
    for k, key in enumerate(keys):
        mydict[key] = np.atleast_1d(
            np.array(myarr[n:n + N_b[k]], dtype=myarr.dtype))
        n += N_b[k]
    return mydict


def find_pos(mydict, n, keys):
    """
    Find the bag and local position of the overall nth element in one set of a
    MilDictionary.

    Parameters
    ----------
    mydict : dict
        A MilDictionary has the bag names as keys and for each key an
        instance dataset as a 2D ndarray.
    n : int
        The global position of the wanted instance.
    keys : list
        keys for each bag to construct the dict from. The ordering of the
         keys will define which array entry will belong to which bag.

    Returns
    ----------
    key : key
        The bag key where to find instance n.
    l : int
        The row of the instance in corresponding bag ndarray.

    """
    n_ = -1
    for k, key in enumerate(keys):
        n_ += len(mydict[key])

        if n_ == n:
            return (key, len(mydict[key]) - 1)
        elif n_ > n:
            return (key, n - n_ + len(mydict[key]) - 1)
        else:
            continue


def save_dict(mydict, fname, path=None):
    """
    Save the MilDictionary to a .json file.

    Parameters
    ----------
    mydict : dict
        A MilDictionary has the bag names as keys and for each key an
        instance dataset as a 2D ndarray.
    fname: string
        The name of the resulting .json file
    path : string, optional
        The path where to store the datafile.
    """
    fname = prepareSaving(fname, path=path, extension=".json")
    pd.Series(mydict).to_json(fname)


def load_dict(fname, path=None):
    """
    Load the MilDictionary from a .json file.

    Parameters
    ----------
    fname: string
        The name of the resulting .json file
    path : string, optional
        The path where to store the datafile.

    Returns
    -------
    mydict : dict
        A MilDictionary has the bag names as keys and for each key an
        instance dataset as a 2D ndarray.
    """
    fname = prepareLoading(fname, path=path, extension=".json")
    mydict = pd.read_json(fname, typ='series').to_dict()
    for key, value in mydict.items():
        mydict[key] = np.array(value)
    return mydict
