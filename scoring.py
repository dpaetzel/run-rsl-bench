import numpy as np


def intersection(l1, u1, l2, u2):
    """
    Computes the intersection between two intervals.

    Parameters
    ----------
    int1, int2: array
        An array of shape (2, `dimension`). I.e. `int1[0]` is the lower
        bound and `int1[1]` is the upper bound of the interval.

    Returns
    -------
    array or None
        If the intervals do not overlap, return `None`. Otherwise return the
        intersection interval.
    """
    l = np.max([l1, l2], axis=0)
    u = np.min([u1, u2], axis=0)

    if np.any(u < l):
        return None
    else:
        return l, u


def volume(l, u):
    """
    Computes the volume of the given interval.
    """
    return np.prod(u - l)


def subsethood(l1, u1, l2, u2):
    intersect = intersection(l1=l1, u1=u1, l2=l2, u2=u2)
    # If not intersecting, subsethood is 0.
    if intersect is None:
        return 0.0
    # If intersecting …
    else:
        # … and the first interval is degenerate that interval is still fully
        # contained in the second and subsethood is 1.0.
        v = volume(l1, u1)
        if v == 0:
            return 1.0
        else:
            l, u = intersect
            return volume(l, u) / v


def interval_similarity_mean(l1, u1, l2, u2):
    """
    Robust interval similarity metric proposed by (Huidobro et al., 2022).
    """
    ssh1 = subsethood(l1=l1, u1=u1, l2=l2, u2=u2)
    ssh2 = subsethood(l1=l2, u1=u2, l2=l1, u2=u1)
    return (ssh1 + ssh2) / 2.0


def similarities(lowers1, uppers1, lowers2, uppers2):
    """
    For each model interval in the first group, compute pairwise similarity
    to all intervals in the second group.

    Returns
    -------
    array of shape (len(lowers1), len(lowers2))
    """
    K1 = len(lowers1)
    K2 = len(lowers2)

    similarity = np.full((K1, K2), -1.0, dtype=float)
    for i in range(K1):
        for j in range(K2):
            similarity[i, j] = interval_similarity_mean(l1=lowers1[i],
                                                        u1=uppers1[i],
                                                        l2=lowers2[j],
                                                        u2=uppers2[j])
    return similarity


def scores(lowers, uppers, lowers_true, uppers_true):
    """
    Ground truth–centric score.

    For each ground truth interval, compute similarities to all model intervals.
    The overall score is a set of numbers, one per ground truth interval (the
    highest score that was achieved for that).

    Returns
    -------
    list of length `len(lowers_true)`
        List of scores, one per ground truth interval.
    """

    similarity = similarities(lowers1=lowers,
                              uppers1=uppers,
                              lowers2=lowers_true,
                              uppers2=uppers_true)

    return _scores(similarities)


def _scores(similarities):
    similarities = similarities.copy()

    similarities[similarities == -1.0] = 0.0

    # Scores are composed of the highest similarity scores achieved for each
    # ground truth interval.
    scores = np.max(similarities, axis=0)

    return scores
