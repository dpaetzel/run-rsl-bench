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


def similarities(lowers, uppers, lowers_true, uppers_true):
    K_true = len(lowers_true)
    K = len(lowers)

    vols_overlap = np.full((K_true, K), -1.0, dtype=float)
    similarity = np.full((K_true, K), -1.0, dtype=float)
    for i in range(K):
        for j in range(K_true):
            lu = intersection(l1=lowers[i],
                              u1=uppers[i],
                              l2=lowers_true[j],
                              u2=uppers_true[j])
            if lu is None:
                vols_overlap[j, i] = 0.0
            else:
                vols_overlap[j, i] = volume(*lu)
            # Note that we do not use vols_overlap currently.

            sim = interval_similarity_mean(l1=lowers[i],
                                           u1=uppers[i],
                                           l2=lowers_true[j],
                                           u2=uppers_true[j])
            similarity[j, i] = sim
    return similarity


def score(lowers, uppers, lowers_true, uppers_true):

    similarity = similarities(lowers=lowers,
                              uppers=uppers,
                              lowers_true=lowers_true,
                              uppers_true=uppers_true)

    similarity[similarity == -1.0] = 0.0

    # The score of a solution rule is the highest similarity score value it
    # received.
    scores = np.max(similarity, axis=0)

    return scores
