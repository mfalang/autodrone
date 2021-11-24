import numpy as np

def dehomogenize(v):
    """
    Dehomogenize v by dividing each element by last element. Works with vectors
    (shape (n,) ) or matrices (shape (n,N) ).
    Args:
        v: homogeneous vector/matrix
    Return:
        u: dehomogenized vector without 1 as last element in each column.
    """
    return homogeneous_normalize(v)[:-1]

def homogeneous_normalize(v):
    return v / v[-1]

def estimate_H_linear(xy_, XY_):
    """
    Estimate the homography between planar points in xy and XY. xy (lower-case)
    are the image coordinates with origin in the center of the image (i.e.
    result from xy = Kinv * uv). XY are the coplanar points in the world.
    Args:
        xy: image coordinates with origin = center, shape (2, n) or (3, n)
            (latter if homogeneous coordinates)
        XY: coplanar points in the world, shape (2, n) or (3, n) (latter if
            homogeneous coorindates)
    Returns:
        H: homography between xy and XY s.t. xy = H * XY
    """
    if xy_.shape[0] == 3:
        xy = dehomogenize(xy_)
    else:
        xy = xy_.copy()

    if XY_.shape[0] == 3:
        XY = dehomogenize(XY_)
    else:
        XY = XY_.copy()

    n = XY.shape[1]

    A = np.zeros((2*n, 9))
    for i in range(n):
        xi, yi = xy[0, i], xy[1, i]
        Xi, Yi = XY[0, i], XY[1, i]

        Ai = np.array([
            [Xi, Yi, 1, 0, 0, 0, -Xi*xi, -Yi*xi, -xi],
            [0, 0, 0, Xi, Yi, 1, -Xi*yi, -Yi*yi, -yi]
        ])

        A[2*i:2*i+2, :] = Ai

    U, S, VT = np.linalg.svd(A)

    H = VT[-1].reshape((3, 3))  # last column of V.T

    return H

def closest_rotation_matrix(Q):
    U, S, VT = np.linalg.svd(Q)
    R = U @ VT
    return R

def decompose_H(H, best_approx=True):
    # best_approx: if True, use Zhangs method of approximating by minimizing
    # the frobenius norm.
    k = np.linalg.norm(H[:, 0])
    r1 = H[:, 0] / k
    r2 = H[:, 1] / k
    r3_pos = np.cross(r1, r2)
    r3_neg = np.cross(-r1, -r2)
    t = H[:, 2] / k

    R_pos = np.hstack((r1[:, None], r2[:, None], r3_pos[:, None]))
    R_neg = np.hstack((-r1[:, None], -r2[:, None], r3_neg[:, None]))

    if best_approx:
        R_pos = closest_rotation_matrix(R_pos)
        R_neg = closest_rotation_matrix(R_neg)

    # index [:, None] gives extra dimensions so vectors are (3,1) and not (3,)
    T1 = np.hstack((R_pos, t[:, None]))
    T1 = np.vstack((T1, np.array([0, 0, 0, 1])))

    T2 = np.hstack((R_neg, -t[:, None]))
    T2 = np.vstack((T2, np.array([0, 0, 0, 1])))

    return T1, T2

def choose_decomposition(T1, T2, XYZ1):
    # here, XYZ1 = XY01 because Z = 0
    c1 = T1 @ XYZ1
    c2 = T2 @ XYZ1

    if np.all(c1[2, :] >= 0):
        T = T1
    elif np.all(c2[2, :] >= 0):
        T = T2
    else:
        raise ValueError("Neither T1 nor T2 gives physically plausable pose.")

    return T

def reproject_using_Rt(K, R, t, XYZ1):
    """
    Reprojects the points in XYZ1 using the rigid-body transformation R and t.
    Returns:
        Dehomogenized coordinates uv in pixel coords.
    """
    # projection matrix
    P = np.hstack((
        np.eye(3), np.zeros((3, 1))
    ))

    T = create_T_from_Rt(R, t)
    uv_homgen = K @ P @ T @ XYZ1
    uv = dehomogenize(uv_homgen)
    return uv


def create_T_from_Rt(R, t):
    T = np.hstack((R, t[:, None]))
    T = np.vstack((T, np.array([0, 0, 0, 1])))
    return T

def levenberg_marquardt(
    residualsfun,
    p0,
    num_iterations=100,
    termination_tolerance=0.001,
    finite_difference_epsilon=1e-5,
    debug=False
):
    p = p0.copy()
    mu = None
    iterations = 0

    def objfun(p):
        r = residualsfun(p)
        return r.T @ r

    prev = 0
    while iterations < num_iterations:
        if debug:
            print("Iteration number", iterations)
        J = finite_difference(residualsfun, p, finite_difference_epsilon)

        JTJ = J.T @ J
        r = residualsfun(p)
        JTr = J.T @ r

        # Initialize mu if not already initialized
        if mu is None:
            mu = 1e-3 * np.max(np.diag(JTJ))
            if debug:
                print("  LM: Initialized mu to:", mu)

        # Update rule for damping parameter mu
        delta = np.ones((JTJ.shape[0],))
        while np.linalg.norm(delta) > 1e-10:
            delta = np.linalg.solve(JTJ + mu*np.eye(JTJ.shape[0]), -JTr)
            if objfun(p + delta) < objfun(p):
                mu /= 3
                if debug:
                    print("  LM: Reduced mu:", mu)
                break
            else:
                mu *= 2
                if debug:
                    print("  LM: Increased mu:", mu)
        else:
            # will run only if the while-termination criterion is met
            if debug:
                print("  LM: Delta small:", np.linalg.norm(delta))

        # Update estimate
        p = p + delta
        iterations += 1
        if debug:
            print(f"-> E(p) = {objfun(p)}")

        curr = objfun(p)
        if np.abs(curr - prev) <= termination_tolerance:
            if debug:
                print("  LM: Stop because below termination tolerance.")
            break
        prev = curr

    return p, iterations

def finite_difference(f, x, epsilon):
    """
    Args:
        f: function, returns n x 1, shape = (n,)
        x: vector, m x 1, shape = (m,)
        epsilon: finite difference epsilon
    Return:
        J = df/fx, n x m
    """
    y = f(x)
    n = y.shape[0]

    m = x.shape[0]
    J = np.zeros((n, m))
    for i in range(m):
        e = np.zeros((m,))
        e[i] = epsilon
        Ji = f(x + e) - f(x - e)
        Ji /= 2*epsilon
        J[:, i] = Ji

    return J