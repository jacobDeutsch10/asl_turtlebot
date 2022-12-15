import numpy as np
import scipy.interpolate

def compute_smoothed_traj(path, V_des, k, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        k (int): The degree of the spline fit.
            For this assignment, k should equal 3 (see documentation for
            scipy.interpolate.splrep)
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        t_smoothed (np.array [N]): Associated trajectory times
        traj_smoothed (np.array [N,7]): Smoothed trajectory
    Hint: Use splrep and splev from scipy.interpolate
    """
    ########## Code starts here ##########
    dts_nom = [
      np.linalg.norm(np.array(path[i])- path[i-1]) /V_des  for i in range(1, len(path))
    ]

    ts = np.array([0] + np.cumsum(dts_nom).tolist())

   
    t_smoothed = np.arange(0, ts[-1], dt, dtype=float)


    tck_x = scipy.interpolate.splrep(ts, np.array(path)[:,0], w=None, k=k, s=alpha)
    x_d = scipy.interpolate.splev(t_smoothed, tck_x, der=0)
    xd_d = scipy.interpolate.splev(t_smoothed, tck_x, der=1)
    xdd_d = scipy.interpolate.splev(t_smoothed, tck_x, der=2)


    tck_y = scipy.interpolate.splrep(ts, np.array(path)[:,1], k=k, s=alpha)
    y_d = scipy.interpolate.splev(t_smoothed, tck_y, der=0)
    yd_d = scipy.interpolate.splev(t_smoothed, tck_y, der=1)
    ydd_d = scipy.interpolate.splev(t_smoothed, tck_y, der=2)
    theta_d = np.arctan2(yd_d, xd_d)
    ########## Code ends here ##########
    traj_smoothed = np.stack([x_d, y_d, theta_d, xd_d, yd_d, xdd_d, ydd_d]).transpose()

    return t_smoothed, traj_smoothed
