import math
import PIL
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple
import matplotlib.pyplot as plt
import pdb

from pyquaternion import Quaternion
from shapely import affinity
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.map_expansion.arcline_path_utils import discretize_lane, ArcLinePath
from shapely.geometry import LineString
from skimage.draw import polygon
from math import cos, sin
from stp3.utils.Optnode_obs_unbatched import OPTNode_obs
from stp3.utils.Optnode import OPTNode, bernstein_coeff_order10_new
from stp3.utils.Optnode_naive import OPTNode_batched
from stp3.utils.Optnode_waypoint import OPTNode_waypoint

import pdb

from typing import Iterable, List, Sequence, Set, Tuple

n_present = 3

W = 1.85
H = 4.084
bbx = np.array([-50.0 + 0.5/2.0, -50.0 + 0.5/2.0])
ddx = np.array([0.5, 0.5])
bx = np.array([-50.0 + 0.5/2.0, -50.0 + 0.5/2.0])
dx = np.array([0.5, 0.5])

def get_rot(theta):
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    return rot

def get_origin_points(lambda_=0, theta=0):
    pts = np.array([
        [-H / 2. + 0.5 - lambda_, W / 2. + lambda_],
        [H / 2. + 0.5 + lambda_, W / 2. + lambda_],
        [H / 2. + 0.5 + lambda_, -W / 2. - lambda_],
        [-H / 2. + 0.5 - lambda_, -W / 2. - lambda_],
    ])
    rot = get_rot(theta)
    pts = np.dot(rot, pts.T).T
    pts = (pts - bx) / (dx)    
    pts[:, [0, 1]] = pts[:, [1, 0]]
    rr , cc = polygon(pts[:,1], pts[:,0])
    rc = np.concatenate([rr[:,None], cc[:,None]], axis=-1)
    return rc # (27,2)

def get_points(self, pts, lambda_=0, bev_dimension=None):
    '''
    pts: np array (n_future, 2)
    return:
    List[ torch.Tensor<int> (B, N, n_future), torch.Tensor<int> (B, N, n_future)]
    '''
    n_future, _ = pts.shape
    rc = get_origin_points(lambda_)
    if len(pts) > 1:
        theta = pts[1:] - pts[:-1]
    else:
        theta = 0
    pts = pts + rc
    pts = torch.tensor(pts)

    rr = pts[:,0].long()
    rr = torch.clamp(rr, 0, bev_dimension[0] - 1)

    cc = pts[:,1].long()
    cc = torch.clamp(cc, 0, bev_dimension[1] - 1)

    return rr, cc

def extract_trajs(centerlines, trj, seg_prediction=None, hdmap_prediction=None, nvar=11, viz=False, debugg=False, device='cpu', labels=None, griddim=None, problem=None, avoid_obs=False, obs=None, a_obs=1, b_obs=1, ind="0", bev_dimension=None):
    save_device = device
    device = 'cpu'
    save_obs = obs
    bbx = np.array([-50.0 + 0.5/2.0, -50.0 + 0.5/2.0])
    ddx = np.array([0.5, 0.5])
    if griddim == None:
        X_MAX = 50
        X_MIN = 0
        NUM_X = 30
        Y_MAX = 5
        Y_MIN = -5
        NUM_Y = 5
    else:
        X_MIN, X_MAX, NUM_X, Y_MIN, Y_MAX, NUM_Y = griddim


    # get trajectories
    x = torch.linspace(X_MIN, X_MAX, NUM_X).to(device)
    y = torch.linspace(Y_MIN, Y_MAX, NUM_Y).to(device)
    xv, yv = torch.meshgrid(x, y)
    print(device)
    grid = torch.dstack((xv, yv)).to(device)
    ccc = centerlines.detach().cpu().numpy()
    cx, _ = interp_polyline_by_fixed_waypt_interval_np(ccc, 0.1)
    cx = extend_both_sides(cx, resolution=0.1, extension=500)
    cx = cx * ddx
    cx += bbx
    cx[:, [1, 0]] = cx[:, [0, 1]]
    cx[:, :1] = cx[:, :1] * -1
    cx = torch.tensor(cx).to(device)

    trr = trj.squeeze().detach().cpu()
    ref_line_ = torch.tensor(cx.reshape(1, cx.shape[0], 2))
    ref_line = torch.zeros((1, ref_line_.shape[1] - 1, 3))
    ref_line[:, :, :2] = ref_line_[:, 1:] 
    off = ref_line_[:, 1:] - ref_line_[:,:-1]
    ref_line[:, :, 2] = torch.atan2(off[:, :, 1], off[:, :, 0])
    ref_line = ref_line.float().to(device)
    trj = trj.float().to(device)

    test_ref_line=ref_line[:, 100:-100]
    sdd = project_to_frenet_frame(test_ref_line,  ref_line)

    ssd = project_to_frenet_frame(trj, ref_line) # (1, 7, 2)


    cx = interp_arc_np(2000, cx.detach().cpu().numpy())
    cx = extend_both_sides(cx, resolution=0.1, extension=50)
    cx = torch.tensor(cx).to(device)
    ssd = project_to_frenet_frame(trj, ref_line) # (1, 7, 2)
    s_offset = ssd[:,2,0]
    d_offset = ssd[0,2,1]
    vel = ssd[0,2] - ssd[0,1]
    vel_prev = ssd[0,1] - ssd[0, 0]
    theta = torch.atan2(vel[1], vel[0]) # to make angle 0 wrt ego-agent
    theta_prev = torch.atan2(vel_prev[1], vel_prev[0])
    psidot_init = (theta - theta_prev)/0.5
    psi_init = theta
    v_init = torch.linalg.norm(vel)/0.5
    y_init = 0
    x_init = 0
    rot_inv = torch.tensor([[cos(-theta), -sin(-theta)], [sin(-theta), cos(-theta)]]).to(device)
    rot = torch.tensor([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]).to(device)


    # optimized smapling
    grid_ref = grid.reshape(1, NUM_X * NUM_Y, 2)
    s_offset = ssd[:,2,0]
    grid_ref = grid_ref + torch.tensor([s_offset, 0])
    grid_ref_cartesian = project_to_cartesian_frame(grid_ref, ref_line) # in cartesian frame
    grid_ref_cartesian = (grid_ref_cartesian[0, :, :2])/ ddx
    grid_ref_cartesian[ :, :1] = grid_ref_cartesian[:, :1] * -1
    grid_ref_cartesian[:, [0, 1]] = grid_ref_cartesian[:, [1, 0]]
    # pdb.set_trace()

    rectangle = get_origin_points()
    grid_ref_cartesian = grid_ref_cartesian.unsqueeze(1) + rectangle
    gc = grid_ref_cartesian.long()
    hdmap = hdmap_prediction[0, 1]
    seg = seg_prediction[0, -1, 0]
    map_vals = hdmap[gc[:,:,0], gc[:,:,1]].sum(-1)
    seg_vals = seg[gc[:,:,0], gc[:,:,1]].sum(-1)

    cutoff = 0
    take_map = (map_vals >= (len(rectangle) - 1 - cutoff)) # do not take any point if any portion outside hdmap prediction

    seg_map = (seg_vals == 0) # do not take any point if any portion intersects with segmentation prediction

    final_grid = []
    grid_ref = grid.reshape(1, NUM_X * NUM_Y, 2)
    for i in range(len(grid_ref[0])):
        if take_map[i] and seg_map[i]:
            print(grid_ref[0][i])
            final_grid.append(grid_ref[0][i].detach().cpu().numpy())
    final_grid = np.array(final_grid)
    grid = final_grid


    # translate
    sdd_frame = sdd -  torch.tensor([s_offset, d_offset]).to(device)
    ssd_frame = ssd -  torch.tensor([s_offset, d_offset]).to(device)
    eps_frame = torch.tensor(grid).to(device) - torch.tensor([0, d_offset]).to(device) # (20, 7, 2)

    # rotate
    sdd_frame[0] = torch.matmul(rot_inv, sdd_frame[0].T).T # (1, M, 2)
    ssd_frame = torch.matmul(rot_inv, ssd_frame[0].T).T # ()
    # eps_frame = eps_frame.permute(1, 2, 0) # (7, 2, 20)
    eps_frame = torch.matmul(rot_inv, eps_frame.T).T # (2, 2) x (M, 2) -> (M, 2)

    """
        save params of the optimizer before calling solve
    """
    # obs -> (M, num, 2) : M obstacles for num timesteps
    # avoid obstacles
    num_obs = obs.shape[0]
    num = obs.shape[1]
    # obs = obs.reshape(1, num_obs * num, 2) # (1, M * num, 2)

    # first bring to agent frame
    obs = obs * ddx
    obs += bbx
    obs[:, :, [1, 0]] = obs[:, :, [0, 1]]
    obs[:, :, :1] = obs[:, :, :1] * -1

    ssd_obs = torch.zeros((1, num_obs, num, 2))
    for n in range(num_obs):
        print(obs[n].unsqueeze(0).shape, "OK")
        print(project_to_frenet_frame(obs[n].unsqueeze(0).double().to(device), ref_line.double().to(device)).shape)
        ssd_obs[:, n] = project_to_frenet_frame(obs[n].unsqueeze(0).double().to(device), ref_line.double().to(device)) # (1, M * num, 2)

    # translate
    ssd_obs_frame = ssd_obs[0] - torch.tensor([s_offset, d_offset]).to(device) # (M, num, 2)
    # print(ssd_obs.shape, ssd_obs_frame.shape, " OBS")
    for i in range(num_obs):
        ssd_obs_frame[i] = torch.matmul(rot_inv, ssd_obs_frame[i].float().T).T
    # ssd_obs_frame = ssd_obs_frame[0].permute(1, 2, 0) # (num, 2, M)
    # ssd_obs_frame = torch.matmul(rot_inv, ssd_obs_frame.float()) # (2, 2) x (num, 2, M) -> (num, 2, M)
    # ssd_obs_frame = ssd_obs_frame.reshape(num_obs, num, 2)  # (M, num, 2)
    x_obs = ssd_obs_frame[:, : ,0]
    y_obs = ssd_obs_frame[:, : ,1]
    # x_obs, y_obs = problem.x_obs, problem.y_obs
    obs_frame = torch.dstack((x_obs, y_obs))

    x_eps = eps_frame[:,0].flatten()
    y_eps = eps_frame[:,1].flatten()

    fixed_params = torch.tensor([x_init, y_init, v_init, psi_init, psidot_init]).to(device)
    fixed_params = fixed_params.expand(len(x_eps), 5)
    variable_params = torch.zeros((len(x_eps), 3)).to(device)
    variable_params[:, 0] = torch.tensor(x_eps)
    variable_params[:, 1] = torch.tensor(y_eps)
    variable_params[:, 2] = torch.tensor(-theta)

    """
        prepare x_obs and y_obs
    """
    num_obs = len(obs_frame)
    num_local_obs = 61
    x_obs = np.ones((num_obs, num_local_obs))
    y_obs = np.ones((num_obs, num_local_obs))
    b_inp = np.ones((len(fixed_params), 12))
    for i, ob in enumerate(obs_frame.detach().cpu().numpy()):
        # extrapolating obstacle points to (61, 2)
        x_obs[i][0] = ob[0, 0]
        x_obs[i][-1] = ob[-1, 0]
        y_obs[i][0] = ob[0, 1]
        y_obs[i][-1] = ob[-1, 1]
        pts = interp_arc_np(t=num_local_obs - 2 + 1, points=ob[1:-1])
        pts = pts[1:] # nan at start for some reason
        x_obs[i][1:-1] = pts[:, 0]
        y_obs[i][1:-1] = pts[:, 1]
        if viz:
            plt.plot(ob[:, 0], ob[:, 1])
            plt.scatter(ob[:, 0], ob[:, 1])
    print(x_obs.shape, y_obs.shape)
    for i in range(len(fixed_params)):
        x_init, y_init = fixed_params[i,0], fixed_params[i, 1]
        x_fin, y_fin = variable_params[i, 0], variable_params[i, 1]
        b_inp[i] = np.array([x_init, 0, 0, x_fin, 0, 0, y_init, 0, 0, y_fin, 0, 0])

    """
        prepare Optnode, refine collision bound trajectories
    """
    t_fin = 2
    a_obs = 2.0
    b_obs = 2.0
    # rho_obs = 0.3
    rho_obs = 1.2
    rho_eq = 10.0
    weight_smoothness = 10
    tot_time = np.linspace(0.0, t_fin, num=num_local_obs)
    tot_time_copy = tot_time.reshape(num_local_obs, 1)
    P, Pdot, Pddot = bernstein_coeff_order10_new(10, tot_time_copy[0], tot_time_copy[-1], tot_time_copy)
    A_obs = np.tile(P, (num_obs, 1))
    A_eq = np.vstack((P[0], Pdot[0], Pddot[0], P[-1], Pdot[-1], Pddot[-1]))
    Q_smoothness = np.dot(Pddot.T, Pddot)
    a_obs = 2.0
    b_obs = 2.0
    # rho_obs = 0.3
    rho_obs = 1.2
    rho_eq = 10.0
    weight_smoothness = 10
    batch_size = len(fixed_params)
    x_obs = np.ones((batch_size, num_obs, num_local_obs)) * x_obs
    y_obs = np.ones((batch_size, num_obs, num_local_obs)) * y_obs
    opt_node = OPTNode_batched(P, Pddot, A_eq, A_obs, Q_smoothness, x_obs, y_obs,num=num_local_obs, num_obs=num_obs, nvar=nvar, a_obs=a_obs, b_obs=b_obs, rho_obs=2, rho_eq=10,weight_smoothness=weight_smoothness, maxiter=3000, eps=1e-7, num_tot=num_local_obs*num_obs,batch_size=batch_size)
    lamda_x = torch.zeros(batch_size, nvar, dtype=torch.double)
    lamda_y = torch.zeros(batch_size, nvar, dtype=torch.double)
    sol = opt_node.solve(torch.tensor(b_inp), lamda_x, lamda_y)[0]
    x_pred = torch.matmul(torch.tensor(P), sol[:, :nvar].transpose(0, 1)) # (num_local_obs, batch_size)
    y_pred = torch.matmul(torch.tensor(P), sol[:, nvar:2*nvar].transpose(0, 1))# (num_local_obs, batch_size)
    x_pred = x_pred.T
    y_pred = y_pred.T

    not_to_refine_inds = []
    refine_inds = []
    for i in range(len(x_pred)):
        thetas = torch.atan2(y_pred[1:, i] - y_pred[:-1, i], x_pred[1:, i] - x_pred[:-1, i])
        unique_thetas = len(torch.unique(torch.clamp(thetas - thetas[0], 1e-5, 1e11)))
        if unique_thetas == 1:
            # straight line
            not_to_refine_inds.append(i)
        else:
            refine_inds.append(i)

    x_pred_waypt = x_pred[refine_inds]
    y_pred_waypt = y_pred[refine_inds]
    eps_frame_nonholo = eps_frame[not_to_refine_inds]
    eps_frame_waypt = eps_frame[refine_inds]
    torch.save(x_obs, f"obs/x_obs_{ind}.pt")
    torch.save(y_obs, f"obs/y_obs_{ind}.pt")
    torch.save(obs_frame, f"obs/obs_{ind}.pt")
    torch.save(fixed_params, f"obs/fixed_params_{ind}.pt")
    torch.save(variable_params, f"obs/variable_params_{ind}.pt")

    problem = OPTNode(t_fin=problem.t_fin, num=problem.num, device='cpu')
    x_eps = eps_frame_nonholo[:,0].flatten()
    y_eps = eps_frame_nonholo[:,1].flatten()
    fixed_params = torch.tensor([x_init, y_init, v_init, psi_init, psidot_init]).to(device)
    fixed_params = fixed_params.expand(len(eps_frame_nonholo), 5)
    variable_params = torch.zeros((len(eps_frame_nonholo), 3)).to(device)
    variable_params[:, 0] = torch.tensor(x_eps)
    variable_params[:, 1] = torch.tensor(y_eps)
    variable_params[:, 2] = torch.tensor(-theta)
    P = problem.P
    cc, _ = problem.solve(fixed_params.double(), variable_params.double())
    x_sol = torch.matmul(P, cc.T[:nvar]).T
    y_sol = torch.matmul(P, cc.T[nvar:2*nvar]).T    
    trajs = torch.dstack((x_sol, y_sol))



    """
        use waypoint optimizer to refine obstacle avoided trajectories
    """
    if len(eps_frame_waypt) > 1:
        x_eps = eps_frame_waypt[:,0].flatten()
        y_eps = eps_frame_waypt[:,1].flatten()
        fixed_params = torch.tensor([x_init, y_init, v_init, psi_init, psidot_init]).to(device)
        fixed_params = fixed_params.expand(len(eps_frame_waypt), 5)
        variable_params = torch.zeros((len(eps_frame_waypt), 13)).to(device)
        variable_params[:, 0] = torch.tensor(x_eps)
        variable_params[:, 1] = torch.tensor(y_eps)
        variable_params[:, 2] = torch.tensor(-theta)
        variable_params[:, 3] = torch.tensor(x_pred_waypt[:, 10])
        variable_params[:, 4] = torch.tensor(y_pred_waypt[:, 10])
        variable_params[:, 5] = torch.tensor(x_pred_waypt[:, 20])
        variable_params[:, 6] = torch.tensor(y_pred_waypt[:, 20])
        variable_params[:, 7] = torch.tensor(x_pred_waypt[:, 30])
        variable_params[:, 8] = torch.tensor(y_pred_waypt[:, 30])
        variable_params[:, 9] = torch.tensor(x_pred_waypt[:, 40])
        variable_params[:, 10] = torch.tensor(y_pred_waypt[:, 40])
        variable_params[:, 11] = torch.tensor(x_pred_waypt[:, 50])
        variable_params[:, 12] = torch.tensor(y_pred_waypt[:, 50])
        problem_waypt = OPTNode_waypoint(rho_eq=1.1, num=61, t_fin=t_fin, bernstein_coeff_order10_new=bernstein_coeff_order10_new)
        cc, _  = problem_waypt.solve(fixed_params.double(), variable_params.double())
        x_sol = torch.matmul(P, cc.T[:nvar]).T
        y_sol = torch.matmul(P, cc.T[nvar:2*nvar]).T    
        trajs_waypt = torch.dstack((x_sol, y_sol))
        trajs_waypt = trajs_waypt[:, ::10] # (B, 7, 2)
        trajs = torch.cat((trajs, trajs_waypt), dim=0)

    trajs = trajs.permute(0, 2, 1).float()
    trajs = torch.matmul(rot, trajs)
    trajs = trajs.permute(0, 2, 1)
    trajs = trajs + torch.tensor([s_offset, d_offset]).to(device)

    min_val = 1e11
    min_ind = 0

    dense_trajs = project_to_cartesian_frame(trajs.reshape(1, trajs.shape[0] * trajs.shape[1], 2), ref_line).reshape(1, trajs.shape[0], trajs.shape[1], 2)#(B, N, n_future, 2)  
    return dense_trajs.to(save_device)

def extrapolate_points(points, num_points, num=6):
    """
        num_points 
            -> [1, 2, -1, -1, 5, 6] -> (-1) indicates tracking unavaiable for that timestep.
        points 
            -> [(), (), (), ()] 
    """
    total_points = []
    prev_timestep = 0
    count_unavailable = (np.array(num_points) == -1).sum() # number of points where it is unavailable.
    if not count_unavailable:
        for i in range(num):
            # ? is this possible
            total_points.append([-1e11, -1e11])
    elif count_unavailable == 1:
        for i in range(num):
            total_points.append(points[0])
    else:
        whe = np.where(points != -1)[0]

    dist = np.linalg.norm(points[1:] - points[:-1], axis=1).mean()
    
    slopes = np.zeros_like(num_points)
    cnt = 0
    for i in range(len(num_points) - 1):
        if num_points[i] != -1 and num_points[i + 1] != -1:
            # overwrite slope at these points
            slope[i] = num_points[i + 1] - num_points[i]

    return np.array(total_points)

def extract_obs_from_centerness(labels, n_present=3, num_obs=20, num=6, traj=None):
    """
        instance: (B, 9, 200, 200)
        centerness: (B, 9, 1, 200, 200)
        offset: (B, 9, 2, 200, 200)
        flow: (B, 9, 2, 200, 200)
        traj: (B, 3, 2)

        returns: 
            obs: (num_obs, num, 2) in bev space
    """
    bx = np.array([-50.0 + 0.5/2.0, -50.0 + 0.5/2.0])
    dx = np.array([0.5, 0.5])

    instance_seg = labels['instance'].detach().cpu().numpy()
    centerness = labels['centerness'].detach().cpu().numpy()
    offset = labels['offset'].detach().cpu().numpy()
    flow = labels['flow'].detach().cpu().numpy()
    traj = traj.detach().cpu().numpy() # (1, 3, 2) for B=1
    vel = np.linalg.norm(traj[:, -1] - traj[:, -2])
    traj = extend_both_sides(traj[0], resolution=vel, extension=num,reverse=False) # extend forward by num timesteps

    unique_ids = np.unique(labels['instance'].detach().cpu().numpy())
    instance_map = dict(zip(unique_ids, unique_ids))
    instance_colors = generate_instance_colours(instance_map)

    """
        instance: (6, 200, 200)
        centerness: (6, 1, 200, 200)
        offset: (6, 2, 200, 200)
        flow: (6, 2, 200, 200)
        we use centerness for now
    """
    instance = instance_seg[0, n_present - 1:]
    centerness = centerness[0, n_present - 1:, 0]
    offset = centerness[0, n_present - 1:, 0]
    flow = centerness[0, n_present - 1:, 0]

    instance_trajs = {}
    top_num_obs_ids = []
    for instance_id in unique_ids:
        if instance_id == 0:
            # background
            continue
        # compute across time
        points = []
        time_dict = {}
        num_points = []
        for i in range(0, 6):
            whe = np.where(instance[i] == instance_id)
            if len(whe[0]) == 0:
                # instance not present in scene
                num_points.append(-1)
                continue
            max_num = np.argmax(centerness[i][whe[0], whe[1]])
            points.append([whe[0][max_num], whe[1][max_num]])
            num_points.append(i)
        if len(points) == 0:
            for i in range(6):
                points.append([-1e11, -1e11])
        points = np.array(points)
        if len(points) == 6:
            instance_trajs[instance_id] = points

    x_obs = np.ones((num_obs, num)) * -5e5
    y_obs = np.ones((num_obs, num)) * -5e5
    obs_ids = list(instance_trajs.keys())
    distance_dict = {}
    for obs in obs_ids:
        points = np.array(instance_trajs[obs])
        distance_dict[obs] = np.linalg.norm(traj[0, -num:] - points)
    distance_dict = {k: v for k, v in sorted(distance_dict.items(), key=lambda item: item[1])}    # sort points by distance
    obs_ids = list(distance_dict.keys())
    obs_cnt = 0
    for instance_ids in obs_ids:
        if instance_ids == 0: continue
        points = np.array(instance_trajs[instance_ids])
        try:
            x_obs[obs_cnt] = points[:, 0]
            y_obs[obs_cnt] = points[:, 1]
            obs_cnt = obs_cnt + 1
        except Exception as e:
            print("Exception", e, x_obs.shape, points.shape)
            pass
        if obs_cnt >= num_obs:
            break
    for i in range(len(x_obs)):
        plt.plot(x_obs[i], y_obs[i])
    plt.plot(traj[:, 0], traj[:, 1])
    plt.savefig("output_vis/debug_trj.png")
    plt.clf()
    x_obs = torch.tensor(x_obs)
    y_obs = torch.tensor(y_obs)
    return torch.dstack((x_obs, y_obs)) # (num_obs, num, 2)

def interp_arc(t: int, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    # px, py = eliminate_duplicates_2d(px, py)
    eq_spaced_points = np.linspace(0, 1, t)
    n = px.size
    assert px.size == py.size
    pxy = np.array((px, py)).T  # 2d polyline
    chordlen = np.linalg.norm(np.diff(pxy, axis=0), axis=1)
    chordlen = chordlen / np.sum(chordlen)
    cumarc = np.append(0, np.cumsum(chordlen))
    tbins = np.digitize(eq_spaced_points, cumarc)
    tbins[np.where((tbins <= 0) | (eq_spaced_points <= 0))] = 1
    tbins[np.where((tbins >= n) | (eq_spaced_points >= 1))] = n - 1
    s = np.divide((eq_spaced_points - cumarc[tbins - 1]), chordlen[tbins - 1])
    pt = pxy[tbins - 1, :] + np.multiply((pxy[tbins, :] - pxy[tbins - 1, :]), (np.vstack([s] * 2)).T)
    return pt
    

def get_polyline_length(polyline):
    """Calculate the length of a polyline.
    Args:
        polyline: Numpy array of shape (N,3)
    Returns:
        The length of the polyline as a scalar.
    Raises:
        RuntimeError: If `polyline` doesn't have shape (N,2) or (N,3).
    """
    if polyline.shape[1] not in [2, 3]:
        raise RuntimeError("Polyline must have shape (N,2) or (N,3)")
    offsets = torch.diff(polyline, axis=0)
    return float(torch.linalg.norm(offsets, axis=1).sum())

def interp_polyline_by_fixed_waypt_interval(polyline, waypt_interval):
    """Resample waypoints of a polyline so that waypoints appear roughly at fixed intervals from the start.
    Args:
        polyline: array pf shape (N,2) or (N,3) representing a polyline.
        waypt_interval: space interval between waypoints, in meters.
    Returns:
        interp_polyline: array of shape (N,2) or (N,3) representing a resampled/interpolated polyline.
        num_waypts: number of computed waypoints.
    Raises:
        RuntimeError: If `polyline` doesn't have shape (N,2) or (N,3).
    """
    if polyline.shape[1] not in [2, 3]:
        raise RuntimeError("Polyline must have shape (N,2) or (N,3)")

    # get the total length in meters of the line segment
    len_m = get_polyline_length(polyline)

    # count number of waypoints to get the desired length
    # add one for the extra endpoint
    num_waypts = torch.floor(torch.tensor(len_m / waypt_interval)) + 1
    interp_polyline = interp_arc(t=int(num_waypts.detach().cpu().item()), points=polyline)
    return interp_polyline, num_waypts

def get_polyline_length_np(polyline):
    """Calculate the length of a polyline.
    Args:
        polyline: Numpy array of shape (N,3)
    Returns:
        The length of the polyline as a scalar.
    Raises:
        RuntimeError: If `polyline` doesn't have shape (N,2) or (N,3).
    """
    if polyline.shape[1] not in [2, 3]:
        raise RuntimeError("Polyline must have shape (N,2) or (N,3)")
    offsets = np.diff(polyline, axis=0)
    return float(np.linalg.norm(offsets, axis=1).sum())

def interp_arc_np(t, points):
    """Linearly interpolate equally-spaced points along a polyline, either in 2d or 3d.

    We use a chordal parameterization so that interpolated arc-lengths
    will approximate original polyline chord lengths.
        Ref: M. Floater and T. Surazhsky, Parameterization for curve
            interpolation. 2005.
            https://www.mathworks.com/matlabcentral/fileexchange/34874-interparc

    For the 2d case, we remove duplicate consecutive points, since these have zero
    distance and thus cause division by zero in chord length computation.

    Args:
        t: number of points that will be uniformly interpolated and returned
        points: Numpy array of shape (N,2) or (N,3), representing 2d or 3d-coordinates of the arc.

    Returns:
        Numpy array of shape (N,2)

    Raises:
        ValueError: If `points` is not in R^2 or R^3.
    """
    if points.ndim != 2:
        raise ValueError("Input array must be (N,2) or (N,3) in shape.")

    # the number of points on the curve itself
    n, _ = points.shape

    # equally spaced in arclength -- the number of points that will be uniformly interpolated
    eq_spaced_points = np.linspace(0, 1, t)

    # Compute the chordal arclength of each segment.
    # Compute differences between each x coord, to get the dx's
    # Do the same to get dy's. Then the hypotenuse length is computed as a norm.
    chordlen= np.linalg.norm(np.diff(points, axis=0), axis=1)
    # Normalize the arclengths to a unit total
    chordlen = chordlen / np.sum(chordlen)
    # cumulative arclength

    cumarc = np.zeros(len(chordlen) + 1)
    cumarc[1:] = np.cumsum(chordlen)

    # which interval did each point fall in, in terms of eq_spaced_points? (bin index)
    tbins: NDArrayInt = np.digitize(eq_spaced_points, bins=cumarc).astype(int)

    # #catch any problems at the ends
    tbins[np.where((tbins <= 0) | (eq_spaced_points <= 0))] = 1
    tbins[np.where((tbins >= n) | (eq_spaced_points >= 1))] = n - 1

    s = np.divide((eq_spaced_points - cumarc[tbins - 1]), chordlen[tbins - 1])
    anchors = points[tbins - 1, :]
    # broadcast to scale each row of `points` by a different row of s
    offsets = (points[tbins, :] - points[tbins - 1, :]) * s.reshape(-1, 1)
    points_interp: NDArrayFloat = anchors + offsets

    return points_interp

def interp_polyline_by_fixed_waypt_interval_np(polyline, waypt_interval):
    """Resample waypoints of a polyline so that waypoints appear roughly at fixed intervals from the start.
    Args:
        polyline: array pf shape (N,2) or (N,3) representing a polyline.
        waypt_interval: space interval between waypoints, in meters.
    Returns:
        interp_polyline: array of shape (N,2) or (N,3) representing a resampled/interpolated polyline.
        num_waypts: number of computed waypoints.
    Raises:
        RuntimeError: If `polyline` doesn't have shape (N,2) or (N,3).
    """
    if polyline.shape[1] not in [2, 3]:
        raise RuntimeError("Polyline must have shape (N,2) or (N,3)")

    # get the total length in meters of the line segment
    len_m = get_polyline_length_np(polyline)

    # count number of waypoints to get the desired length
    # add one for the extra endpoint
    num_waypts = math.floor(len_m / waypt_interval) + 1
    interp_polyline = interp_arc_np(t=num_waypts, points=polyline)
    return interp_polyline, num_waypts

def get_polyline_length_np(polyline):
    """Calculate the length of a polyline.
    Args:
        polyline: Numpy array of shape (N,3)
    Returns:
        The length of the polyline as a scalar.
    Raises:
        RuntimeError: If `polyline` doesn't have shape (N,2) or (N,3).
    """
    if polyline.shape[1] not in [2, 3]:
        raise RuntimeError("Polyline must have shape (N,2) or (N,3)")
    offsets = np.diff(polyline, axis=0)
    return float(np.linalg.norm(offsets, axis=1).sum())

def extend_both_sides(oracle_centerline, resolution=1, extension=15, forward=True, reverse=True):
    """
        oracle_centerline: (N, 2)
        resolution (m)
        extension (m)
    """
    x, y = oracle_centerline[:, 0], oracle_centerline[:, 1]
    angle_start = np.arctan2(y[1] - y[0], x[1] - x[0])
    angle_end = np.arctan2(y[-1] - y[-2], x[-1] - x[-2])
    
    x_start = x[0] + np.cos(angle_start) * np.arange(- extension, 0, resolution)
    x_end = x[-1] + np.cos(angle_end) * np.arange(0, 0 + extension, resolution)

    y_start = y[0] + np.sin(angle_start) * np.arange(- extension, 0, resolution)
    y_end = y[-1] + np.sin(angle_end) * np.arange(0, 0 + extension, resolution)

    if forward and reverse:
        x = np.concatenate((x_start, x.reshape(x.shape[0]), x_end))
        y = np.concatenate((y_start, y.reshape(y.shape[0]), y_end))
    elif forward: 
        x = np.concatenate((x.reshape(x.shape[0]), x_end))
        y = np.concatenate((y.reshape(y.shape[0]), y_end))
    elif reverse:
        x = np.concatenate((x_start, x.reshape(x.shape[0])))
        y = np.concatenate((y_start, y.reshape(y.shape[0])))

    return np.dstack((x,y))[0]

def project_to_frenet_frame(traj, ref_line):
    distance_to_ref = torch.cdist(traj[:, :, :2], ref_line[:, :, :2])
    k = torch.argmin(distance_to_ref, dim=-1).view(-1, traj.shape[1], 1).expand(-1, -1, 3)
    ref_points = torch.gather(ref_line, 1, k)
    x_r, y_r, theta_r = ref_points[:, :, 0], ref_points[:, :, 1], ref_points[:, :, 2]
    x, y = traj[:, :, 0], traj[:, :, 1]
    s = 0.1 * (k[:, :, 0])
    l = torch.sign((y-y_r)*torch.cos(theta_r)-(x-x_r)*torch.sin(theta_r)) * torch.sqrt(torch.square(x-x_r)+torch.square(y-y_r))
    sl = torch.stack([s, l], dim=-1)

    return sl

def project_to_cartesian_frame(traj, ref_line):
    k = (10 * traj[:, :, 0]).long()
    k = torch.clip(k, 0, len(ref_line[0, :]) - 1)
    ref_points = torch.gather(ref_line, 1, k.view(-1, traj.shape[1], 1).expand(-1, -1, 3))
    x_r, y_r, theta_r = ref_points[:, :, 0], ref_points[:, :, 1], ref_points[:, :, 2]
    x = x_r - traj[:, :, 1] * torch.sin(theta_r)
    y = y_r + traj[:, :, 1] * torch.cos(theta_r)
    xy = torch.stack([x, y], dim=-1)
    return xy

def denoise(arr, w = 7):
    gt_x, gt_y = arr[:, 0], arr[:, 1]
    # denoising
    w = w
    gt_x_t = []
    gt_y_t = []
    for iq in range(len(gt_x)):
        if iq >= w and iq + w <= len(gt_x):
            gt_x_t.append(np.mean(gt_x[iq: iq + w]))
            gt_y_t.append(np.mean(gt_y[iq: iq + w]))
        elif iq < w:
            okx = np.mean(gt_x[w: w + w])
            gt_x_t.append(gt_x[0] + (okx - gt_x[0]) * (iq) / w)
            oky = np.mean(gt_y[w: w + w])
            gt_y_t.append(gt_y[0] + (oky - gt_y[0]) * (iq) / w)
        else:
            okx = np.mean(gt_x[len(gt_x) - w:len(gt_x) - w  + w])
            oky = np.mean(gt_y[len(gt_x) - w: len(gt_x) - w + w])
            gt_x_t.append(okx + (gt_x[-1] - okx) * (w - (len(gt_x) - iq)) / w)
            gt_y_t.append(oky + (gt_y[-1] - oky) * (w - (len(gt_y) - iq)) / w)                   

    gt_x = gt_x_t
    gt_y = gt_y_t
    print(gt_x, gt_y)
    return np.dstack((gt_x, gt_y))[0]

def generate_instance_colours(instance_map):
    # Most distinct 22 colors (kelly colors from https://stackoverflow.com/questions/470690/how-to-automatically-generate
    # -n-distinct-colors)
    # plus some colours from AD40k
    INSTANCE_COLOURS = np.asarray([
        [0, 0, 0],
        [255, 179, 0],
        [128, 62, 117],
        [255, 104, 0],
        [166, 189, 215],
        [193, 0, 32],
        [206, 162, 98],
        [129, 112, 102],
        [0, 125, 52],
        [246, 118, 142],
        [0, 83, 138],
        [255, 122, 92],
        [83, 55, 122],
        [255, 142, 0],
        [179, 40, 81],
        [244, 200, 0],
        [127, 24, 13],
        [147, 170, 0],
        [89, 51, 21],
        [241, 58, 19],
        [35, 44, 22],
        [112, 224, 255],
        [70, 184, 160],
        [153, 0, 255],
        [71, 255, 0],
        [255, 0, 163],
        [255, 204, 0],
        [0, 255, 235],
        [255, 0, 235],
        [255, 0, 122],
        [255, 245, 0],
        [10, 190, 212],
        [214, 255, 0],
        [0, 204, 255],
        [20, 0, 255],
        [255, 255, 0],
        [0, 153, 255],
        [0, 255, 204],
        [41, 255, 0],
        [173, 0, 255],
        [0, 245, 255],
        [71, 0, 255],
        [0, 255, 184],
        [0, 92, 255],
        [184, 255, 0],
        [255, 214, 0],
        [25, 194, 194],
        [92, 0, 255],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
    ])

    return {instance_id: INSTANCE_COLOURS[global_instance_id % len(INSTANCE_COLOURS)] for
            instance_id, global_instance_id in instance_map.items()
            }


def is_overlapping_lane_seq(lane_seq1: Sequence[int], lane_seq2: Sequence[int]) -> bool:
    """
    Check if the 2 lane sequences are overlapping.
    Overlapping is defined as::
        s1------s2-----------------e1--------e2
    Here lane2 starts somewhere on lane 1 and ends after it, OR::
        s1------s2-----------------e2--------e1
    Here lane2 starts somewhere on lane 1 and ends before it
    Args:
        lane_seq1: list of lane ids
        lane_seq2: list of lane ids
    Returns:
        bool, True if the lane sequences overlap
    """

    if lane_seq2[0] in lane_seq1[1:] and lane_seq1[-1] in lane_seq2[:-1]:
        return True
    elif set(lane_seq2) <= set(lane_seq1):
        return True
    return False

def remove_overlapping_lane_seq(lane_seqs: List[List[int]]) -> List[List[int]]:
    """
    Remove lane sequences which are overlapping to some extent
    Args:
        lane_seqs (list of list of integers): List of list of lane ids (Eg. [[12345, 12346, 12347], [12345, 12348]])
    Returns:
        List of sequence of lane ids (e.g. ``[[12345, 12346, 12347], [12345, 12348]]``)
    """
    redundant_lane_idx: Set[int] = set()
    for i in range(len(lane_seqs)):
        for j in range(len(lane_seqs)):
            if i in redundant_lane_idx or i == j:
                continue
            if is_overlapping_lane_seq(lane_seqs[i], lane_seqs[j]):
                redundant_lane_idx.add(j)

    unique_lane_seqs = [lane_seqs[i] for i in range(len(lane_seqs)) if i not in redundant_lane_idx]
    return unique_lane_seqs

def dfs_incoming(nusc_map, lane_id, dist, threshold, resolution_meters=1):
    if dist > threshold:
        return [[lane_id]]
    else:
        traversed_lanes = []
        parent_lanes = (
                nusc_map.get_incoming_lane_ids(lane_id)
        )            
        if parent_lanes is not None:
            for parent in parent_lanes:
                # centerline = nusc_map.get_outgoing_lane_ids(child)
                try:
                    lane_record = nusc_map.get_arcline_path(parent)
                    centerline = np.array(discretize_lane(lane_record, resolution_meters=resolution_meters))
                    cl_length = LineString(centerline).length
                    curr_lane_ids = dfs_incoming(
                        nusc_map,
                        parent,
                        dist + cl_length,
                        threshold
                    )
                    traversed_lanes.extend(curr_lane_ids)
                except:
                    pass
        if len(traversed_lanes) == 0:
            return [[lane_id]]
        lanes_to_return = []
        for lane_seq in traversed_lanes:
            lanes_to_return.append(lane_seq + [lane_id])
        return lanes_to_return


def dfs(nusc_map, lane_id, dist, threshold, resolution_meters=1):
    if dist > threshold:
        return [[lane_id]]
    else:
        traversed_lanes = []
        child_lanes = (
                nusc_map.get_outgoing_lane_ids(lane_id)
        )            
        if child_lanes is not None:
            for child in child_lanes:
                # centerline = nusc_map.get_outgoing_lane_ids(child)
                try:
                    lane_record = nusc_map.get_arcline_path(child)
                    centerline = np.array(discretize_lane(lane_record, resolution_meters=resolution_meters))
                    cl_length = LineString(centerline).length
                    curr_lane_ids = dfs(
                        nusc_map,
                        child,
                        dist + cl_length,
                        threshold
                    )
                    traversed_lanes.extend(curr_lane_ids)
                except:
                    pass
        if len(traversed_lanes) == 0:
            return [[lane_id]]
        lanes_to_return = []
        for lane_seq in traversed_lanes:
            lanes_to_return.append([lane_id] + lane_seq)
        return lanes_to_return


def resize_and_crop_image(img, resize_dims, crop):
    # Bilinear resizing followed by cropping
    img = img.resize(resize_dims, resample=PIL.Image.BILINEAR)
    img = img.crop(crop)
    return img

def get_patch_coord(patch_box: Tuple[float, float, float, float],
                    patch_angle: float = 0.0) -> Polygon:
    """
    Convert patch_box to shapely Polygon coordinates.
    :param patch_box: Patch box defined as [x_center, y_center, height, width].
    :param patch_angle: Patch orientation in degrees.
    :return: Box Polygon for patch_box.
    """
    patch_x, patch_y, patch_h, patch_w = patch_box

    x_min = patch_x - patch_w / 2.0
    y_min = patch_y - patch_h / 2.0
    x_max = patch_x + patch_w / 2.0
    y_max = patch_y + patch_h / 2.0

    patch = box(x_min, y_min, x_max, y_max)
    patch = affinity.rotate(patch, patch_angle, origin=(patch_x, patch_y), use_radians=False)

    return patch


def update_intrinsics(intrinsics, top_crop=0.0, left_crop=0.0, scale_width=1.0, scale_height=1.0):
    """
    Parameters
    ----------
        intrinsics: torch.Tensor (3, 3)
        top_crop: float
        left_crop: float
        scale_width: float
        scale_height: float
    """
    updated_intrinsics = intrinsics.clone()
    # Adjust intrinsics scale due to resizing
    updated_intrinsics[0, 0] *= scale_width
    updated_intrinsics[0, 2] *= scale_width
    updated_intrinsics[1, 1] *= scale_height
    updated_intrinsics[1, 2] *= scale_height

    # Adjust principal point due to cropping
    updated_intrinsics[0, 2] -= left_crop
    updated_intrinsics[1, 2] -= top_crop

    return updated_intrinsics


def calculate_birds_eye_view_parameters(x_bounds, y_bounds, z_bounds):
    """
    Parameters
    ----------
        x_bounds: Forward direction in the ego-car.
        y_bounds: Sides
        z_bounds: Height

    Returns
    -------
        bev_resolution: Bird's-eye view bev_resolution
        bev_start_position Bird's-eye view first element
        bev_dimension Bird's-eye view tensor spatial dimension
    """
    bev_resolution = torch.tensor([row[2] for row in [x_bounds, y_bounds, z_bounds]])
    bev_start_position = torch.tensor([row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
    bev_dimension = torch.tensor([(row[1] - row[0]) / row[2] for row in [x_bounds, y_bounds, z_bounds]],
                                 dtype=torch.long)

    return bev_resolution, bev_start_position, bev_dimension


def convert_egopose_to_matrix_numpy(egopose):
    transformation_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = Quaternion(egopose['rotation']).rotation_matrix
    translation = np.array(egopose['translation'])
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    transformation_matrix[3, 3] = 1.0
    return transformation_matrix

def get_global_pose(rec, nusc, inverse=False):
    lidar_sample_data = nusc.get('sample_data', rec['data']['LIDAR_TOP'])

    sd_ep = nusc.get("ego_pose", lidar_sample_data["ego_pose_token"])
    sd_cs = nusc.get("calibrated_sensor", lidar_sample_data["calibrated_sensor_token"])
    if inverse is False:
        global_from_ego = transform_matrix(sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=False)
        ego_from_sensor = transform_matrix(sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=False)
        pose = global_from_ego.dot(ego_from_sensor)
    else:
        sensor_from_ego = transform_matrix(sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=True)
        ego_from_global = transform_matrix(sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=True)
        pose = sensor_from_ego.dot(ego_from_global)
    return pose

def invert_matrix_egopose_numpy(egopose):
    """ Compute the inverse transformation of a 4x4 egopose numpy matrix."""
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix


def mat2pose_vec(matrix: torch.Tensor):
    """
    Converts a 4x4 pose matrix into a 6-dof pose vector
    Args:
        matrix (ndarray): 4x4 pose matrix
    Returns:
        vector (ndarray): 6-dof pose vector comprising translation components (tx, ty, tz) and
        rotation components (rx, ry, rz)
    """

    # M[1, 2] = -sinx*cosy, M[2, 2] = +cosx*cosy
    rotx = torch.atan2(-matrix[..., 1, 2], matrix[..., 2, 2])

    # M[0, 2] = +siny, M[1, 2] = -sinx*cosy, M[2, 2] = +cosx*cosy
    cosy = torch.sqrt(matrix[..., 1, 2] ** 2 + matrix[..., 2, 2] ** 2)
    roty = torch.atan2(matrix[..., 0, 2], cosy)

    # M[0, 0] = +cosy*cosz, M[0, 1] = -cosy*sinz
    rotz = torch.atan2(-matrix[..., 0, 1], matrix[..., 0, 0])

    rotation = torch.stack((rotx, roty, rotz), dim=-1)

    # Extract translation params
    translation = matrix[..., :3, 3]
    return torch.cat((translation, rotation), dim=-1)


def euler2mat(angle: torch.Tensor):
    """Convert euler angles to rotation matrix.
    Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angle: rotation angle along 3 axis (in radians) [Bx3]
    Returns:
        Rotation matrix corresponding to the euler angles [Bx3x3]
    """
    shape = angle.shape
    angle = angle.view(-1, 3)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = torch.zeros_like(z)
    ones = torch.ones_like(z)
    zmat = torch.stack([cosz, -sinz, zeros, sinz, cosz, zeros, zeros, zeros, ones], dim=1).view(-1, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros, siny, zeros, ones, zeros, -siny, zeros, cosy], dim=1).view(-1, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros, zeros, cosx, -sinx, zeros, sinx, cosx], dim=1).view(-1, 3, 3)

    rot_mat = xmat.bmm(ymat).bmm(zmat)
    rot_mat = rot_mat.view(*shape[:-1], 3, 3)
    return rot_mat


def pose_vec2mat(vec: torch.Tensor):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz [B,6]
    Returns:
        A transformation matrix [B,4,4]
    """
    translation = vec[..., :3].unsqueeze(-1)  # [...x3x1]
    rot = vec[..., 3:].contiguous()  # [...x3]
    rot_mat = euler2mat(rot)  # [...,3,3]
    transform_mat = torch.cat([rot_mat, translation], dim=-1)  # [...,3,4]
    transform_mat = torch.nn.functional.pad(transform_mat, [0, 0, 0, 1], value=0)  # [...,4,4]
    transform_mat[..., 3, 3] = 1.0
    return transform_mat


def invert_pose_matrix(x):
    """
    Parameters
    ----------
        x: [B, 4, 4] batch of pose matrices

    Returns
    -------
        out: [B, 4, 4] batch of inverse pose matrices
    """
    assert len(x.shape) == 3 and x.shape[1:] == (4, 4), 'Only works for batch of pose matrices.'

    transposed_rotation = torch.transpose(x[:, :3, :3], 1, 2)
    translation = x[:, :3, 3:]

    inverse_mat = torch.cat([transposed_rotation, -torch.bmm(transposed_rotation, translation)], dim=-1) # [B,3,4]
    inverse_mat = torch.nn.functional.pad(inverse_mat, [0, 0, 0, 1], value=0)  # [B,4,4]
    inverse_mat[..., 3, 3] = 1.0
    return inverse_mat


def warp_features(x, flow, mode='nearest', spatial_extent=None):
    """ Applies a rotation and translation to feature map x.
        Args:
            x: (b, c, h, w) feature map
            flow: (b, 6) 6DoF vector (only uses the xy poriton)
            mode: use 'nearest' when dealing with categorical inputs
        Returns:
            in plane transformed feature map
        """
    if flow is None:
        return x
    b, c, h, w = x.shape
    # z-rotation
    angle = flow[:, 5].clone()  # torch.atan2(flow[:, 1, 0], flow[:, 0, 0])
    # x-y translation
    translation = flow[:, :2].clone()  # flow[:, :2, 3]

    # Normalise translation. Need to divide by how many meters is half of the image.
    # because translation of 1.0 correspond to translation of half of the image.
    translation[:, 0] /= spatial_extent[0]
    translation[:, 1] /= spatial_extent[1]
    # forward axis is inverted
    translation[:, 0] *= -1

    cos_theta = torch.cos(angle)
    sin_theta = torch.sin(angle)

    # output = Rot.input + translation
    # tx and ty are inverted as is the case when going from real coordinates to numpy coordinates
    # translation_pos_0 -> positive value makes the image move to the left
    # translation_pos_1 -> positive value makes the image move to the top
    # Angle -> positive value in rad makes the image move in the trigonometric way
    transformation = torch.stack([cos_theta, -sin_theta, translation[:, 1],
                                  sin_theta, cos_theta, translation[:, 0]], dim=-1).view(b, 2, 3)

    # Note that a rotation will preserve distances only if height = width. Otherwise there's
    # resizing going on. e.g. rotation of pi/2 of a 100x200 image will make what's in the center of the image
    # elongated.
    grid = torch.nn.functional.affine_grid(transformation, size=x.shape, align_corners=False)
    grid = grid.to(dtype=x.dtype)
    warped_x = torch.nn.functional.grid_sample(x, grid, mode=mode, padding_mode='zeros', align_corners=False)

    return warped_x


def cumulative_warp_features(x, flow, mode='nearest', spatial_extent=None):
    """ Warps a sequence of feature maps by accumulating incremental 2d flow.

    x[:, -1] remains unchanged
    x[:, -2] is warped using flow[:, -2]
    x[:, -3] is warped using flow[:, -3] @ flow[:, -2]
    ...
    x[:, 0] is warped using flow[:, 0] @ ... @ flow[:, -3] @ flow[:, -2]

    Args:
        x: (b, t, c, h, w) sequence of feature maps
        flow: (b, t, 6) sequence of 6 DoF pose
            from t to t+1 (only uses the xy poriton)

    """
    sequence_length = x.shape[1]
    if sequence_length == 1:
        return x

    flow = pose_vec2mat(flow)

    out = [x[:, -1]]
    cum_flow = flow[:, -2]
    for t in reversed(range(sequence_length - 1)):
        out.append(warp_features(x[:, t], mat2pose_vec(cum_flow), mode=mode, spatial_extent=spatial_extent))
        # @ is the equivalent of torch.bmm
        cum_flow = flow[:, t - 1] @ cum_flow

    return torch.stack(out[::-1], 1)


def cumulative_warp_features_reverse(x, flow, mode='nearest', spatial_extent=None):
    """ Warps a sequence of feature maps by accumulating incremental 2d flow.

    x[:, 0] remains unchanged
    x[:, 1] is warped using flow[:, 0].inverse()
    x[:, 2] is warped using flow[:, 0].inverse() @ flow[:, 1].inverse()
    ...

    Args:
        x: (b, t, c, h, w) sequence of feature maps
        flow: (b, t, 6) sequence of 6 DoF pose
            from t to t+1 (only uses the xy poriton)

    """
    flow = pose_vec2mat(flow)

    out = [x[:,0]]
    
    for i in range(1, x.shape[1]):
        if i==1:
            cum_flow = invert_pose_matrix(flow[:, 0])
        else:
            cum_flow = cum_flow @ invert_pose_matrix(flow[:,i-1])
        out.append( warp_features(x[:,i], mat2pose_vec(cum_flow), mode, spatial_extent=spatial_extent))
    return torch.stack(out, 1)


class VoxelsSumming(torch.autograd.Function):
    """Adapted from https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/tools.py#L193"""
    @staticmethod
    def forward(ctx, x, geometry, ranks):
        """The features `x` and `geometry` are ranked by voxel positions."""
        # Cumulative sum of all features.
        x = x.cumsum(0)

        # Indicates the change of voxel.
        mask = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        mask[:-1] = ranks[1:] != ranks[:-1]

        x, geometry = x[mask], geometry[mask]
        # Calculate sum of features within a voxel.
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        ctx.save_for_backward(mask)
        ctx.mark_non_differentiable(geometry)

        return x, geometry

    @staticmethod
    def backward(ctx, grad_x, grad_geometry):
        (mask,) = ctx.saved_tensors
        # Since the operation is summing, we simply need to send gradient
        # to all elements that were part of the summation process.
        indices = torch.cumsum(mask, 0)
        indices[mask] -= 1

        output_grad = grad_x[indices]

        return output_grad, None, None

def ___extract_trajs(centerlines, trj, nvar=11, viz=False, debugg=False, device='cpu', labels=None, griddim=None, problem=None, avoid_obs=False, obs=None):
    """
        this is just saved to understand where to insert the plotting code.
    """
    save_device = device
    device = 'cpu'
    bbx = np.array([-50.0 + 0.5/2.0, -50.0 + 0.5/2.0])
    ddx = np.array([0.5, 0.5])
    if griddim == None:
        X_MAX = 30
        X_MIN = 0
        Y_MAX = 2
        Y_MIN = -2
        NUM_X = 30
        NUM_Y = 5
    else:
        X_MIN, X_MAX, NUM_X, Y_MIN, Y_MAX, NUM_Y = griddim

    # get trajectories
    x = torch.linspace(X_MIN, X_MAX, NUM_X).to(device)
    y = torch.linspace(Y_MIN, Y_MAX, NUM_Y).to(device)
    xv, yv = torch.meshgrid(x, y)
    print(device)
    grid = torch.dstack((xv, yv)).to(device)
    ccc = centerlines.detach().cpu().numpy()
    cx, _ = interp_polyline_by_fixed_waypt_interval_np(ccc, 0.1)
    cx = extend_both_sides(cx, resolution=0.1, extension=100)
    cx = cx * ddx
    cx += bbx
    cx[:, [1, 0]] = cx[:, [0, 1]]
    cx[:, :1] = cx[:, :1] * -1
    cx = torch.tensor(cx).to(device)
    if viz:
        trr = trj.squeeze().detach().cpu()
        plt.plot(cx[:, 0], cx[:, 1], color='grey')
        plt.scatter(cx[:, 0], cx[:, 1], color='grey')
        plt.plot(trr[:, 0], trr[:, 1], color='blue')
        plt.scatter(trr[:, 0], trr[:, 1], color='blue')
        # print(trr[:, 0], trr[:, 1])
        plt.xlim([-5,5])
        plt.ylim([-5,5])
        plt.savefig('output_vis/debug_actual.png'); plt.clf()

    trr = trj.squeeze().detach().cpu()
    ref_line_ = torch.tensor(cx.reshape(1, cx.shape[0], 2))
    ref_line = torch.zeros((1, ref_line_.shape[1] - 1, 3))
    ref_line[:, :, :2] = ref_line_[:, 1:] 
    off = ref_line_[:, 1:] - ref_line_[:,:-1]
    ref_line[:, :, 2] = torch.atan2(off[:, :, 1], off[:, :, 0])
    ref_line = ref_line.float().to(device)
    trj = trj.float().to(device)

    test_ref_line=ref_line[:, 100:-100]
    sdd = project_to_frenet_frame(test_ref_line,  ref_line)
    if debugg: pdb.set_trace()
    if viz:
        plt.scatter(sdd[0, :, 0].detach().cpu(), sdd[0, :, 1].detach().cpu(), color='grey')
        plt.show()

    ssd = project_to_frenet_frame(trj, ref_line) # (1, 7, 2)

    if avoid_obs:
        # obs -> (M, num, 2) : M obstacles for num timesteps
        # avoid obstacles
        num_obs = 20
        a_obs = 1
        b_obs = 1
        x_obs = obs[:, :, 0]
        y_obs = obs[:, :, 1]

        ssd_obs = project_to_frenet_frame(trj, ref_line) # (1, 7, 2)
        problem = OPTNode(
            x_obs=x_obs, 
            y_obs=y_obs, 
            a_obs=a_obs, 
            b_obs=b_obs, 
            num_obs=num_obs, 
            rho_obs=1, 
            rho_eq=problem.rho_eq, 
            t_fin=problem.t_fin, 
            num=problem.num, 
            bernstein_coeff_order10_new=problem.bernstein_coeff_order10_new, 
            device = centerlines.device
        )



    cx = interp_arc_np(2000, cx.detach().cpu().numpy())
    cx = extend_both_sides(cx, resolution=0.1, extension=50)
    cx = torch.tensor(cx).to(device)
    if debugg: pdb.set_trace()

    if viz:
        gt = labels['gt_trajectory_prev'][0]
        plt.scatter(gt[:2, 0].detach().cpu(), gt[:2, 1].detach().cpu(), color='red')
        plt.plot(gt[:, 0].detach().cpu(), gt[:, 1].detach().cpu(), color='red', label='observed')
        plt.scatter(gt[2:, 0].detach().cpu(), gt[2:, 1].detach().cpu(), color='orange')
        plt.plot(gt[2:, 0].detach().cpu(), gt[2:, 1].detach().cpu(), color='orange', label='future')
        # plt.axis('equal')
        # plt.xlim([-5, 5])
        # plt.ylim([-5, 5])
        plt.legend()
        plt.savefig("output_vis/debug_mid.png");plt.clf()

    ssd = project_to_frenet_frame(trj, ref_line) # (1, 7, 2)
    s_offset = ssd[:,2,0]
    d_offset = ssd[0,2,1]
    vel = ssd[0,2] - ssd[0,1]
    vel_prev = ssd[0,1] - ssd[0, 0]
    theta = torch.atan2(vel[1], vel[0]) # to make angle 0 wrt ego-agent
    theta_prev = torch.atan2(vel_prev[1], vel_prev[0])
    psidot_init = (theta - theta_prev)/0.5
    psi_init = theta
    v_init = torch.linalg.norm(vel)/0.5
    y_init = 0
    x_init = 0
    rot_inv = torch.tensor([[cos(-theta), -sin(-theta)], [sin(-theta), cos(-theta)]]).to(device)
    rot = torch.tensor([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]).to(device)
    if debugg: pdb.set_trace()

    sdd_frame = sdd -  torch.tensor([s_offset, d_offset]).to(device)
    ssd_frame = ssd -  torch.tensor([s_offset, d_offset]).to(device)
    eps_frame = torch.tensor(grid).to(device) - torch.tensor([0, d_offset]).to(device) # (20, 7, 2)


    if debugg: pdb.set_trace()
    if viz:
        for eps in eps_frame:
            ep = eps.detach().cpu().numpy()
            plt.scatter(ep[:, 0], ep[:, 1])
        ss = ssd_frame[0].detach().cpu().numpy()
        plt.scatter(ss[:2, 0], ss[:2, 1], color='red')
        plt.scatter(ss[2:, 0], ss[2:, 1], color='orange')
        sd = sdd_frame[0].detach().cpu().numpy()
        plt.scatter(sd[:, 0], sd[:, 1], color='grey')
        plt.plot(sd[:, 0], sd[:, 1], color='grey')
        plt.savefig("output_vis/debug_ss.png"); plt.clf()
        plt.clf()

    if debugg: pdb.set_trace()
    # rotate
    sdd_frame[0] = torch.matmul(rot_inv, sdd_frame[0].T).T # (1, M, 2)
    ssd_frame = torch.matmul(rot_inv, ssd_frame[0].T).T # ()
    eps_frame = eps_frame.permute(1, 2, 0) # (7, 2, 20)
    eps_frame = torch.matmul(rot_inv, eps_frame) # (2, 2) x (7, 2, 20) -> (7, 2, 20)
    if debugg: pdb.set_trace()

    if viz:
        ss = ssd_frame.detach().cpu().numpy()
        plt.scatter(ss[:2, 0], ss[:2, 1], color='red')
        plt.scatter(ss[2:, 0], ss[2:, 1], color='orange')
        sd = sdd_frame[0].detach().cpu().numpy()
        plt.scatter(sd[:, 0], sd[:, 1], color='grey', label='centerline')
        plt.plot(sd[:, 0], sd[:, 1], color='grey')
        # for ep in eps_frame:
        #     plt.scatter(ep[:, 0], ep[:, 1], color='yellow')
        plt.scatter(eps_frame[:, 0, :].flatten().detach().cpu().numpy(), eps_frame[:, 1, :].flatten().detach().cpu().numpy(), color='yellow')
        # plt.savefig("output_vis/debug_ss.png"); plt.clf()
        plt.xlim([-25, 25])
        plt.ylim([-25, 25])
        plt.legend()
        plt.savefig("output_vis/debug_ss_rotated.png"); plt.clf()
        plt.clf()

    if debugg: pdb.set_trace()
    x_eps = eps_frame[:,0,:].flatten()
    y_eps = eps_frame[:,1,:].flatten()
    fixed_params = torch.tensor([x_init, y_init, v_init, psi_init, psidot_init]).to(device)
    fixed_params = fixed_params.expand(len(x_eps), 5)
    variable_params = torch.zeros((len(x_eps), 3)).to(device)
    variable_params[:, 0] = torch.tensor(x_eps)
    variable_params[:, 1] = torch.tensor(y_eps)
    variable_params[:, 2] = torch.tensor(-theta)
    P = problem.P
    cc, _ = problem.solve(fixed_params.double().detach().cpu(), variable_params.double().detach().cpu())
    x_sol = torch.matmul(P, cc.T[:nvar]).T
    y_sol = torch.matmul(P, cc.T[nvar:2*nvar]).T

    if debugg: pdb.set_trace()

    if viz:
        ss = ssd_frame.detach().cpu().numpy()
        plt.scatter(eps_frame[:, 0, :].flatten().detach().cpu().numpy(), eps_frame[:, 1, :].flatten().detach().cpu().numpy(), color='yellow')
        sd = sdd_frame[0].detach().cpu().numpy()
        plt.scatter(sd[:, 0], sd[:, 1], color='grey', label='centerline')
        plt.plot(sd[:, 0], sd[:, 1], color='grey')
        for i in range(len(x_sol)):
            plt.plot(x_sol[i].flatten().detach().cpu().numpy(), y_sol[i].flatten().detach().cpu().numpy())
        # plt.savefig("output_vis/debug_ss.png"); plt.clf()
        plt.scatter(ss[:2, 0], ss[:2, 1], color='red')
        plt.scatter(ss[2:, 0], ss[2:, 1], color='orange')
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])
        plt.legend()
        plt.savefig("output_vis/debug_frenet.png")
        plt.clf()

    if debugg: pdb.set_trace()
    trajs = torch.dstack((x_sol, y_sol)).to(device) # N, 7, 2
    trajs = trajs.permute(0, 2, 1).float()
    trajs = torch.matmul(rot, trajs)
    trajs = trajs.permute(0, 2, 1)
    trajs = trajs + torch.tensor([s_offset, d_offset]).to(device)

    min_val = 1e11
    min_ind = 0

    dense_trajs = project_to_cartesian_frame(trajs.reshape(1, trajs.shape[0] * trajs.shape[1], 2), ref_line).reshape(1, trajs.shape[0], trajs.shape[1], 2)#(B, N, n_future, 2)  
    return dense_trajs.to(save_device)

