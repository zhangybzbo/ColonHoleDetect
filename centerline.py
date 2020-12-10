import open3d
import os
import pickle
import argparse
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import time
from collections import defaultdict
from skimage import morphology
from scipy import ndimage, interpolate, spatial

def ComputeCovariance(demo, idx):
    points = np.asarray(demo.points)
    neighbors = points[idx]
    covariance = np.zeros((3, 3))
    cumulants = np.zeros(9)
    cumulants[:3] = np.mean(neighbors, axis=0)
    dots = neighbors * neighbors
    cumulants[3], cumulants[6], cumulants[8] = np.mean(dots, axis=0)
    cumulants[4] = np.mean(neighbors[:, 0] * neighbors[:, 1])
    cumulants[5] = np.mean(neighbors[:, 0] * neighbors[:, 2])
    cumulants[7] = np.mean(neighbors[:, 1] * neighbors[:, 2])

    covariance[0, 0] = cumulants[3] - cumulants[0] * cumulants[0]
    covariance[1, 1] = cumulants[6] - cumulants[1] * cumulants[1]
    covariance[2, 2] = cumulants[8] - cumulants[2] * cumulants[2]
    covariance[0, 1] = cumulants[4] - cumulants[0] * cumulants[1]
    covariance[1, 0] = covariance[0, 1]
    covariance[0, 2] = cumulants[5] - cumulants[0] * cumulants[2]
    covariance[2, 0] = covariance[0, 2]
    covariance[1, 2] = cumulants[7] - cumulants[1] * cumulants[2]
    covariance[2, 1] = covariance[1, 2]

    w, v = np.linalg.eig(covariance)
    ind = np.argsort(w, axis=0)
    w = w[ind]
    v = v[:, ind]
    return w, v


def GetDir(demo, r, principle=False):
    pcd_tree = open3d.geometry.KDTreeFlann(demo)
    has_norm = demo.has_normals()
    eigenvalues = []
    for i in range(len(demo.points)):
        # create ring
        [k, idx, dist] = pcd_tree.search_radius_vector_3d(demo.points[i], r)
        [k2, idx2, dist2] = pcd_tree.search_radius_vector_3d(demo.points[i], r / 5)
        assert idx[k2 - 1] == idx2[k2 - 1]
        k = k - k2
        idx = idx[k2:]
        dist = dist[k2:]

        if k >= 3:
            w, v = ComputeCovariance(demo, idx)
            if principle:
                if w[2] < r ** 2 / 4 or w[1] < w[2] * 2 / 3:
                    # if False:
                    normal = [0., 0., 0.]
                else:
                    eigenvalues.append(w)
                    normal = v[:, 2].reshape(3)
                    if normal[2] < 0:
                        normal = -normal
            else:
                normal = v[:, 0].reshape(3)
            if np.linalg.norm(normal) == 0.0:
                normal = [0., 0., 0.]
            if has_norm:
                demo.normals[i] = normal
            else:
                demo.normals.append(normal)

        else:
            print(i)
            if has_norm:
                demo.normals[i] = [0., 0., 0.]
            else:
                demo.normals.append([0., 0., 0.])

    if not principle:
        demo.orient_normals_to_align_with_direction([1, 0, 0])
    # demo.orient_normals_to_align_with_direction([0, 0, 1])
    open3d.visualization.draw_geometries([demo])
    # print(np.asarray(demo.normals))

    return eigenvalues

def FindDir(demo):
    norms = open3d.geometry.PointCloud()
    for n in demo.normals:
        if np.linalg.norm(n) != 0.0:
            norms.points.append(n)
    norms.paint_uniform_color([0.5, 0.5, 0.5])
    # open3d.visualization.draw_geometries([norms])
    initials = np.asarray(norms.points)
    goods = np.ones(initials.shape[0]).astype(bool)
    index = np.where(goods>0)
    ave = np.average(initials[goods, :], axis=0)
    ave = ave / np.linalg.norm(ave)
    # print(ave)
    ratio = 2
    for i in range(5):
        z = np.abs(stats.zscore(initials[goods, :], axis=0))
        goods[index] = (z < 2).all(axis=1)
        index  = np.where(goods)
        norms.paint_uniform_color([0.5, 0.5, 0.5])
        np.asarray(norms.colors)[goods, :] = [1, 0, 0]
        ave = np.average(initials[goods, :], axis=0)
        ave = ave / np.linalg.norm(ave)
        # open3d.visualization.draw_geometries([norms])
        # print(ave)
    open3d.visualization.draw_geometries([norms])
    return ave

def np_img(x, y, cs_scale, scale=255):
    ymax, ymin = y.max(), y.min()
    ysize = int((ymax - ymin) // cs_scale + 1)
    y = (y - ymin) // cs_scale
    xmax, xmin = x.max(), x.min()
    xsize = int((xmax - xmin) // cs_scale + 1)
    x = (x - xmin) // cs_scale
    img = np.zeros((xsize, ysize), dtype=np.uint8)
    point_mapping = defaultdict(list)
    for i in range(x.shape[0]):
        img[int(x[i]), int(y[i])] = scale
        point_mapping[(int(x[i]), int(y[i]))].append(i)
    return img, np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1), point_mapping

def hole_sample_points(labeled, num=10):
    xys = np.nonzero(labeled)
    total = len(xys[0])
    indexs = np.random.choice(np.arange(total), num)
    xs = xys[0][indexs]
    ys = xys[1][indexs]
    return np.concatenate((xs.reshape(-1, 1), ys.reshape(-1, 1)), axis=1)

def get_hole_r(xys, rs, hole_coors, inter_n=50):
    assert xys.shape[0] == rs.shape[0]
    tree = spatial.KDTree(list(zip(xys[:, 0], xys[:, 1])))
    neighbors = tree.query(hole_coors, k=inter_n)
    hole_r = np.zeros(hole_coors.shape[0])
    for i in range(hole_coors.shape[0]):
        nears = neighbors[1][i, :]
        # r_f = interpolate.interp2d(xys[:, 0][nears], xys[:, 1][nears], rs[nears], kind='linear')  # interpolate to get r
        # hole_r[i] = r_f(hole_coors[i, 0], hole_coors[i, 1])
        hole_r[i] = np.mean(rs[nears])
    return hole_r

def xy_2_xyz(xys, rs, mean_r, center_d, theta_d, xmin_point, cs_scale):
    l_p = xmin_point + xys[:, 0].reshape(-1, 1) * cs_scale * center_d
    thetas = (xys[:, 1] * cs_scale) / mean_r
    y_d = rs.reshape(-1, 1) * np.cos(thetas).reshape(-1, 1) * theta_d
    x_d = rs.reshape(-1, 1) * np.sin(thetas).reshape(-1, 1) * np.cross(center_d, theta_d)
    ps = l_p + x_d + y_d
    return ps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_dir", type=str, default='mesh/results/')
    parser.add_argument("--save_dir", type=str, default='dump/')
    parser.add_argument("--center", choices=['extremal', 'centroid'])
    parser.add_argument("--oversample", action="store_true")
    parser.add_argument("--closing", action="store_true")
    parser.add_argument("--opening", action="store_true")
    parser.add_argument("--disc_size", type=int, default=6)
    parser.add_argument("--chunk", type=str, default='')
    args = parser.parse_args()
    print(args)
    if args.chunk == '':
        print('No chunk selected, using default 000.')
        chunks = ['000']
    else:
        chunks = [args.chunk]
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for c in chunks[:]:
        print(c)
        save_root = args.save_dir + os.path.splitext(c)[0]

        # read chuck
        files = os.listdir(os.path.join(args.chunk_dir,c))
        for file in files:
            print(file)
            if file.split('.')[-1] == 'obj':
                mesh = open3d.io.read_triangle_mesh(os.path.join(args.chunk_dir, c, file))
            else:
                continue
        pc = open3d.geometry.PointCloud()
        pc.points = mesh.vertices
        pc.colors = mesh.vertex_colors
        pc.normals = mesh.vertex_normals

        start = time.time()
        # normal estimation
        downpcd = pc.voxel_down_sample(voxel_size=0.05)
        downpcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        downpcd.normalize_normals()
        downpcd.orient_normals_to_align_with_direction([1, 0, 0])
        # open3d.visualization.draw_geometries([downpcd], point_show_normal=False)

        # get principle directions
        eigen = GetDir(downpcd, 0.6, True)

        # get centerline direction
        ave = FindDir(downpcd)

        # print centerline
        points = np.asarray(downpcd.points)
        if args.center == 'extremal':
            xmax, ymax, zmax = np.max(points, axis=0)
            xmin, ymin, zmin = np.min(points, axis=0)
            center = np.array([(xmax + xmin) / 2, (ymax + ymin) / 2, (zmax + zmin) / 2])
        else:
            center = np.mean(points, axis=0)
        linepoints = [center - ave * 3, center + ave * 3]
        lines = [[0, 1]]
        line_set = open3d.geometry.LineSet(
            points=open3d.utility.Vector3dVector(linepoints),
            lines=open3d.utility.Vector2iVector(lines),
        )
        open3d.visualization.draw_geometries([pc, line_set])
        print(time.time()-start)

        # unfold
        direction = ave / np.linalg.norm(ave)
        ori_dire = [0, -direction[2], direction[1]]
        ori_dire = ori_dire / np.linalg.norm(ori_dire)

        allpoints = np.asarray(pc.points)
        vpc = allpoints - center
        projected_l = np.dot(vpc, direction).reshape(-1, 1)
        projected_p = center + projected_l * direction
        vpr_vec = allpoints - projected_p
        r = np.linalg.norm(vpr_vec, axis=1).reshape(-1, 1)
        vpr = vpr_vec / r
        cos_theta = np.dot(vpr, ori_dire)
        cos_theta = np.minimum(1, np.maximum(cos_theta, -1))
        theta = np.arccos(cos_theta)
        theta[vpr[:, 0] < 0] *= -1
        theta[vpr[:, 0] < 0] += 2 * np.pi
        mean_r = np.mean(r)
        colors = np.asarray(pc.colors)
        # over sample
        if args.oversample:
            oversample = theta < np.pi/4
            overtheta = theta[oversample] + 2 * np.pi
            theta = np.concatenate((theta, overtheta), axis=0)
            projected_l = np.concatenate((projected_l, projected_l[oversample]), axis=0)
            r = np.concatenate((r, r[oversample]), axis=0)
            colors = np.concatenate((colors, colors[oversample]), axis=0)
        theta *= mean_r

        '''img, positions, mapping = np_img(projected_l, theta, 0.01, scale=1)
        shapes = img.shape
        end_length = int(shapes[0] * 0.075)
        getridof = np.where(positions[:, 0] > (shapes[0] - end_length*2.1))[0]
        projected_l = np.delete(projected_l, getridof)
        theta = np.delete(theta, getridof)
        colors = np.delete(colors, getridof, axis=0)

        import matplotlib
        matplotlib.rcParams['xtick.labelsize'] = 20
        matplotlib.rcParams['ytick.labelsize'] = 20
        plt.scatter(projected_l, theta, s=2, c=colors, marker=',')
        plt.xlabel('d', fontsize=20)
        plt.ylabel('rθ', fontsize=20)
        plt.title('Flattened surface', fontsize=25)
        plt.savefig('test.png', transparent=True)
        plt.show()'''
        print(mean_r)

        # morphology
        img, positions, mapping = np_img(projected_l, theta, 0.01, scale=1)
        plt.imsave(save_root + '_2d.png', img * 255, cmap='gray')
        save_mor = save_root
        if args.closing:
            img = morphology.binary_closing(img, morphology.diamond(args.disc_size)).astype(np.uint8)
            save_mor += '_close'
        if args.opening:
            img = morphology.binary_opening(img, morphology.diamond(args.disc_size)).astype(np.uint8)
            save_mor += '_open'

        # pick center 85% area, avoid end
        shapes = img.shape
        end_length = int(shapes[0] * 0.1)
        # new_img = np.zeros((shapes[0]*3, shapes[1]))
        # new_img[shapes[0]:shapes[0]*2, :] = img
        new_img = img[end_length:-end_length, :]
        new_shapes = new_img.shape
        surface = np.count_nonzero(new_img)

        new_img = new_img.astype(float)
        labeled, nr_objects = ndimage.label(new_img < 1.0)
        save_img = np.repeat(np.expand_dims(new_img, 2), 3, axis=2)

        hole_point_num = [] # how many hole point in each hole
        hole_samples = []

        areas = ''
        areas_stat = 0
        for i in range(nr_objects):
            area = np.count_nonzero(labeled == i + 1)
            if area >= shapes[0] * shapes[1]:
                continue
            areas_stat += area
            areas += '%d ' % area
            save_img[:, :, 0][labeled == i + 1] = i / nr_objects

            hole_sample = hole_sample_points(labeled == i + 1, num=area // 3)
            hole_samples.append(hole_sample)
            hole_point_num.append(hole_sample.shape[0])
        plt.imsave(save_mor + '.png', save_img)
        print("%s:\nSurface area %d\nNumber of holes is %d, area %s" % (c, surface, nr_objects, areas))
        hole_rate = areas_stat / (areas_stat + surface)
        # assert areas_stat + surface == new_shapes[0] * new_shapes[1]
        print('Hole ratio %.4f' % hole_rate)
        print(time.time() - start)

        # create hole points
        # hole_samples = hole_sample_points(labeled, num=areas_stat//10)
        hole_samples = np.concatenate(hole_samples, axis=0)
        hole_samples[:, 0] += end_length
        # hole_samples[:, 0] -= shapes[0]
        overlayed_pixel = []
        for location, p_ls in mapping.items():
            overlayed_pixel.append(p_ls[0])
        consistant_sf = np.concatenate((positions[overlayed_pixel], positions[overlayed_pixel], positions[overlayed_pixel]), axis=0)
        consistant_rs = np.concatenate((r[overlayed_pixel], r[overlayed_pixel], r[overlayed_pixel]), axis=0)
        consistant_sf[:len(overlayed_pixel), 1] -= shapes[1]
        consistant_sf[-len(overlayed_pixel):, 1] += shapes[1]
        hole_r = get_hole_r(consistant_sf, consistant_rs, hole_samples)
        left_p = min(projected_l) * direction + center
        hole_sample_xyz = xy_2_xyz(hole_samples, hole_r, mean_r, direction, ori_dire, left_p, 0.01)

        hole_point = open3d.geometry.PointCloud()
        hole_point.points = open3d.utility.Vector3dVector(hole_sample_xyz)
        all_colors = []
        # hole_point.paint_uniform_color([0, 0, 1])
        pc.paint_uniform_color([0.5, 0.5, 0.5])

        hole_sample_ls = {}
        hole_sample_ls['all'] = hole_sample_xyz
        accum = 0
        for i, hole in enumerate(hole_point_num):
            hole_sample_ls[i] = hole_sample_xyz[accum:accum+hole, :]
            all_colors.append(np.repeat(np.random.uniform(0, 1, (1, 3)), hole, axis=0))
            accum += hole
        hole_point.colors = open3d.utility.Vector3dVector(np.concatenate(all_colors, axis=0))
        open3d.visualization.draw_geometries([pc, hole_point])
        with open(save_root+'.pkl', 'wb') as f:
            pickle.dump(hole_sample_ls, f)
        # np.save(save_root+'.npy', hole_sample_xyz)
