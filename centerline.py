import open3d
import os
import argparse
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import time
from skimage import morphology
from scipy import ndimage

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

    '''
    for i in range(len(idx)):
        point = demo.points[idx[i]]
        cumulants[0] += point[0]
        cumulants[1] += point[1]
        cumulants[2] += point[2]
        cumulants[3] += point[0] * point[0]
        cumulants[4] += point[0] * point[1]
        cumulants[5] += point[0] * point[2]
        cumulants[6] += point[1] * point[1]
        cumulants[7] += point[1] * point[2]
        cumulants[8] += point[2] * point[2]
    cumulants /= len(idx)
    '''

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
    # open3d.visualization.draw_geometries([demo], point_show_normal=True)
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
    print(ave)
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
        print(ave)
    return ave

def np_img(x, y, cs_scale, scale=255):
    ymax, ymin = y.max(), y.min()
    ysize = int((ymax - ymin) // cs_scale + 1)
    y = (y - ymin) // cs_scale
    xmax, xmin = x.max(), x.min()
    xsize = int((xmax - xmin) // cs_scale + 1)
    x = (x - xmin) // cs_scale
    img = np.zeros((xsize, ysize), dtype=np.uint8)
    point_mapping = {}
    for i in range(x.shape[0]):
        img[int(x[i]), int(y[i])] = scale
        point_mapping[(int(x[i]), int(y[i]))] = i
    return img, point_mapping

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chuck_dir", type=str, default='../mesh/')
    parser.add_argument("--save_dir", type=str, default='dump/')
    parser.add_argument("--r_version", choices=['cs', 'global'])
    parser.add_argument("--closing", action="store_true")
    parser.add_argument("--opening", action="store_true")
    parser.add_argument("--disc_size", type=int, default=6)
    args = parser.parse_args()
    print(args)
    chucks = ['mesh_i1_n0.4.obj', 'refine_mesh_gt004.obj', 'refine_mesh_gt019.obj', 'refine_mesh_gt027.obj',
              'refine_mesh_gt031.obj', 'refine_mesh_gt047.obj', 'refine_mesh_gt051.obj']

    for c in chucks[:]:
        print(c)
        save_root = args.save_dir + os.path.splitext(c)[0]

        # read chuck
        mesh = open3d.io.read_triangle_mesh("../mesh/" + c)
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
        # center = np.average(np.asarray(downpcd.points), axis=0)
        points = np.asarray(downpcd.points)
        xmax, ymax, zmax = np.max(points, axis=0)
        xmin, ymin, zmin = np.min(points, axis=0)
        center = np.array([(xmax + xmin) / 2, (ymax + ymin) / 2, (zmax + zmin) / 2])
        linepoints = [center - ave * 3, center + ave * 3]
        lines = [[0, 1]]
        line_set = open3d.geometry.LineSet(
            points=open3d.utility.Vector3dVector(linepoints),
            lines=open3d.utility.Vector2iVector(lines),
        )
        # open3d.visualization.draw_geometries([pc, line_set])
        print(time.time()-start)

        # unfold
        direction = ave / np.linalg.norm(ave)
        ori_dire = [0, -direction[2], direction[1]]
        ori_dire = ori_dire / np.linalg.norm(ori_dire)

        allpoints = np.asarray(pc.points)
        vpc = allpoints - center
        projected_l = np.dot(vpc, direction).reshape(-1, 1)
        projected_p = center + projected_l * direction
        vpr = allpoints - projected_p
        r = np.linalg.norm(vpr, axis=1).reshape(-1, 1)
        vpr = vpr / r
        cos_theta = np.dot(vpr, ori_dire)
        cos_theta = np.minimum(1, np.maximum(cos_theta, -1))
        theta = np.arccos(cos_theta)
        theta[vpr[:, 0] < 0] *= -1
        theta[vpr[:, 0] < 0] += 2 * np.pi
        oversample = theta < np.pi/4
        overtheta = theta[oversample] + 2 * np.pi
        theta = np.concatenate((theta, overtheta), axis=0)
        projected_l = np.concatenate((projected_l, projected_l[oversample]), axis=0)
        colors = np.asarray(pc.colors)
        colors = np.concatenate((colors, colors[oversample]), axis=0)
        theta *= np.mean(r)

        plt.scatter(projected_l, theta, s=2, c=colors, marker=',')
        plt.xlabel('d')
        plt.ylabel('rθ')
        plt.show()
        print(np.mean(r))

        # morphology
        img, mapping = np_img(projected_l, theta, 0.01, scale=1)
        plt.imsave(save_root + '_2d.png', img * 255, cmap='gray')
        save_mor = save_root
        if args.closing:
            img = morphology.binary_closing(img, morphology.diamond(args.disc_size)).astype(np.uint8)
            save_mor += '_close'
        if args.opening:
            img = morphology.binary_opening(img, morphology.diamond(args.disc_size)).astype(np.uint8)
            save_mor += '_open'

        img = img.astype(float)
        labeled, nr_objects = ndimage.label(img < 1.0)
        save_img = np.repeat(np.expand_dims(img, 2), 3, axis=2)
        areas = ''
        for i in range(nr_objects):
            area = np.count_nonzero(labeled == i + 1)
            areas += '%d ' % area
            save_img[:, :, 0][labeled == i + 1] = i / nr_objects
        plt.imsave(save_mor + '.png', save_img)
        print("%s: Number of holes is %d, area %s" % (c, nr_objects, areas))
        print(time.time() - start)