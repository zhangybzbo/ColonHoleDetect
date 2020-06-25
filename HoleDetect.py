import argparse
import vtk
import math
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology
import skimage.feature
from scipy import ndimage

ColorBackground = [0.0, 0., 0.]

def visual_pc(pc_poly, name):
    vertexFilter = vtk.vtkVertexGlyphFilter()
    vertexFilter.SetInputData(pc_poly)
    vertexFilter.Update()
    polydata = vtk.vtkPolyData()
    polydata.ShallowCopy(vertexFilter.GetOutput())
    # mapper, vtkPolyData to graphics primitives
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    # actor, represents an object (geometry & properties) in a rendered scene
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    
    # renderer
    ren = vtk.vtkRenderer()
    ren.SetBackground(ColorBackground)
    ren.AddActor(actor)
    ren.ResetCamera()

    # render window
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)

    # Create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    
    # visualize
    renWin.Render()
    windowToImageFilter = vtk.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renWin)
    windowToImageFilter.ReadFrontBufferOff()
    windowToImageFilter.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(name)
    writer.SetInputConnection(windowToImageFilter.GetOutputPort())
    writer.Write()
    iren.Initialize()
    renWin.Render()
    iren.Start()
    
def visual_img(vtk_img, name=None):
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputData(vtk_img)
    Actor = vtk.vtkActor()
    Actor.SetMapper(mapper)
    Renderer = vtk.vtkRenderer()
    Renderer.AddActor(Actor)
    Renderer.ResetCamera()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(Renderer)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    # visualize
    if name:
        renWin.Render()
        windowToImageFilter = vtk.vtkWindowToImageFilter()
        windowToImageFilter.SetInput(renWin)
        windowToImageFilter.ReadFrontBufferOff()
        windowToImageFilter.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(name)
        writer.SetInputConnection(windowToImageFilter.GetOutputPort())
        writer.Write()
    iren.Initialize()
    renWin.Render()
    iren.Start()
    
class colon(object):
    def __init__(self, path):
        reader = vtk.vtkOBJReader()
        reader.SetFileName(path)
        reader.Update()
        self.obj = reader.GetOutput()               
        self.math = vtk.vtkMath()
        
        # renderer
        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(ColorBackground)
        self.actors = {}
        
        # add colon
        # self.add_render(self.obj, 'main')
        
    
    def add_render(self, obj, name, color=None):
        # mapper, vtkPolyData to graphics primitives
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(obj)
        # actor, represents an object (geometry & properties) in a rendered scene
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        if color:
            actor.GetProperty().SetColor(color)
        
        self.actors[name] = actor
        self.ren.AddActor(self.actors[name])
    
    def remove_render(self, name):
        self.ren.RemoveActor(self.actors[name])
        del self.actors[name]
    
    def visual(self):        
        # render window
        renWin = vtk.vtkRenderWindow()
        renWin.AddRenderer(self.ren)
        # Create a renderwindowinteractor
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)
        # visualize
        iren.Initialize()
        renWin.Render()
        iren.Start()
        
    def set_centerline(self, resolution):
        xmin,xmax, ymin,ymax, zmin,zmax = self.obj.GetBounds()
        center = self.obj.GetCenter() # colon center
        # key points
        pts = vtk.vtkPoints()
        pts.InsertNextPoint([center[0], center[1], zmin])
        pts.InsertNextPoint(center)
        pts.InsertNextPoint([center[0], center[1], zmax])
        
        # spline from keypoints
        spline = vtk.vtkParametricSpline() 
        spline.SetPoints(pts)
        function = vtk.vtkParametricFunctionSource()
        function.SetParametricFunction(spline)
        function.Update()
        function.SetUResolution(resolution)
        function.Update()
        self.centerline = function.GetOutput() # centerline
        self.cl_scale = (zmax-zmin)/resolution
        
        # centerline direction, [0, 0, 1]
        self.direction = [0, 0, zmax-zmin]
        self.math.Normalize(self.direction) 
        
    def cs_cut(self, scale):
        assert scale in ['cs', 'global']
        
        # all tangents [0, 0, 1]
        N = self.centerline.GetNumberOfPoints()
        Tangents = vtk.vtkDoubleArray()
        Tangents.SetNumberOfComponents(3)
        Tangents.SetNumberOfTuples(N)
        for i in range(N):
            Tangents.SetTuple(i, self.direction) 
            
        # cutter, find cross section
        cutter = vtk.vtkCutter()
        cutter.SetInputData(self.obj)
        connectivityFilter = vtk.vtkPolyDataConnectivityFilter()
        connectivityFilter.SetInputConnection(cutter.GetOutputPort())
        connectivityFilter.SetExtractionModeToClosestPointRegion()
        plane = vtk.vtkPlane()
        self.all_cs = {}
        self.cs_deform = vtk.vtkPolyData()
        self.cs_points = vtk.vtkPoints()
        self.arrays = [[], []]        
        dis_ls = []
        theta_ls = []
        i_index = []
        self.point_mapping = []
        
        for i in range(N):
            # cut, get cross section
            centerPoint = [0] * 3
            self.centerline.GetPoint(i, centerPoint)
            connectivityFilter.SetClosestPoint(centerPoint)
            plane.SetOrigin(self.centerline.GetPoint(i))
            plane.SetNormal(Tangents.GetTuple(i))
            cutter.SetCutFunction(plane)
            cutter.Update()
            connectivityFilter.Update()
            cutline = cutter.GetOutput()
            
            newcut = vtk.vtkPolyData()
            newcut.DeepCopy(cutline)
            self.all_cs[i] = newcut
            self.add_render(newcut, 'cut%d' % i, (255, 0, 0))
                        
            n_p = newcut.GetNumberOfPoints()
            
            # origin direction on plane, [0, tz, -ty]
            # TODO: nake sure tangent not [1, 0, 0]
            tangent_d = Tangents.GetTuple(i)
            origin_d = [0, tangent_d[2], -tangent_d[1]]
            self.math.Normalize(origin_d)
            if scale == 'cs':
                dis_ls = []
                theta_ls = []
            for j in range(n_p):
                p = newcut.GetPoint(j)
                dj = [0] * 3
                self.math.Subtract(p, centerPoint, dj)
                self.math.Normalize(dj)
                cos_theta = self.math.Dot(origin_d, dj)
                cos_theta = min(1, max(cos_theta, -1))
                theta = math.acos(cos_theta)
                if dj[0] < 0:
                    theta = - theta
                dis = self.math.Distance2BetweenPoints(centerPoint, p)
                dis = dis**0.5
                dis_ls.append(dis)
                theta_ls.append(theta)
                i_index.append(i)
                self.point_mapping.append((i, j))
                if scale == 'global': # over sample
                    if theta >= math.pi / 2:
                        theta_ls.append(-math.pi * 2 + theta)
                        i_index.append(i)
                        self.point_mapping.append((i, j))
        
            # distance weight
            if scale == 'cs':
                ave_dis = np.average(np.array(dis_ls))
                for j in range(n_p):
                    self.arrays[0].append(i*self.cl_scale)
                    self.arrays[1].append(ave_dis*theta_ls[j])
                    self.cs_points.InsertNextPoint(i*self.cl_scale, ave_dis*theta_ls[j], 0)   
        
        # ave dis
        if scale == 'global':
            ave_dis = np.average(np.array(dis_ls))
            for i, theta in enumerate(theta_ls):
                self.arrays[0].append(i_index[i]*self.cl_scale)
                self.arrays[1].append(ave_dis*theta)
                self.cs_points.InsertNextPoint(i_index[i]*self.cl_scale, ave_dis*theta, 0)   
            
        self.cs_deform.SetPoints(self.cs_points)
        
        
#############################################

def np_img(obj, scale=255):
    y = np.array(obj.arrays[1])
    ymax, ymin = y.max(), y.min()
    ysize = int((ymax - ymin) // obj.cl_scale + 1)
    y = (y - ymin) // obj.cl_scale
    x = (np.array(obj.arrays[0]) + obj.cl_scale/2) // obj.cl_scale
    xsize = int(x.max() + 1)
    img = np.zeros((xsize, ysize), dtype=np.uint8)
    point_mapping = {}
    for i in range(x.shape[0]):
        img[int(x[i]), int(y[i])] = scale
        point_mapping[(int(x[i]), int(y[i]))] = i
    return img, point_mapping
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RNN depth")
    parser.add_argument("--chuck", type=str)
    parser.add_argument("--r_version", choices=['cs', 'global'])
    parser.add_argument("--closing", action="store_true")
    parser.add_argument("--opening", action="store_true")
    parser.add_argument("--disc_size", type=int, default=2)
    args = parser.parse_args()
    print(args)
    
    save_root = args.chuck + '_' + args.r_version
    objects = colon(args.chuck + '.obj')
    objects.set_centerline(200)
    objects.add_render(objects.centerline, 'centerline', (255, 255, 0))
    objects.cs_cut(args.r_version)
    objects.visual()
    visual_pc(objects.cs_deform, save_root + '.png')
    
    img, mapping = np_img(objects, scale=1)
    plt.imsave(save_root+'_2d.png', img)
    save_mor = save_root
    if args.closing:
        img = morphology.binary_closing(img, morphology.diamond(args.disc_size)).astype(np.uint8)
        save_mor += '_close'
    if args.opening:
        img = morphology.binary_opening(img, morphology.diamond(args.disc_size)).astype(np.uint8)
        save_mor += '_open'
    plt.imsave(save_mor+'.png', img)
    labeled, nr_objects = ndimage.label(img < 1) 
    print("Number of holes is {}".format(nr_objects))
    
    edges = skimage.feature.canny(image=img*255)
    plt.imshow(edges)