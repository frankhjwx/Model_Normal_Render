from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.GL.shaders import *
from PIL import Image
import numpy as np
import os

global angle, eye, look
xsize = 128
ysize = 128

class Point(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __sub__(self, p):
        return Point(self.x - p.x, self.y - p.y, self.z - p.z)


class Face(object):
    def __init__(self, x1, x2, x3):
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3


class Model(object):
    def __init__(self, p, f):
        self.p = p
        self.f = f


class Camera(object):
    def __init__(self, eye, center, up):
        self.eye = eye
        self.up = up
        self.center = center


class OcclusionRate(object):

    def read_objmodel(self, filename):
        with open(filename) as file:
            points = []
            f = []
            self.vertex_positions = []
            self.vertex_indices = []
            self.vertices = []
            while 1:
                line = file.readline()
                if not line:
                    break
                strs = line.split(' ')
                if strs[0] == "v":
                    points.append(Point(float(strs[1]), float(strs[2]), float(strs[3])))
                    self.vertex_positions.append(float(strs[1]))
                    self.vertex_positions.append(float(strs[2]))
                    self.vertex_positions.append(float(strs[3]))
                if strs[0] == "f":
                    if len(strs) == 4:
                        f.append(Face(int(strs[1]) - 1, int(strs[2]) - 1, int(strs[3]) - 1))
                        self.vertex_indices.append(int(strs[1]) - 1)
                        self.vertex_indices.append(int(strs[2]) - 1)
                        self.vertex_indices.append(int(strs[3]) - 1)
                    else:
                        p1 = int(strs[2].split("/")[0]) - 1
                        p2 = int(strs[3].split("/")[0]) - 1
                        p3 = int(strs[4].split("/")[0]) - 1

                        f.append(Face(p1, p2, p3))
                        self.vertex_indices.append(p1)
                        self.vertex_indices.append(p2)
                        self.vertex_indices.append(p3)
            model = Model(points, f)
            # print(np.array(self.vertex_positions).reshape(len(points),3))
            self.model = model
            # self.ModelStandarize()
            for i in range(len(f)):
                self.vertices.append([points[f[i].x1].x, points[f[i].x1].y, points[f[i].x1].z])
                self.vertices.append([points[f[i].x2].x, points[f[i].x2].y, points[f[i].x2].z])
                self.vertices.append([points[f[i].x3].x, points[f[i].x3].y, points[f[i].x3].z])
            # print(np.array(self.vertices))

        return model

    def read_camera(self):
        return Camera(Point(0, 0, 0), Point(0, -1, 0), Point(0, 0, -1))

    def vector_div(self, v1, v2):
        if v2.x != 0:
            return v1.x / v2.x
        if v2.y != 0:
            return v1.y / v2.y
        return v1.z / v2.z

    def single_triangle_area(self, p1, p2, p3):
        if type(p1) == Point:
            p1 = np.array([p1.x, p1.y, p1.z])
            p2 = np.array([p2.x, p2.y, p2.z])
            p3 = np.array([p3.x, p3.y, p3.z])
        e1 = p2 - p1
        e2 = p3 - p1
        return (np.linalg.norm(np.cross(e1, e2)) / 2)

    def triangle_area(self, points, faces):
        p = np.zeros((len(points), 3))
        for i in range(len(points)):
            p[i][0] = points[i].x
            p[i][1] = points[i].y
            p[i][2] = points[i].z
        e1 = np.zeros((len(faces), 3))
        e2 = np.zeros((len(faces), 3))
        for i in range(len(faces)):
            v1 = faces[i].x1
            v2 = faces[i].x2
            v3 = faces[i].x3
            e1[i] = p[v2] - p[v1]
            e2[i] = p[v3] - p[v1]

        S = np.linalg.norm(np.cross(e1, e2), axis=1) / 2
        return S

    def Savescene(self):
        glReadBuffer(GL_BACK)
        mdata = glReadPixels(0, 0, xsize, ysize, GL_RGB, GL_UNSIGNED_BYTE)
        image = Image.frombytes('RGB', (xsize, ysize), mdata)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        # path = "ShapeNetDepthRendering/" + self.p + "/" + str(self.itr) + ".jpg"
        #path = os.path.join(self.savedir, '%s_%d.jpg' % (self.p, self.itr))
        path = os.path.join(self.savedir, '%s_%d.bmp' % (self.p, self.itr))
        image.save(path)

        self.itr += 1

    def getModelCenter(self):
        ranges = [[100000, -100000],
                  [100000, -100000],
                  [100000, -100000]]
        for p in self.model.p:
            if p.x < ranges[0][0]:
                ranges[0][0] = p.x
            if p.x > ranges[0][1]:
                ranges[0][1] = p.x
            if p.y < ranges[1][0]:
                ranges[1][0] = p.y
            if p.y > ranges[1][1]:
                ranges[1][1] = p.y
            if p.z < ranges[2][0]:
                ranges[2][0] = p.z
            if p.z > ranges[2][1]:
                ranges[2][1] = p.z
        # print(ranges)
        return [(ranges[0][1] + ranges[0][0]) / 2, (ranges[1][1] + ranges[1][0]) / 2, (ranges[2][1] + ranges[2][0]) / 2]


    def ModelStandarize(self):
        mean = self.getModelCenter()
        # print(mean)
        for p in self.model.p:
            p.x -= mean[0]
            p.y -= mean[1]
            p.z -= mean[2]
        return


    def drawModel(self):

        #  print(Model, Projection)

        # glBindVertexArray(self.vao)
        # glDrawArrays(GL_TRIANGLES, 0, len(self.vertices))
        # glUseProgram(0)
        # glBindVertexArray(0)

        faces = self.model.f
        points = self.model.p
        k = 0
        #glColor3d(1,1,1)
        for f in faces:
            glBegin(GL_TRIANGLES)
            #glBegin(GL_POINTS)
            k = k + 1
            #k = k and 255
            a = points[f.x1]
            b = points[f.x2]
            c = points[f.x3]
            ab = Point(a.x - b.x, a.y - b.y, a.z - b.z)
            bc = Point(b.x - c.x, b.y - c.y, b.z - c.z)
            norm_x = (ab.y * bc.z) - (ab.z * bc.y)
            norm_y = -((ab.x * bc.z) - (ab.z * bc.x))
            norm_z = (ab.x * bc.y) - (ab.y * bc.x)
            r = np.sqrt(norm_x ** 2 + norm_y ** 2 + norm_z ** 2)
            if r != 0:
                norm_x /= r
                norm_y /= r
                norm_z /= r
            norm_x = (norm_x + 1) / 2
            norm_y = (norm_y + 1) / 2
            norm_z = (norm_z + 1) / 2
            glColor3ub(int(norm_x * 255), int(norm_y * 255), int(norm_z * 255))
            glVertex3f(points[f.x1].x, points[f.x1].y, points[f.x1].z)
            glVertex3f(points[f.x2].x, points[f.x2].y, points[f.x2].z)
            glVertex3f(points[f.x3].x, points[f.x3].y, points[f.x3].z)
            glEnd()


    def initialize(self):
        self.itr = 0
        self.triangle = self.triangle_area(self.model.p, self.model.f)

        return


    def __init__(self):
        self.vao = 0
        self.vertices = []
        self.vertex_positions = []
        self.vertex_indices = []
        self.model = self.read_objmodel("test.obj")
        self.camera = self.read_camera()
        self.triangle = 0
        self.initialize()


    def __init__(self, modelname, c):
        self.program = 0
        self.vao = 0
        self.modelname = modelname
        self.vertices = []
        self.vertex_positions = []
        self.vertex_indices = []
        self.model = self.read_objmodel(modelname)
        self.camera = self.read_camera()
        self.camera.eye.x = c[0]
        self.camera.eye.y = c[1]
        self.camera.eye.z = c[2]
        self.camera.center.x = c[3]
        self.camera.center.y = c[4]
        self.camera.center.z = c[5]
        self.camera.up.x = c[6]
        self.camera.up.y = c[7]
        self.camera.up.z = c[8]
        self.triangle = 0
        self.initialize()


    def setcamera(self, c):
        self.camera.eye.x = c[0]
        self.camera.eye.y = c[1]
        self.camera.eye.z = c[2]
        self.camera.center.x = c[3]
        self.camera.center.y = c[4]
        self.camera.center.z = c[5]
        self.camera.up.x = c[6]
        self.camera.up.y = c[7]
        self.camera.up.z = c[8]
        return


    def __init__(self, modelname):
        self.program = 0
        self.vao = 0
        self.modelname = modelname
        self.savedir = ''
        self.vertices = []
        self.vertex_positions = []
        self.vertex_indices = []
        self.model = self.read_objmodel(modelname)
        self.camera = self.read_camera()
        self.camera.eye.x = 0
        self.camera.eye.y = 0
        self.camera.eye.z = 0
        self.camera.center.x = 0
        self.camera.center.y = 0
        self.camera.center.z = 0
        self.camera.up.x = 0
        self.camera.up.y = 0
        self.camera.up.z = 0
        self.triangle = 0
        self.initialize()


    def setmodel(self, modelname, p):
        self.program = 0
        self.vao = 0
        self.modelname = modelname
        self.vertices = []
        self.vertex_positions = []
        self.vertex_indices = []
        self.vertex_colors = []
        self.model = self.read_objmodel(modelname)
        self.camera = self.read_camera()
        self.camera.eye.x = 0
        self.camera.eye.y = 0
        self.camera.eye.z = 0
        self.camera.center.x = 0
        self.camera.center.y = 0
        self.camera.center.z = 0
        self.camera.up.x = 0
        self.camera.up.y = 0
        self.camera.up.z = 0
        self.triangle = 0
        self.p = p
        self.initialize()


def get_view_point(r, theta, phi):
    # degree to radius
    theta = theta / 180.0 * np.pi
    phi = phi / 180.0 * np.pi
    x = r * np.sin(theta) * np.sin(phi)
    y = r * np.cos(theta)
    z = r * np.sin(theta) * np.cos(phi)
    sight = np.array([x, y, z])
    yaxis = np.array([0, 1.0, 0.0])
    if np.fabs(theta) < 1.0e-6:
        up = np.array([-1.0, 0, 0.0])
    elif np.fabs(theta - np.pi) < 1.0e-6:
        up = np.array([1.0, 0, 0])
    else:
        horizontal = np.cross(yaxis, sight)
        horizontal /= np.linalg.norm(horizontal)
        up = np.cross(sight, horizontal)
        up /= np.linalg.norm(up)

    return (sight, up)


def drawFunc(modelname, angle):
    glEnable(GL_DEPTH_TEST)
    glDisable(GL_LINE_SMOOTH)
    glClearDepth(1.0)
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    eye, up = get_view_point(2.5, 70, angle % 360)

    # eye, up = get_view_point(2, 0, 0)
    # c = ocr.camera
    glPushMatrix()
    glLoadIdentity()
    gluLookAt(eye[0], eye[1], eye[2], 0, 0, 0, up[0], up[1], up[2])
    glPushMatrix()
    # glRotate(90, 1, 0, 0)
    # glScalef(3, 3, 3)
    ocr.drawModel()
    glPopMatrix()
    glPopMatrix()

    ocr.Savescene()
    glutSwapBuffers()


def drawmodel(modelname, p):
    angle = 0
    ocr.setmodel(modelname, p)
    while angle < 360:
        drawFunc(modelname, angle)
        angle += 22.5
    return


def reshape(w, h):
    glViewport(0, 0, xsize, ysize)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    if w <= h:
        # glOrtho(-0.5, 0.5, -0.5*h/w, 0.5*h/w, -10.0, 10.0)
        # glFrustum(-1, 1, -1 * ysize / xsize, 1 * ysize / xsize, 1.866, 10.0)
        gluPerspective(30, 1.0, 1.866, 10)
    else:
        # glOrtho(-0.5*w/h, 0.5*w/h, -0.5, 0.5, -10.0, 10.0)
        # glFrustum(-1, 1, -1, 1, 1.866, 10.0)
        gluPerspective(30, 1.0, 1.866, 10)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


if __name__ == '__main__':

    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)

    glutInitWindowPosition(0, 0)
    glutInitWindowSize(xsize, ysize)
    glutCreateWindow(b"test")

    glViewport(0, 0, xsize, ysize)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    # glFrustum(-1, 1, -1 * ysize / xsize, 1 * ysize / xsize, 1.866, 10.0)
    gluPerspective(30, 1.0, 1.866, 10)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    global ocr
    ocr = OcclusionRate("test.obj")

    drawmodel("test.obj", "test")
