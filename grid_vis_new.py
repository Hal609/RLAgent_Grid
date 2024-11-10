from panda3d.core import Point3, LVector3, LVector4, DirectionalLight, CullFaceAttrib, AntialiasAttrib
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
import numpy as np
import random
from time import sleep
import math

class VoxelGrid(ShowBase):
    def __init__(self, numpy_grid, tick_rate):
        super().__init__()

        self.update_func: function

        self.done = False
        self.tick_rate = tick_rate
        self.grid2d = numpy_grid

        self.cam_pos = Point3(0, 0, 0)

        self.render.setShaderAuto()  # Enable shaders
        
        # Set up camera
        self.camLens.setNearFar(1, 500)

        # Set up lighting
        self.setup_lighting()

        # Create the voxel grid
        self.create_voxel_grid(self.grid2d)
        self.create_floor()

        self.taskMgr.add(self.spin_camera, "spin_camera_task")
        self.taskMgr.doMethodLater(self.tick_rate, self.update_grid_task, "update_grid_task")

        # self.render.setDepthTest(True)
        # self.render.setDepthWrite(True)

    def update_grid_task(self, task):
        new = self.update_func(self.grid2d)
        if new is None: self.userExit()
        new_grid = np.array(new)
        self.update_grid(new_grid)
        self.update_floor()
        return Task.again
    
    def update_grid(self, new_grid):
        if (new_grid == self.grid2d).all(): return
        for node in self.render.findAllMatches("=type=GridCube"):
            node.removeNode()
        self.grid2d = new_grid
        self.create_voxel_grid(self.grid2d)

    def update_floor(self):
        floor = self.render.find("=type=Floor")
        floor.setSx(self.grid2d.shape[0] + 2)
        floor.setSy((self.grid2d.shape[1] + 2))
        floor.setPos(Point3(self.grid2d.shape[0]/2 - 1, self.grid2d.shape[1]/2 - 1, -1))

    def setup_lighting(self):
        self.light = self.render.attachNewNode(DirectionalLight("DirectionalLights"))
        self.light.node().setDirection(LVector3(-1, -0.5, -1))
        self.light.node().setScene(self.render)
        self.light.node().setColor(LVector4(0.7, 0.7, 0.7, 1))
        self.light.node().setShadowCaster(True)
        self.render.setLight(self.light)

        self.light2 = self.render.attachNewNode(DirectionalLight("DirectionalLights"))
        self.light2.node().setDirection(LVector3(1, 1, -1))
        self.light2.node().setScene(self.render)
        self.light2.node().setColor(LVector4(0.5, 0.5, 0.5, 1))
        self.light2.node().setShadowCaster(True)
        self.render.setLight(self.light2)

        # Important! Enable the shader generator.
        self.render.setShaderAuto()

    def create_floor(self):
        voxel = self.loader.loadModel("models/box")
        voxel.setTag("type", "Floor")
        voxel.setSx(self.grid2d.shape[0] + 2)
        voxel.setSy((self.grid2d.shape[1] + 2))
        voxel.setPos(Point3(self.grid2d.shape[0]/2 - 1, self.grid2d.shape[1]/2 - 1, -1))
        colour = (0.647, 0.509, 0.444)
        voxel.setColor(*colour, 1)

        voxel.setAttrib(CullFaceAttrib.make(CullFaceAttrib.MCullNone))
        
        voxel.reparentTo(self.render)

    def create_voxel_grid(self, grid):
        """Creates a grid of small cubes."""
        width, depth = grid.shape
        for x in range(width):
            for y in range(depth):
                if int(self.grid2d[x][y][:-2], 16) != 0:
                    self.create_voxel(x, y, 0, value=self.grid2d[x][y])

    def hex_to_rgb_tuple(self, hex_colour):
        hex_colour = str(f'{hex_colour:06}').lstrip('#')
        return tuple(int(hex_colour[i:i+2], 16)/255 for i in (0, 2, 4))
    
    def create_voxel(self, x, y, z, value):
        """Creates a single voxel cube at a given position."""
        voxel = self.loader.loadModel("models/box")
        voxel.setTag("type", "GridCube")

        size = int(value[-2:], 16)/255 if len(value) == 8 else 1
        voxel.setPos(Point3(x - 1/2, y - 1/2, z))
        voxel.setScale(size)

        colour = self.hex_to_rgb_tuple(value)
        voxel.setColor(*colour, 1)

        voxel.setAttrib(AntialiasAttrib.make(AntialiasAttrib.M_line))
        voxel.setAttrib(CullFaceAttrib.make(CullFaceAttrib.MCullNone))

        # voxel.setShaderAuto()
        voxel.reparentTo(self.render)

    def spin_camera(self, task):
        """Rotate camera around the grid."""
        angleDegrees = task.time * 0.2
        angleRadians = angleDegrees * (3.14159 / 180.0)
        centre = (self.grid2d.shape[0]/2, self.grid2d.shape[1]/2)
        distance = 2 + 2 * max(self.grid2d.shape[0], self.grid2d.shape[1])
        self.cam_pos = LVector3(centre[0] + math.sin(angleRadians)*distance, centre[1] + math.cos(angleRadians)*distance, 25)

        self.camera.setPos(self.cam_pos)
        self.camera.lookAt(*centre, 0)
        return Task.cont


def blank(grid):
    return grid

def run_grid(npgrid, update_func=blank, tick_rate=0.1):
    npgrid = np.array(npgrid)
    app = VoxelGrid(npgrid, tick_rate)
    app.update_func = update_func
    app.run()
