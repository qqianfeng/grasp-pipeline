def obj_to_string(obj, extra='    '):
    return str(obj.__class__) + '\n' + '\n'.join(
        (extra + (str(item) + ' = ' +
                  (obj_to_string(obj.__dict__[item], extra + '    ') if hasattr(obj.__dict__[item], '__dict__') else str(
                      obj.__dict__[item])))
         for item in sorted(obj.__dict__)))

class Header():
    def __init__(self, seq=None, stamp=None, frame_id=None):
        self.seq = seq
        self.stamp = stamp
        self.frame_id = frame_id
    def __str__(self):
        return obj_to_string(self)

class Point():
    def __init__(self, x=None, y=None, z=None):
        self.x = x
        self.y = y
        self.z = z
    def __str__(self):
        return obj_to_string(self)
    def __add__(self, point):
        return Point(x=self.x+point.x, y=self.y+point.y, z=self.z+point.z)
    def __sub__(self, point):
        return Point(x=self.x-point.x, y=self.y-point.y, z=self.z-point.z)

class Quaternion():
    def __init__(self, x=None, y=None, z=None, w=None):
        self.x = x
        self.y = y
        self.z = z
        self.w = w
    def __str__(self):
        return obj_to_string(self)
    def __add__(self, quat):
        return Quaternion(x=self.x+quat.x, y=self.y+quat.y, z=self.z+quat.z, w=self.w+quat.w)
    def __sub__(self, quat):
        return Quaternion(x=self.x-quat.x, y=self.y-quat.y, z=self.z-quat.z, w=self.w-quat.w)

class Pose():
    def __init__(self, position=Point(), orientation=Quaternion()):
        self.position = position
        self.orientation = orientation
    def __str__(self):
        return obj_to_string(self)
    def __add__(self, pose):
        return Pose(position=self.position+pose.position, orientation=self.orientation+pose.orientation)
    def __sub__(self, pose):
        return Pose(position=self.position-pose.position, orientation=self.orientation-pose.orientation)

class PoseStamped():
    def __init__(self, header=Header(), pose=Pose()):
        self.header = header
        self.pose = pose
    def __str__(self):
        return obj_to_string(self)

class JointState():
    def __init__(self, header=Header(), name=None, position=None, velocity=None, effort=None):
        self.header = header
        self.name = name
        self.position = position
        self.velocity = velocity
        self.effort = effort
    def __str__(self):
        return obj_to_string(self)
