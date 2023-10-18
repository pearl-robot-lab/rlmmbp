import numpy as np
from learned_robot_placement.utils.spatial_utils import Rotation, Transform 
from copy import deepcopy as cp

class Visualizer:
    def __init__(self) -> None:
        from omni.isaac.debug_draw import _debug_draw as dd
        self.dd = dd.acquire_debug_draw_interface()
        
        # colors
        self.blue = (0, 0, 1, 1)
        self.red = (1, 0, 0, 1)
        self.green = (0, 1, 0, 1)
        self.light_blue = (0.68, 0.85, 0.9, 1)
        self.pink = (1, 0.75, 0.8, 1)
        self.light_green = (0.7, 1, 0.7, 1)
        self.light_grey = (0.8, 0.8, 0.8, 1)
        self.mid_grey = (0.5, 0.5, 0.5, 1)
        self.light_lilac = (0.8, 0.8, 1, 1)
        self.mid_blue = (0.34, 0.43, 1, 1)
        self.light_teal = (0.6, 0.86, 0.85, 1)

    def draw_frame_matrix(self, frame_matrix, point_size=5, axis_length=0.3, complement_length=0.0) -> None:
        """
        Draw a frame matrix as a set of three lines representing the x (red), y (green) and z (blues) axes
        :param frame_matrix: 4x4 matrix representing the frame
        :param point_size: size of the point size used to draw the axes
        :param axis_length: length of the axes in m
        :return:
        """
        frame = np.asarray(frame_matrix)
        # if axis length is an array
        if type(axis_length) is np.ndarray:
            if len(axis_length) != 3:
                raise ValueError("axis_length must be a scalar or a 3D array")
            axis_lengths = axis_length
        else:
            axis_lengths = np.ones(3) * axis_length
        frame_template = [np.array([[0.0], [0.0], [0.0], [1]]), # center point
                          np.array([[axis_lengths[0]], [0.0], [0.0], [1]]),
                          np.array([[0.0], [axis_lengths[1]], [0.0], [1]]),
                          np.array([[0.0], [0.0], [axis_lengths[2]], [1]])]
        
        c, x, y, z = [frame @ point for point in frame_template]

        point_list_start = [c, c, c]
        point_list_end = [x, y, z]
        self.dd.draw_lines(point_list_start, point_list_end, [self.red, self.green, self.blue], [point_size]*3)

        if complement_length != 0.0:
            com_frame = frame.copy()
            com_frame[:3,:3] = -com_frame[:3,:3]
            com_frame[:3,-1] += complement_length
            self.draw_frame_matrix(com_frame)

    def draw_frame_pos_quat(self, frame_pos_quat, point_size=5, axis_length=0.3) -> None:
        """
        Draw a frame matrix as a set of three lines representing the x (red), y (green) and z (blues) axes
        :param frame_pos_quat: tuple of (pos, quat), np.arras of shape (3,) and (4,) respectively; quat: [w, x, y, z]
        :param point_size: size of the point size used to draw the axes
        :param axis_length: length of the axes in m
        :return:
        """
        pos, quat = frame_pos_quat
        T = Transform.from_translation(pos)
        T.rotation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])

        self.draw_frame_matrix(T.as_matrix(), point_size, axis_length)

    def draw_voxels_from_center(self, pcl, voxel_size, point_size=3, colors=None) -> None:
        """
        Draw voxels as cubes around a list of points as centers
        :param pcl: np.array of shape (N, 3) or list of np.arrays of shape (3,) representing the point cloud 
        :param voxel_size: sidelength of the voxels
        :param point_size: size of the point size used to draw the cubes
        :param colors: color (RGBA tupel)/colors (list of N) used 
        :return:
        """
        if colors is None: colors = [self.mid_blue]*len(pcl)
        if type(colors) is tuple: colors = [colors]*len(pcl)

        box_template = [np.array([-voxel_size / 2, -voxel_size / 2, -voxel_size / 2]),
                        np.array([-voxel_size / 2, -voxel_size / 2,  voxel_size / 2]),
                        np.array([-voxel_size / 2,  voxel_size / 2, -voxel_size / 2]),
                        np.array([-voxel_size / 2,  voxel_size / 2,  voxel_size / 2]),
                        np.array([ voxel_size / 2, -voxel_size / 2, -voxel_size / 2]),
                        np.array([ voxel_size / 2, -voxel_size / 2,  voxel_size / 2]),
                        np.array([ voxel_size / 2,  voxel_size / 2, -voxel_size / 2]),
                        np.array([ voxel_size / 2,  voxel_size / 2,  voxel_size / 2])]
        
        for point, color in zip(pcl, colors):
            m_m_m, m_m_p, m_p_m, m_p_p, p_m_m, p_m_p, p_p_m, p_p_p = point + box_template
            point_list_start =  [m_m_m, m_m_m, m_m_m, m_m_p, m_m_p, m_p_m, m_p_m, p_m_m, p_m_m, m_p_p, p_m_p, p_p_m]
            point_list_end =    [m_m_p, m_p_m, p_m_m, m_p_p, p_m_p, m_p_p, p_p_m, p_m_p, p_p_m, p_p_p, p_p_p, p_p_p]

            self.dd.draw_lines(point_list_start, point_list_end, [color]*len(point_list_start), [point_size]*len(point_list_start))

    def draw_points(self, pcl, point_size=10, colors=None) -> None:
        """
        Draw points/spheres a list of points as centers
        :param pcl: array-like of shape (N, 3) or representing the point cloud 
        :param point_size: size of the point size 
        :param colors: colors used list of N RGBA colors
        :return:
        """
        point_tuples = []
        for point in pcl:
            point_tuples.append(tuple(point))

        if colors is None: colors = [self.pink]*len(point_tuples)
        elif colors=='light_green': colors = [self.light_green]*len(point_tuples)
        elif type(colors) is tuple: colors = [colors]*len(point_tuples)

        self.dd.draw_points(point_tuples, colors, [point_size] * len(point_tuples))

    def draw_box_min_max(self, box_min, box_max, point_size=1, color=None) -> None:
        """
        Draw a box as a set of lines
        :param box_min: np.array of shape (3,) representing the minimum corner of the box
        :param box_max: np.array of shape (3,) representing the maximum corner of the box
        :param point_size: size of the point size used to draw the box
        :param color: color used to draw the box
        :return:
        """
        if color is None: color = self.light_grey

        box_template = [np.array([box_min[0], box_min[1], box_min[2]]),
                        np.array([box_min[0], box_min[1], box_max[2]]),
                        np.array([box_min[0], box_max[1], box_min[2]]),
                        np.array([box_min[0], box_max[1], box_max[2]]),
                        np.array([box_max[0], box_min[1], box_min[2]]),
                        np.array([box_max[0], box_min[1], box_max[2]]),
                        np.array([box_max[0], box_max[1], box_min[2]]),
                        np.array([box_max[0], box_max[1], box_max[2]])]

        m_m_m, m_m_p, m_p_m, m_p_p, p_m_m, p_m_p, p_p_m, p_p_p = box_template
        point_list_start =  [m_m_m, m_m_m, m_m_m, m_m_p, m_m_p, m_p_m, m_p_m, p_m_m, p_m_m, m_p_p, p_m_p, p_p_m]
        point_list_end =    [m_m_p, m_p_m, p_m_m, m_p_p, p_m_p, m_p_p, p_p_m, p_m_p, p_p_m, p_p_p, p_p_p, p_p_p]

        self.dd.draw_lines(point_list_start, point_list_end, [color]*len(point_list_start), [point_size]*len(point_list_start))

    def draw_views(self, views, colors=None, draw_frames=True, point_size=1, axis_length=0.1, connect_views=False, intrinsics=None) -> None:
        """
        Draw voxels as cubes around a list of points as centers
        :param views: list of Transform objects representing the views (z pointing towards view direction)
        :param draw_frames: Boolean indicating whether to draw the frame of the view
        :param point_size: size of the point size used to draw the camera pose & frame
        :param axis_length: length of the frame axis in m
        :param connect_views: Boolean indicating whether to draw lines connecting the views' origins
        :param intrinsics: list of intrinsic parameters: [fx, fy, width, height]
        :return:
        """
        if colors is None: colors = [self.light_green]*len(views)

        if intrinsics is None:
            camera_template = [Transform.from_translation([0.0, 0.0, 0.0]), # center point
                            Transform.from_translation([-0.032, -0.024, 0.03]),
                            Transform.from_translation([-0.032, 0.024, 0.03]),
                            Transform.from_translation([0.032, -0.024, 0.03]),
                            Transform.from_translation([0.032, 0.024, 0.03])]
        else:
            fx, fy, width, height = intrinsics
            half_aperture_x = width/fx * 0.03
            half_aperture_y = height/fy * 0.03

            camera_template = [Transform.from_translation([0.0, 0.0, 0.0]), # center point
                            Transform.from_translation([-half_aperture_x, -half_aperture_y, 0.03]),
                            Transform.from_translation([-half_aperture_x, half_aperture_y, 0.03]),
                            Transform.from_translation([half_aperture_x, -half_aperture_y, 0.03]),
                            Transform.from_translation([half_aperture_x, half_aperture_y, 0.03])]

        camera_poses = [[view * point for point in camera_template] for view in views]
        
        for index, view_sign_points in enumerate(camera_poses):
            c, lb, lt, rb, rt = [tuple(point.translation) for point in view_sign_points]
            point_list_start = [lt, lb, c, lt, rt, c, rb, rt, rb]
            point_list_end = [lb, c, lt, rt, c, rb, rt, rb, lb]
            self.dd.draw_lines(point_list_start, point_list_end, [colors[index]] * len(point_list_start), [point_size] * len(point_list_start))

        if draw_frames:
            for T in views:
                T = T.as_matrix()
                self.draw_frame_matrix(T, point_size=point_size, axis_length=axis_length)

        if connect_views:
            point_list_start = [view.translation for view in views]
            point_list_end = cp(point_list_start[1:])
            point_list_start = point_list_start[:-1]

            self.dd.draw_lines(point_list_start, point_list_end, colors[:-1], [point_size]*len(point_list_start))

    def interpolate_quality_colors(self, qualities, qual_min=None, qual_max=None, opacity=1) -> list:
        """
        Interpolate colors from red to green according to qual_min and qual_max
        :param qualities: list of N qualities
        :param qual_min: minimum quality value
        :param qual_max: maximum quality value
        :param opacity: opacity of the colors
        :return: list of N RGBA colors
        """
        if qualities is None or len(qualities)==0: return [(0, 1, 0, opacity)] # green
        if qual_min is None: qual_min = np.min(qualities)
        if qual_max is None: qual_max = np.max(qualities)

        if qual_max > qual_min:
            colors = []
            for q in qualities:
                g = (q - qual_min) / (qual_max - qual_min)
                r = 1.0 - g
                colors.append((r, g, 0, opacity))

            return colors    
        else:
            return [(0, 1, 0, opacity)]*len(qualities) # green

    def draw_grasps_at(self, grasp_poses, depth=0.05, width=0.05, point_size=5, colors=None):
        """
        Creates a list of VisualCylinder objects representing the grasp
        :param poses: Transform representing the grasp poses
        :param depth: depth of the grasp
        :param width: width of the grasp
        :param radius: radius of the cylinders
        :param colors: color the grasps are drawn in
        :return:
        """
        if type(grasp_poses) is not list: grasp_poses = [grasp_poses]
        if colors is None: colors = [self.green]*len(grasp_poses)
        elif type(colors) is tuple: colors = [colors]*len(grasp_poses)

        # z pointing towards the grasp direction, y along the width
        grasp_template = [Transform.from_translation([0.0, -width/2,  depth]), 
                          Transform.from_translation([0.0, -width/2,      0]), 
                          Transform.from_translation([0.0,      0.0, -depth]), 
                          Transform.from_translation([0.0,      0.0,    0.0]), 
                          Transform.from_translation([0.0,  width/2,      0]), 
                          Transform.from_translation([0.0,  width/2,  depth])]
        
        grasp_poses = [[pose * point for point in grasp_template] for pose in grasp_poses]

        for index, grasp_sign_points in enumerate(grasp_poses):
            dl, ul, cu, cd, ur, dr = [tuple(point.translation) for point in grasp_sign_points]
            point_list_start = [dl, ul, ur, cu]
            point_list_end = [ul, ur, dr, cd]

            self.dd.draw_lines(point_list_start, point_list_end, [colors[index]] * len(point_list_start), [point_size] * len(point_list_start))
                          
    def clear_all(self) -> None:
        self.dd.clear_lines()
        self.dd.clear_points()