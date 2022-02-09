
import numpy as np

class BoundingBoxPoseEstimator():

    def __init__(self, image_height, image_width, focal_length):
        self.img_height = image_height
        self.img_width = image_width
        self.focal_length = focal_length

    def remove_bad_bounding_boxes(self, bounding_boxes):
        """
        Removes bounding boxes that are not relatively square.
        """
        ret = [bb for bb in bounding_boxes.bounding_boxes if self._is_good_bb(bb)]
        return ret

    def estimate_position_from_helipad_perimeter(self, bounding_boxes, perimeter_radius_mm=390, bb_scale_factor=0.97):
        best_bb = self._find_best_bb_of_class(bounding_boxes, "Helipad")

        if best_bb != None:
            center_px = self._est_center_of_bb(best_bb)
            radius_px = bb_scale_factor * self._est_radius_of_bb(best_bb)
            if not (all(center_px) and radius_px): # TODO: Check if these are ever run
                return None
        else:
            return None

        x_cam, y_cam, z_cam = self._pixel_to_camera_coordinates(center_px, radius_px, perimeter_radius_mm)
        x_ned_mm, y_ned_mm, z_ned_mm = self._camera_to_helipad_coordinates(x_cam, y_cam, z_cam)

        # Convert to meters
        x_ned = x_ned_mm / 1000
        y_ned = y_ned_mm / 1000
        z_ned = z_ned_mm / 1000

        return x_ned, y_ned, z_ned

    def estimate_rotation_from_helipad_arrow(self, bounding_boxes):

        helipad_arrow_bb = self._find_best_bb_of_class(bounding_boxes, "Arrow")

        if helipad_arrow_bb is None:
            return None

        center_arrow_px = self._est_center_of_bb(helipad_arrow_bb)

        helipad_perimeter_bb = self._find_best_bb_of_class(bounding_boxes, "Helipad")
        helipad_h_bb = self._find_best_bb_of_class(bounding_boxes, "H")

        # Calculate rotation from perimeter bounding box center and arrow
        if helipad_perimeter_bb is not None:
            center_perimeter_px = self._est_center_of_bb(helipad_perimeter_bb)
            rotation_perimeter = self._est_rotation(center_perimeter_px, center_arrow_px)
            if not all(center_perimeter_px): # TODO: Check if these are ever run
                return None

        # Calculate rotation from H bounding box center
        if helipad_h_bb is not None:
            center_h_px = self._est_center_of_bb(helipad_h_bb)
            rotation_h = self._est_rotation(center_h_px, center_arrow_px)
            if not all(center_h_px): # TODO: Check if these are ever run
                return None

        # Combine measurements if both available
        if (helipad_perimeter_bb is not None) and (helipad_h_bb is not None):
            rotation = (rotation_perimeter + rotation_h) / 2
        elif helipad_perimeter_bb is not None:
            rotation = rotation_perimeter
        elif helipad_h_bb is not None:
            rotation = rotation_h
        else:
            return None

        return rotation

    def _pixel_to_camera_coordinates(self, center_px, radius_px, radius_mm):

        # Image center
        x_c = self.img_width / 2
        y_c = self.img_height / 2

        # Distance from image center to object center
        d_x = x_c - center_px[0]
        d_y = y_c - center_px[1]

        # Find altitude from similar triangles
        z_camera = self.focal_length * radius_mm / radius_px

        # Find x and y coordiantes using similar triangles as well
        x_camera = -(z_camera * d_x / self.focal_length)
        y_camera = -(z_camera * d_y / self.focal_length)

        return x_camera, y_camera, z_camera

    def _camera_to_helipad_coordinates(self, x_camera, y_camera, z_camera):
        # TODO: These must be investigated. Maybe using a transformation matrix that
        # accounts for the rotation of the camera in addition to its linear offset.
        x_ned = x_camera
        y_ned = y_camera
        z_ned = z_camera

        return x_ned, y_ned, z_ned

    def _is_good_bb(self, bb):
        """
        Returns true for bounding boxes that are within the desired proportions,
        them being relatively square.
        Bbs that have one side being 5 times or longer than the other are discarded.

        input:
            bb: yolo bounding box

        output.
            discard or not: bool
        """
        bb_w = bb.xmax - bb.xmin
        bb_h = bb.ymax - bb.ymin
        if 0.2 > bb_w/bb_h > 5:
            return False
        else:
            return True

    def _est_radius_of_bb(self, bb):
        width = bb.xmax - bb.xmin
        height = bb.ymax - bb.ymin
        radius = (width + height)/4
        return radius

    def _est_center_of_bb(self, bb):
        width = bb.xmax - bb.xmin
        height = bb.ymax - bb.ymin
        center = [bb.xmin + width/2.0 ,bb.ymin + height/2.0]
        map(int, center)
        return center

    def _find_best_bb_of_class(self, bounding_boxes, classname):
        matches =  list(item for item in bounding_boxes if item.Class == classname)
        try:
            best = max(matches, key=lambda x: x.probability)
        except ValueError as e:
            best = None
        return best

    def _est_rotation(self, center, Arrow):
        """
        Estimates the quadcopter yaw rotation given the estimated center of the Helipad
        as well as the estimated center of the arrow. Quadcopter rotation is defined
        with respect to world frame coordinate axis.
        yaw=0 is when arrow is pointing at three-o-clock, and positively increasing when
            arrow is rotating the same direction as clock arm movement.
        y is defined 0 in top of the image, and increases positively downwards.

        input:
            center: np.array[2] - [x,y] pixel coordinates
            Arrow: np.array[2] - [x,y] pixel coordinates

        output:
            degs: float - estimated yaw angle of the quadcopter
        """
        dx = Arrow[0] - center[0]
        dy = center[1] - Arrow[1]
        rads = np.arctan2(dx, dy)
        degs = (rads*180 / np.pi - 180) % 360 - 180
        return degs