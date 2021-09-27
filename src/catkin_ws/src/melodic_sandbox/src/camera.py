#!/usr/bin/env python3

import rospy
import olympe
import time
import olympe.messages.camera as camera
import olympe.messages.gimbal as gimbal
import olympe.messages.ardrone3.Piloting as piloting
import olympe.messages.ardrone3.PilotingState as piloting_state
import olympe.media as media
import xml.etree.ElementTree as ET
import re

DRONE_IP = "10.202.0.1"

XMP_TAGS_OF_INTEREST = (
    "CameraRollDegree",
    "CameraPitchDegree",
    "CameraYawDegree",
    "CaptureTsUs",
    "GPSLatitude",
    "GPSLongitude",
    "GPSAltitude",
)

class TestCamera():

    def __init__(self, ip):
        self.drone = olympe.Drone(ip)
        self.drone.logger.setLevel(40)
        self.drone.connect()
    
    def start(self):
        self.init_camera()
        self.init_gimbal()
        self.takeoff()
        self.take_picture()
        self.pitch_camera(-45)
        self.take_picture()
        self.pitch_camera(-90)
        self.take_picture()
        self.pitch_camera(0)
        self.move(10, 0, 0, 3.14)
        self.take_picture()
        self.pitch_camera(-45)
        self.take_picture()
        self.move(10, 0, 0, 0)
        self.land()
        self.drone.disconnect()

    def takeoff(self):
        rospy.loginfo("Taking off")
        assert self.drone(
            piloting.TakeOff()
            >> piloting_state.FlyingStateChanged(state="hovering", _timeout=5)
        ).wait().success()
        rospy.loginfo("Takeoff complete")

    def land(self):
        rospy.loginfo("Landing")
        assert self.drone(piloting.Landing()).wait().success()

    def move(self, dx, dy, dz, dpsi):
        rospy.loginfo(f"Moving dx: {dx} dy: {dy} dz: {dz} dpsi: {dpsi}")
        assert self.drone(
            piloting.moveBy(dx, dy, dz, dpsi)
            >> piloting_state.FlyingStateChanged(state="hovering", _timeout=5)
        ).wait().success()
        rospy.loginfo("Reached target")

    def init_camera(self):
        self.drone(camera.set_camera_mode(cam_id=0, value="photo")).wait()
        self.drone(
            camera.set_photo_mode(
                cam_id=0,
                mode="single",
                format="rectilinear",
                file_format="jpeg",
                burst="burst_14_over_1s",
                bracketing="preset_1ev",
                capture_interval=1
            )
        ).wait().success()

        rospy.loginfo("Initialized camera")

    def take_picture(self):
        rospy.loginfo("Taking photo")
        
        photo_saved = self.drone(camera.photo_progress(result="photo_saved", _policy="wait"))
        self.drone(camera.take_photo(cam_id=0)).wait()
        rospy.loginfo("Photo taken")
        assert photo_saved.wait(_timeout=0.5).success(), "take_photo timeout"
        rospy.loginfo("Photo saved")

        media_id = photo_saved.received_events().last().args["media_id"]
        self.drone.media.download_dir = "./"
        rospy.loginfo(f"Downloading photo for media id {media_id} in {self.drone.media.download_dir}")
        media_download = self.drone(media.download_media(media_id, integrity_check=False))
        resources = media_download.as_completed(timeout=0.1)
        for resource in resources:
            if not resource.success():
                rospy.logerr("Failed to download photo")
            with open(resource.download_path, "rb") as image_file:
                image_data = image_file.read()
                image_xmp_start = image_data.find(b"<x:xmpmeta")
                image_xmp_end = image_data.find(b"</x:xmpmeta")
                image_xmp = ET.fromstring(image_data[image_xmp_start: image_xmp_end + 12])
                for image_meta in image_xmp[0][0]:
                    xmp_tag = re.sub(r"{[^}]*}", "", image_meta.tag)
                    xmp_value = image_meta.text
                    # only print the XMP tags we are interested in
                    if xmp_tag in XMP_TAGS_OF_INTEREST:
                        rospy.loginfo("{} {} {}".format(resource.resource_id, xmp_tag, xmp_value))

    def init_gimbal(self):
        rospy.loginfo("Resetting gimbal orientation")
        self.drone(gimbal.reset_orientation(gimbal_id=0))
        assert self.drone(gimbal.attitude(
            gimbal_id=0,
            roll_relative=0,
            pitch_relative=0,
            yaw_relative=0
        )).wait().success(), "Failed to reset gimbal orientation"
        rospy.loginfo("Setting gimbal max speed")

        # Max speeds: Pitch 180, Roll/Yaw 0.
        max_speed = 90

        self.drone(gimbal.set_max_speed(
            gimbal_id=0,
            yaw=0,
            pitch=max_speed,
            roll=0,
        ))

        assert self.drone(gimbal.max_speed(
            gimbal_id=0,
            current_yaw=0,
            current_pitch=max_speed,
            current_roll=0,
        )).wait().success(), "Failed to set max gimbal speed"

        rospy.loginfo("Gimbal initialized")


    def pitch_camera(self, angle):
        rospy.loginfo(f"Pitching camera: {angle}")

        self.drone(gimbal.set_target(
            gimbal_id=0,
            control_mode="position",
            pitch_frame_of_reference="relative",
            pitch=angle,
            roll_frame_of_reference="relative",
            roll=0,
            yaw_frame_of_reference="relative",
            yaw=0
        ))

        assert self.drone(gimbal.attitude(
            gimbal_id=0,
            pitch_relative=angle,
        )).wait(5).success(), "Failed to pitch camera"
        
        rospy.loginfo("Done pitching camera")
        


def main():
    rospy.init_node("camera_test", anonymous=False)
    mission = TestCamera(DRONE_IP)
    mission.start()

if __name__ == "__main__":
    main()