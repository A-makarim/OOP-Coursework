import math
import pybullet as p
import time
from abc import ABC, abstractmethod


class BaseGripper(ABC):
    """
    Abstract base class for robotic grippers.
    All grippers must implement open_gripper() and close_gripper() methods.
    """
    
    def __init__(self, position, orientation, urdf_file):
        """Initialize a generic gripper."""
        # Load the URDF file using PyBullet
        self.position = position
        self.orientation = orientation
        self.urdf_file = urdf_file

        self.gripper = p.loadURDF(
            self.urdf_file,
            basePosition=self.position,
            baseOrientation=self.orientation,
            useFixedBase=False
        )

        # Make the gripper massless or very light so it can move easily
        p.changeDynamics(self.gripper, -1, mass=0)

        self.active_joints, self.fixed_joints = self.get_joint_info()

    def get_joint_info(self):
        """Retrieve joint information for the gripper."""
        active_joints, fixed_joints = [], []
        num_joints = p.getNumJoints(self.gripper)
        for i in range(num_joints):
            info = p.getJointInfo(self.gripper, i)
            if info[2] != p.JOINT_FIXED:
                active_joints.append(i)
            else:
                fixed_joints.append(i)
        return active_joints, fixed_joints

    def get_id(self):
        """
        Return the PyBullet body ID of this gripper.
        This allows main.py to call p.getBasePositionAndOrientation(gripper.get_id()).
        """
        return self.gripper

    def sim_step(self, steps=500, delay=0.001):
        """Simulate physics for the given number of steps."""
        for _ in range(steps):
            p.stepSimulation()
            time.sleep(delay)

    @abstractmethod
    def open_gripper(self):
        """
        Open the gripper fingers.
        Must be implemented in subclasses.
        """
        pass

    @abstractmethod
    def close_gripper(self):
        """
        Close the gripper fingers to grasp.
        Must be implemented in subclasses.
        """
        pass

    def set_position(self, position, orientation):
        """Set the position and orientation of the gripper."""
        p.resetBasePositionAndOrientation(self.gripper, position, orientation)

    def move_up_smoothly(self, target_z, steps=200, delay=0.01):
        """Move the gripper up smoothly by setting a small linear velocity."""
        current_position, current_orientation = p.getBasePositionAndOrientation(
            self.gripper)
        start_z = current_position[2]
        delta_z = (target_z - start_z) / steps if steps > 0 else 0.0
        velocity = delta_z / delay if delay > 0 else 0.0

        for _ in range(steps):
            p.resetBaseVelocity(self.gripper, linearVelocity=[0, 0, velocity])
            p.stepSimulation()
            time.sleep(delay)

        # Stop any velocity
        p.resetBaseVelocity(self.gripper, linearVelocity=[0, 0, 0])


class PR2Gripper(BaseGripper):
    def __init__(self, position, orientation):
        """Initialize a PR2 gripper."""
        super().__init__(position, orientation, "pr2_gripper.urdf")

    def open_gripper(self):
        """Open the PR2 gripper."""
        open_positions = [
            0.548, 0.548]  # Typically ~ 0.54 is wide open for PR2
        for target_position, joint_index in zip(open_positions, self.active_joints):
            p.setJointMotorControl2(
                bodyIndex=self.gripper,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_position,
                force=100
            )
        self.sim_step(steps=200, delay=0.005)

    def close_gripper(self):
        """Close the PR2 gripper."""
        close_positions = [0.0, 0.0] # Close position
        for target_position, joint_index in zip(close_positions, self.active_joints):
            p.setJointMotorControl2(
                bodyIndex=self.gripper,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_position,
                force=500  # Increased force
            )
        # Increase friction for better grip
        for joint_index in self.active_joints:
            p.changeDynamics(self.gripper, joint_index, lateralFriction=1.0)

        self.sim_step(steps=200, delay=0.005)


class SDHGripper(BaseGripper):
    def __init__(self, position, orientation):
        """Initialize an SDH (Schunk Dexterous Hand) gripper."""
        import os
        # Use absolute path to the SDH URDF file
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sdh_path = os.path.join(current_dir, "models", "sdh.urdf")

        
        super().__init__(position, orientation, sdh_path)
        
        # SDH URDF has non-standard coordinate frame - apply correction rotation
        # This rotation aligns SDH's local frame with the world frame
        urdf_correction = p.getQuaternionFromEuler([0, -math.pi/2, 0])
        
        # Combine the desired orientation with the URDF correction
        corrected_orientation = p.multiplyTransforms(
            [0, 0, 0], orientation,
            [0, 0, 0], urdf_correction
        )[1]
        
        # Apply the corrected orientation
        p.resetBasePositionAndOrientation(self.gripper, position, corrected_orientation)

        
        # Set initial finger positions to be spread outward (negative values open fingers)
        for joint_index in self.active_joints:
            p.resetJointState(self.gripper, joint_index, targetValue=-0.5)
            p.setJointMotorControl2(
                bodyIndex=self.gripper,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=-0.5,
                force=100
            )
        
    def open_gripper(self):
        """Open the SDH gripper."""
        # SDH has 3 fingers with multiple joints each
        # Negative values spread fingers outward
        for joint_index in self.active_joints:
            p.setJointMotorControl2(
                bodyIndex=self.gripper,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=-0.5,  # Negative value to spread outward
                force=100
            )
        self.sim_step(steps=200, delay=0.005)

    def close_gripper(self, fraction=1.0, target_positions=None):
        """Close the SDH gripper.

        Parameters
        - fraction: float in [0,1]. 0.0 leaves the gripper at the open pose; 1.0 fully closes.
        - target_positions: explicit list of joint targets (overrides fraction if provided).
        """
        # Close all active joints to grasp
        # Allow a partial close by accepting an optional fraction argument
        # fraction=0.0 -> open position (-0.5), fraction=1.0 -> fully closed (2.0)
        close_positions_full = [2.0] * len(self.active_joints)
        open_positions = [-0.5] * len(self.active_joints)

        # Determine target positions per joint
        if target_positions is None:
            # Clamp fraction
            if fraction is None:
                fraction = 1.0
            fraction = max(0.0, min(1.0, float(fraction)))
            target_positions = [open_p + fraction * (close_p - open_p)
                                for open_p, close_p in zip(open_positions, close_positions_full)]

        for joint_index, target_position in zip(self.active_joints, target_positions):
            p.setJointMotorControl2(
                bodyIndex=self.gripper,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_position,
                force=500
            )
            # Increase friction for better grip
            p.changeDynamics(self.gripper, joint_index, lateralFriction=1.0)

        self.sim_step(steps=200, delay=0.005)


class CustomGripper(BaseGripper):
    def __init__(self, position, orientation):
        """Initialize a custom gripper."""
        super().__init__(position, orientation, "custom_gripper.urdf")

    def open_gripper(self):
        """Open the custom gripper."""
        print("[INFO] CustomGripper: open_gripper not implemented, doing nothing.")

    def close_gripper(self):
        """Close the custom gripper."""
        print("[INFO] CustomGripper: close_gripper not implemented, doing nothing.")


def select_gripper(gripper_type, position, orientation):
    if gripper_type == "pr2":
        return PR2Gripper(position, orientation)
    elif gripper_type == "sdh":
        return SDHGripper(position, orientation)
    elif gripper_type == "custom":
        return CustomGripper(position, orientation)
    else:
        raise ValueError(
            "Unsupported gripper type. Supported types: 'pr2', 'sdh', 'custom'.")
