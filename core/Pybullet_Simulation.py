from scipy.spatial.transform import Rotation as npRotation
from scipy.special import comb
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np
import math
import re
import time
import yaml

from Pybullet_Simulation_base import Simulation_base


# TODO: Rename class name after copying this file
class Simulation(Simulation_base):
    """A Bullet simulation involving Nextage robot"""

    def __init__(self, pybulletConfigs, robotConfigs, refVect=None):
        """Constructor
        Creates a simulation instance with Nextage robot.
        For the keyword arguments, please see in the Pybullet_Simulation_base.py
        """
        super().__init__(pybulletConfigs, robotConfigs)
        if refVect:
            self.refVector = np.array(refVect)
        else:
            self.refVector = np.array([1, 0, 0])

    ########## Task 1: Kinematics ##########
    # Task 1.1 Forward Kinematics
    jointRotationAxis = {
        'base_to_dummy': np.zeros(3),  # Virtual joint
        'base_to_waist': np.zeros(3),  # Fixed joint   ######!!!!
        # TODO: modify from here
        'CHEST_JOINT0': np.array([0, 0, 1]),
        'HEAD_JOINT0': np.array([0, 0, 1]),
        'HEAD_JOINT1': np.array([0, 1, 0]),
        'LARM_JOINT0': np.array([0, 0, 1]),
        'LARM_JOINT1': np.array([0, 1, 0]),
        'LARM_JOINT2': np.array([0, 1, 0]),
        'LARM_JOINT3': np.array([1, 0, 0]),
        'LARM_JOINT4': np.array([0, 1, 0]),
        'LARM_JOINT5': np.array([0, 0, 1]),
        'RARM_JOINT0': np.array([0, 0, 1]),
        'RARM_JOINT1': np.array([0, 1, 0]),
        'RARM_JOINT2': np.array([0, 1, 0]),
        'RARM_JOINT3': np.array([1, 0, 0]),
        'RARM_JOINT4': np.array([0, 1, 0]),
        'RARM_JOINT5': np.array([0, 0, 1])
    }

    frameTranslationFromParent = {
        'base_to_dummy': np.zeros(3),  # Virtual joint
        'base_to_waist': np.array([0, 0, 0.85]),  # Fixed joint
        # TODO: modify from here
        'CHEST_JOINT0': np.array([0, 0, 0.267]),
        'HEAD_JOINT0': np.array([0, 0, 0.302]),
        'HEAD_JOINT1': np.array([0, 0, 0.066]),
        'LARM_JOINT0': np.array([0.04, 0.135, 0.1015]),
        'LARM_JOINT1': np.array([0, 0, 0.066]),
        'LARM_JOINT2': np.array([0, 0.095, -0.25]),
        'LARM_JOINT3': np.array([0.1805, 0, -0.03]),
        'LARM_JOINT4': np.array([0.1495, 0, 0]),
        'LARM_JOINT5': np.array([0, 0, -0.1335]),
        'RARM_JOINT0': np.array([0.04, -0.135, 0.1015]),
        'RARM_JOINT1': np.array([0, 0, 0.066]),
        'RARM_JOINT2': np.array([0, -0.095, -0.25]),
        'RARM_JOINT3': np.array([0.1805, 0, -0.03]),
        'RARM_JOINT4': np.array([0.1495, 0, 0]),
        'RARM_JOINT5': np.array([0, 0, -0.1335])
    }

    jointOrderFK = {
        'base_to_dummy': [], # Virtual joint
        'base_to_waist': [], # Fixed joint
        'CHEST_JOINT0': ['base_to_waist'],
        'HEAD_JOINT0': ['CHEST_JOINT0', 'base_to_waist'],
        'HEAD_JOINT1': ['HEAD_JOINT0', 'CHEST_JOINT0', 'base_to_waist'],
        'LARM_JOINT0': ['CHEST_JOINT0', 'base_to_waist'],
        'LARM_JOINT1': ['LARM_JOINT0', 'CHEST_JOINT0', 'base_to_waist'],
        'LARM_JOINT2': ['LARM_JOINT1', 'LARM_JOINT0', 'CHEST_JOINT0', 'base_to_waist'],
        'LARM_JOINT3': ['LARM_JOINT2', 'LARM_JOINT1', 'LARM_JOINT0', 'CHEST_JOINT0', 'base_to_waist'],
        'LARM_JOINT4': ['LARM_JOINT3', 'LARM_JOINT2', 'LARM_JOINT1', 'LARM_JOINT0', 'CHEST_JOINT0', 'base_to_waist'],
        'LARM_JOINT5': ['LARM_JOINT4', 'LARM_JOINT3', 'LARM_JOINT2', 'LARM_JOINT1', 'LARM_JOINT0', 'CHEST_JOINT0', 'base_to_waist'],
        'RARM_JOINT0': ['CHEST_JOINT0', 'base_to_waist'],
        'RARM_JOINT1': ['RARM_JOINT0', 'CHEST_JOINT0', 'base_to_waist'],
        'RARM_JOINT2': ['RARM_JOINT1', 'RARM_JOINT0', 'CHEST_JOINT0', 'base_to_waist'],
        'RARM_JOINT3': ['RARM_JOINT2', 'RARM_JOINT1', 'RARM_JOINT0', 'CHEST_JOINT0', 'base_to_waist'],
        'RARM_JOINT4': ['RARM_JOINT3', 'RARM_JOINT2', 'RARM_JOINT1', 'RARM_JOINT0', 'CHEST_JOINT0', 'base_to_waist'],
        'RARM_JOINT5': ['RARM_JOINT4', 'RARM_JOINT3', 'RARM_JOINT2', 'RARM_JOINT1', 'RARM_JOINT0', 'CHEST_JOINT0', 'base_to_waist']
    }

    jointOrderIK = {
        'base_to_dummy': [], # Virtual joint
        'base_to_waist': [], # Fixed joint
        'CHEST_JOINT0': ['base_to_waist'],
        'HEAD_JOINT0': ['CHEST_JOINT0'],
        'HEAD_JOINT1': ['CHEST_JOINT0', 'HEAD_JOINT0'],
        'LARM_JOINT0': ['CHEST_JOINT0'],
        'LARM_JOINT1': ['CHEST_JOINT0', 'LARM_JOINT0'],
        'LARM_JOINT2': ['CHEST_JOINT0', 'LARM_JOINT0', 'LARM_JOINT1'],
        'LARM_JOINT3': ['CHEST_JOINT0', 'LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2'],
        'LARM_JOINT4': ['CHEST_JOINT0', 'LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2', 'LARM_JOINT3'],
        # 'LARM_JOINT5': ['CHEST_JOINT0', 'LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2', 'LARM_JOINT3', 'LARM_JOINT4'],
        'LARM_JOINT5': ['LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2', 'LARM_JOINT3', 'LARM_JOINT4'],
        'RARM_JOINT0': ['CHEST_JOINT0'],
        'RARM_JOINT1': ['CHEST_JOINT0', 'RARM_JOINT0'],
        'RARM_JOINT2': ['CHEST_JOINT0', 'RARM_JOINT0', 'RARM_JOINT1'],
        'RARM_JOINT3': ['CHEST_JOINT0', 'RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2'],
        'RARM_JOINT4': ['CHEST_JOINT0', 'RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2', 'RARM_JOINT3'],
        'RARM_JOINT5': ['CHEST_JOINT0', 'RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2', 'RARM_JOINT3', 'RARM_JOINT4']
    }






    def getJointRotationalMatrix(self, jointName=None, theta=None):
        """
            Returns the 3x3 rotation matrix for a joint from the axis-angle representation,
            where the axis is given by the revolution axis of the joint and the angle is theta.
        """
        if jointName == None:
            raise Exception("[getJointRotationalMatrix] \
                Must provide a joint in order to compute the rotational matrix!")
        # Hint: the output should be a 3x3 rotational matrix as a numpy array
        # return np.matrix()
        joint_rot_axis = self.jointRotationAxis.get(jointName)
        joint_x = joint_rot_axis[0]
        joint_y = joint_rot_axis[1]
        joint_z = joint_rot_axis[2]

        if joint_x == 1:
            rotation_matrix = np.matrix([[1, 0, 0],
                                         [0, np.cos(theta), -np.sin(theta)],
                                         [0, np.sin(theta), np.cos(theta)]])
        elif joint_y == 1:
            rotation_matrix = np.matrix([[np.cos(theta), 0, np.sin(theta)],
                                         [0, 1, 0],
                                         [-np.sin(theta), 0, np.cos(theta)]])
        elif joint_z == 1:
            rotation_matrix = np.matrix([[np.cos(theta), -np.sin(theta), 0],
                                         [np.sin(theta), np.cos(theta), 0],
                                         [0, 0, 1]])
        else:
            rotation_matrix = np.identity(3)

        return rotation_matrix

    def getTransformationMatrices(self):
        """
            Returns the homogeneous transformation matrices for each joint as a dictionary of matrices.
        """
        transformationMatrices = {}
        # Hint: the output should be a dictionary with joint names as keys and
        # their corresponding homogeneous transformation matrices as values.
        for joint_name in self.jointRotationAxis.keys():
            angle = self.getJointPos(joint_name)
            # Rotation
            result = self.getJointRotationalMatrix(jointName=joint_name, theta=angle)
            # Translation
            result = np.c_[result, self.frameTranslationFromParent[joint_name]]
            # homogenous
            result = np.concatenate((result, np.array([[0, 0, 0, 1]])), axis=0)
            transformationMatrices[joint_name] = result
        return transformationMatrices

    def getJointLocationAndOrientation(self, jointName):
        """
            Returns the position and rotation matrix of a given joint using Forward Kinematics
            according to the topology of the Nextage robot.
        """
        # Remember to multiply the transformation matrices following the kinematic chain for each arm.
        homogeneousTransformationMatrices = self.getTransformationMatrices()
        transformationOrder = self.jointOrderFK.get(jointName)
        JointToWorldFrame = homogeneousTransformationMatrices.get(jointName)
        for next_joint in transformationOrder:
            JointToWorldFrame = homogeneousTransformationMatrices.get(next_joint) * JointToWorldFrame
        return (JointToWorldFrame[0:3, 3]).squeeze(), JointToWorldFrame[0:3, 0:3]
        # Hint: return two numpy arrays, a 3x1 array for the position vector,
        # and a 3x3 array for the rotation matrix
        # return pos, rotmat
        pass

    def getJointPosition(self, jointName):
        """Get the position of a joint in the world frame, leave this unchanged please."""
        return self.getJointLocationAndOrientation(jointName)[0]

    def getJointOrientation(self, jointName, ref=None):
        """Get the orientation of a joint in the world frame, leave this unchanged please."""
        if ref is None:
            return np.array(self.getJointLocationAndOrientation(jointName)[1] @ self.refVector).squeeze()
        else:
            return np.array(self.getJointLocationAndOrientation(jointName)[1] @ ref).squeeze()

    def getJointAxis(self, jointName):
        """Get the orientation of a joint in the world frame, leave this unchanged please."""
        return np.array(self.getJointLocationAndOrientation(jointName)[1] @ self.jointRotationAxis[jointName]).squeeze()

    def jacobianMatrix(self, endEffector):
        """Calculate the Jacobian Matrix for the Nextage Robot."""
        joints = self.jointOrderIK[endEffector]
        jacobian = np.cross(self.jointRotationAxis[joints[0]], self.getJointPosition(endEffector) - self.getJointPosition(joints[0]))
        jacobian.reshape(1, 3)
        for joint in joints[1:]: # skip the first joint since we already calculated it
            # position
            temp = np.cross(self.jointRotationAxis[joint], self.getJointPosition(endEffector) - self.getJointPosition(joint))
            # orientation
            # temp = temp + np.cross(self.getJointAxis(joint), self.getJointAxis(endEffector))]
            jacobian = np.append(jacobian, temp, axis=0)
        return jacobian.T

        # You can implement the cross product yourself or use calculateJacobian().
        # Hint: you should return a numpy array for your Jacobian matrix. The
        # size of the matrix will depend on your chosen convention. You can have
        # a 3xn or a 6xn Jacobian matrix, where 'n' is the number of joints in
        # your kinematic chain.
        # return np.array()
        pass

    # Task 1.2 Inverse Kinematics

    def inverseKinematics(self, endEffector, targetPosition, orientation, interpolationSteps, maxIterPerStep,
                          threshold):
        """Your IK solver \\
        Arguments: \\
            endEffector: the jointName the end-effector \\
            targetPosition: final destination the the end-effector \\
            orientation: the desired orientation of the end-effector
                         together with its parent link \\
            interpolationSteps: number of interpolation steps
            maxIterPerStep: maximum iterations per step
            threshold: accuracy threshold
        Return: \\
            Vector of x_refs
        """
        curr_pos = self.getJointPosition(endEffector)
        # Calculate Jacobian
        jacobian = self.jacobianMatrix(endEffector)
        # Calculate dy
        dy = targetPosition - curr_pos
        # Calculate delta
        drad = np.linalg.pinv(jacobian) @ dy.T
        curr_q = {}
        for j, joint in enumerate(self.jointOrderIK[endEffector]):
            curr_q[joint] = self.getJointPos(joint) + drad[j,0]

        return curr_q
        # Hint: return a numpy array which includes the reference angular
        # positions for all joints after performing inverse kinematics.
        pass

    def move_without_PD(self, endEffector, targetPosition, speed=0.01, orientation=None,
                        threshold=1e-3, maxIter=3000, debug=False, verbose=False):
        """
        Move joints using Inverse Kinematics solver (without using PD control).
        This method should update joint states directly.
        Return:
            pltTime, pltDistance arrays used for plotting
        """
        # iterate through joints and update joint states based on IK solver
        interSteps = 100
        pltDistance = []
        pltTime = np.linspace(0, 1, interSteps)
        targets = np.linspace(self.getJointPosition(endEffector), targetPosition, interSteps)
        for target in targets:
            x_refs = self.inverseKinematics(endEffector, target, orientation, 50, maxIter, threshold)
            for joint in x_refs:
                self.p.resetJointState(self.robot, self.jointIds[joint], x_refs[joint])
            pltDistance.append(np.linalg.norm(self.getJointPosition(endEffector) - targetPosition))
            time.sleep(0.01)
        return pltTime, pltDistance
        pass

    def tick_without_PD(self):
        """Ticks one step of simulation without PD control. """
        # TODO modify from here
        # Iterate through all joints and update joint states.
        # For each joint, you can use the shared variable self.jointTargetPos.
        for joint, targetPos in self.jointTargetPos.items():
            self.p.resetJointState(self.robot, self.jointIds[joint], targetPos)

        self.p.stepSimulation()
        self.drawDebugLines()
        time.sleep(self.dt)

    ########## Task 2: Dynamics ##########
    # Task 2.1 PD Controller
    def calculateTorque(self, x_ref, x_real, dx_ref, dx_real, integral, kp, ki, kd):
        """ This method implements the closed-loop control \\
        Arguments: \\
            x_ref - the target position \\
            x_real - current position \\
            dx_ref - target velocity \\
            dx_real - current velocity \\
            integral - integral term (set to 0 for PD control) \\
            kp - proportional gain \\
            kd - derivetive gain \\
            ki - integral gain \\
        Returns: \\
            u(t) - the manipulation signal
        """
        error = x_ref - x_real
        derror = dx_ref - dx_real
        return kp * error + kd * derror + ki * integral
        pass

    # Task 2.2 Joint Manipulation
    def moveJoint(self, joint, targetPosition, targetVelocity, verbose=False):
        """ This method moves a joint with your PD controller. \\
        Arguments: \\
            joint - the name of the joint \\
            targetPos - target joint position \\
            targetVel - target joint velocity
        """

        def toy_tick(x_ref, x_real, dx_ref, dx_real, integral):
            # loads your PID gains
            jointController = self.jointControllers[joint]
            kp = self.ctrlConfig[jointController]['pid']['p']
            ki = self.ctrlConfig[jointController]['pid']['i']
            kd = self.ctrlConfig[jointController]['pid']['d']

            ### Start your code here: ###
            # Calculate the torque with the above method you've made
            torque = self.calculateTorque(x_ref, x_real, dx_ref, dx_real, integral, kp, ki, kd)
            ### To here ###

            pltTorque.append(torque)

            # send the manipulation signal to the joint
            self.p.setJointMotorControl2(
                bodyIndex=self.robot,
                jointIndex=self.jointIds[joint],
                controlMode=self.p.TORQUE_CONTROL,
                force=torque
            )
            # calculate the physics and update the world
            self.p.stepSimulation()
            time.sleep(self.dt)

        targetPosition, targetVelocity = float(targetPosition), float(targetVelocity)

        # disable joint velocity controller before apply a torque
        self.disableVelocityController(joint)
        # logging for the graph
        pltTime, pltTarget, pltTorque, pltTorqueTime, pltPosition, pltVelocity = [], [], [], [], [], []

        return pltTime, pltTarget, pltTorque, pltTorqueTime, pltPosition, pltVelocity

    def move_with_PD(self, endEffector, targetPosition, speed=0.01, orientation=None,
                     threshold=1e-3, maxIter=3000, debug=False, verbose=False):
        """
        Move joints using inverse kinematics solver and using PD control.
        This method should update joint states using the torque output from the PD controller.
        Return:
            pltTime, pltDistance arrays used for plotting
        """
        # TODO add your code here
        # Iterate through joints and use states from IK solver as reference states in PD controller.
        # Perform iterations to track reference states using PD controller until reaching
        # max iterations or position threshold.

        # Hint: here you can add extra steps if you want to allow your PD
        # controller to converge to the final target position after performing
        # all IK iterations (optional).

        # return pltTime, pltDistance
        pass

    def tick(self):
        """Ticks one step of simulation using PD control."""
        # Iterate through all joints and update joint states using PD control.
        for joint in self.joints:
            # skip dummy joints (world to base joint)
            jointController = self.jointControllers[joint]
            if jointController == 'SKIP_THIS_JOINT':
                continue

            # disable joint velocity controller before apply a torque
            self.disableVelocityController(joint)

            # loads your PID gains
            kp = self.ctrlConfig[jointController]['pid']['p']
            ki = self.ctrlConfig[jointController]['pid']['i']
            kd = self.ctrlConfig[jointController]['pid']['d']

            ### Implement your code from here ... ###
            # TODO: obtain torque from PD controller
            torque = 0.0  # TODO: fix me
            ### ... to here ###

            self.p.setJointMotorControl2(
                bodyIndex=self.robot,
                jointIndex=self.jointIds[joint],
                controlMode=self.p.TORQUE_CONTROL,
                force=torque
            )

            # Gravity compensation
            # A naive gravitiy compensation is provided for you
            # If you have embeded a better compensation, feel free to modify
            compensation = self.jointGravCompensation[joint]
            self.p.applyExternalForce(
                objectUniqueId=self.robot,
                linkIndex=self.jointIds[joint],
                forceObj=[0, 0, -compensation],
                posObj=self.getLinkCoM(joint),
                flags=self.p.WORLD_FRAME
            )
            # Gravity compensation ends here

        self.p.stepSimulation()
        self.drawDebugLines()
        time.sleep(self.dt)

    ########## Task 3: Robot Manipulation ##########
    def cubic_interpolation(self, points, nTimes=100):
        """
        Given a set of control points, return the
        cubic spline defined by the control points,
        sampled nTimes along the curve.
        """
        # TODO add your code here
        # Return 'nTimes' points per dimension in 'points' (typically a 2xN array),
        # sampled from a cubic spline defined by 'points' and a boundary condition.
        # You may use methods found in scipy.interpolate

        # return xpoints, ypoints
        pass

    # Task 3.1 Pushing
    def dockingToPosition(self, leftTargetAngle, rightTargetAngle, angularSpeed=0.005,
                          threshold=1e-1, maxIter=300, verbose=False):
        """A template function for you, you are free to use anything else"""
        # TODO: Append your code here
        pass

    # Task 3.2 Grasping & Docking
    def clamp(self, leftTargetAngle, rightTargetAngle, angularSpeed=0.005, threshold=1e-1, maxIter=300, verbose=False):
        """A template function for you, you are free to use anything else"""
        # TODO: Append your code here
        pass

### END