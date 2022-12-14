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
        'CHEST_JOINT0': ['CHEST_JOINT0'],
        'HEAD_JOINT0': ['HEAD_JOINT0'],
        'HEAD_JOINT1': ['CHEST_JOINT0', 'HEAD_JOINT0', 'HEAD_JOINT1'],
        'LARM_JOINT0': ['CHEST_JOINT0', 'LARM_JOINT0'],
        'LARM_JOINT1': ['LARM_JOINT0', 'LARM_JOINT1'],
        'LARM_JOINT2': ['LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2'],
        'LARM_JOINT3': ['LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2', 'LARM_JOINT3'],
        'LARM_JOINT4': ['LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2', 'LARM_JOINT3', 'LARM_JOINT4'],
        'LARM_JOINT5': ['LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2', 'LARM_JOINT3', 'LARM_JOINT4', 'LARM_JOINT5'],
        'RARM_JOINT0': ['RARM_JOINT0'],
        'RARM_JOINT1': ['RARM_JOINT0', 'RARM_JOINT1'],
        'RARM_JOINT2': ['RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2'],
        'RARM_JOINT3': ['RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2', 'RARM_JOINT3'],
        'RARM_JOINT4': ['RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2', 'RARM_JOINT3', 'RARM_JOINT4'],
        'RARM_JOINT5': ['RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2', 'RARM_JOINT3', 'RARM_JOINT4', 'RARM_JOINT5']
    }

    start = time.time()
    oldPositions = {}
    errorIntegral = {}

    def myGetJointVel(self, jointName, x_real):
        dx_real = (x_real[jointName] - self.oldPositions[jointName]) / self.dt
        # dx_real = self.getJointVel(joint)
        return dx_real







    def getJointRotationalMatrix(self, jointName=None, theta=None):
        """
            Returns the 3x3 rotation matrix for a joint from the axis-angle representation,
            where the axis is given by the revolution axis of the joint and the angle is theta.
        """
        if jointName == None:
            raise Exception("[getJointRotationalMatrix] \
                Must provide a joint in order to compute the rotational matrix!")

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

        #note that there are only 3 possible rotation axes for a joint (except base_to_waist which has [0 0 0] since it is fixed)

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
        transformMats = self.getTransformationMatrices() # our transformation matrices
        result = transformMats.get(jointName) # the transformation matrix for the joint we want
        for next_joint in self.jointOrderFK.get(jointName):
            result = transformMats.get(next_joint) @ result # multiply the transformation matrices along the kinematic chain
        return (result[0:3, 3]).squeeze(), result[0:3, 0:3]

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
        jacobian = np.ndarray((0,6))
        for joint in self.jointOrderIK[endEffector]: # for each joint in the chain
            # position
            temp = np.cross(self.getJointOrientation(joint, ref=self.jointRotationAxis[joint]),
                            self.getJointPosition(endEffector) - self.getJointPosition(joint))
            # orientation
            temp = np.hstack([temp,
                             [np.cross(self.getJointOrientation(joint, ref=self.jointRotationAxis[joint]),
                                       self.getJointOrientation(endEffector, ref=self.jointRotationAxis[endEffector]))]])
                        # note that we applied the joint transformation matrix to the joint axis
            jacobian = np.vstack([jacobian, temp])
            #note that we transpose the jacobian matrix to add rows instead of columns
        return jacobian.T



    # Task 1.2 Inverse Kinematics

    def inverseKinematics(self, endEffector, targetPosition, orientation, direction=None):
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
        if orientation is not None:
            dori = (orientation - self.getJointOrientation(endEffector, ref=self.jointRotationAxis[endEffector])).reshape((1, 3))
            dy = np.hstack([dy, dori])
        # Calculate delta
        if orientation is None:
            drad = np.linalg.pinv(jacobian[:3, :]) @ dy.T
        else:
            drad = np.linalg.pinv(jacobian) @ dy.T

        curr_q = {}
        for j, joint in enumerate(self.jointOrderIK[endEffector]):
            curr_q[joint] = self.getJointPos(joint) + drad[j,0]

        if direction is not None:
            #we project the direction onto the plane perpendicular to the orientation
            #and then we calculate the angle between the projected direction and the current orientation
            #we then add this angle to the current joint angle
            #this is a hacky way to make the robot move in a direction
            #but it works
            # choose a reference vector that is not parallel to the rotation axis of the end effector
            normal = self.getJointOrientation(endEffector, ref=self.jointRotationAxis[endEffector])

            ref = np.array([0, 0, 1])
            if np.dot(ref, self.jointRotationAxis[endEffector]) > 0.9:
                ref = np.array([1, 0, 0])
            eeref = self.getJointOrientation(endEffector, ref=ref)


            proj = direction - np.dot(direction, normal) * normal
            proj = proj / np.linalg.norm(proj)

            angle = np.arctan2(np.dot(np.cross(eeref, proj), normal), np.dot(eeref, proj))
            # angle = np.arccos(np.dot(proj, self.getJointOrientation(endEffector, ref=ref)))
            curr_q[endEffector] += angle

            # print("drad endEffector: ", drad[len(self.jointOrderIK[endEffector])-1])
        return curr_q

    def move_without_PD(self, endEffector, targetPosition, speed=0.01, orientation=None, direction=None,
                        threshold=1e-3, maxIter=3000, debug=False, verbose=False):
        """
        Move joints using Inverse Kinematics solver (without using PD control).
        This method should update joint states directly.
        Return:
            pltTime, pltDistance arrays used for plotting
        """

        # Initialize variables
        interSteps = 100
        pltDistance = []
        pltTime = np.linspace(0, 1, interSteps)
        targets = np.linspace(self.getJointPosition(endEffector), targetPosition, interSteps)
        if orientation is not None:
            orientations = np.linspace(self.getJointOrientation(endEffector, ref=self.jointRotationAxis[endEffector]), orientation, interSteps)
        else:
            orientations = [None] * interSteps
        if direction is not None:
            # we need to make sure we don't pick a plane that is parallel to the orientation
            ref = np.array([0, 0, 1])
            if np.dot(ref, self.jointRotationAxis[endEffector]) > 0.9:
                ref = np.array([1, 0, 0])
            directions = np.linspace(self.getJointOrientation(endEffector, ref=ref), direction, interSteps)
        else:
            directions = [None] * interSteps

        # for each interpolation step
        for (target, ori, direc) in zip(targets, orientations, directions):
            x_refs = self.inverseKinematics(endEffector, target, ori, direction=direc) # perform IK
            for joint in x_refs:
                self.jointTargetPos[joint] = x_refs[joint] # update joint target positions
            pltDistance.append(np.linalg.norm(self.getJointPosition(endEffector) - targetPosition))
            self.tick_without_PD()
        return pltTime, pltDistance
        pass

    def tick_without_PD(self):
        """Ticks one step of simulation without PD control. """
        # Iterate through all joints and update joint states.
        # For each joint, you can use the shared variable self.jointTargetPos.
        for joint, targetPos in self.jointTargetPos.items():
            self.p.resetJointState(self.robot, self.jointIds[joint], targetPos) #set the joint to the target position

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
        error = x_ref - x_real # position error
        derror = dx_ref - dx_real # velocity error
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
            ki = self.ctrlConfig[jointController]['pid']['i'] # always 0 for PD control
            kd = self.ctrlConfig[jointController]['pid']['d']


            torque = self.calculateTorque(x_ref, x_real, dx_ref, dx_real, integral, kp, ki, kd) # we calculate the torque

            pltTorque.append(torque)

            # send the manipulation signal to the joint
            self.p.setJointMotorControl2(
                bodyIndex=self.robot,
                jointIndex=self.jointIds[joint],
                controlMode=self.p.TORQUE_CONTROL,
                force=torque
            )
            compensation = self.jointGravCompensation[joint]
            self.p.applyExternalForce(
                objectUniqueId=self.robot,
                linkIndex=self.jointIds[joint],
                forceObj=[0, 0, -compensation],
                posObj=self.getLinkCoM(joint),
                flags=self.p.WORLD_FRAME
            )
            # calculate the physics and update the world
            self.p.stepSimulation()
            time.sleep(self.dt)

        targetPosition, targetVelocity = float(targetPosition), float(targetVelocity)


        # disable joint velocity controller before apply a torque
        self.disableVelocityController(joint)
        # logging for the graph
        pltTime, pltTarget, pltTorque, pltTorqueTime, pltPosition, pltVelocity = [], [], [], [], [], []

        errorIntegral = 0
        oldPosition = float(self.getJointPos(joint))
        dx_real = 0
        simulationTime = 0
        while abs(targetPosition - oldPosition) > 0.01 or abs(targetVelocity - dx_real) > 0.01:
            # get current joint state
            x_real = self.getJointPos(joint)
            dx_real = (x_real - oldPosition) / self.dt

            # calculate the error integral
            errorIntegral += (targetPosition - x_real) * self.dt

            # call the tick function
            toy_tick(targetPosition, x_real, targetVelocity, dx_real, errorIntegral)

            # update the old position
            oldPosition = x_real

            # logging for the graph
            simulationTime += self.dt
            pltTime.append(simulationTime)
            pltTarget.append(targetPosition)
            pltPosition.append(x_real)
            pltVelocity.append(dx_real)

        pltTorqueTime = pltTime
        return pltTime, pltTarget, pltTorque, pltTorqueTime, pltPosition, pltVelocity

    def move_with_PD(self, endEffector, targetPosition, speed=0.01, orientation=None, direction=None,
                     threshold=5e-3, maxIter=3000, debug=False, verbose=False, velocityControl=False):
        """
        Move joints using inverse kinematics solver and using PD control.
        This method should update joint states using the torque output from the PD controller.
        Return:
            pltTime, pltDistance arrays used for plotting
        """

        # initialization
        interSteps = 50
        pltDistance = []
        iterCounter = 0
        targetPositions = np.linspace(self.getJointPosition(endEffector), targetPosition, 50)
        if orientation is not None:
            orientations = np.linspace(self.getJointOrientation(endEffector, ref=self.jointRotationAxis[endEffector]), orientation, 50)
        else:
            orientations = [None] * interSteps
        if direction is not None:
            # we need to make sure we don't pick a plane that is parallel to the orientation
            ref = np.array([0, 0, 1])
            if np.dot(ref, self.jointRotationAxis[endEffector]) > 0.9:
                ref = np.array([1, 0, 0])
            directions = np.linspace(self.getJointOrientation(endEffector, ref=ref), direction, interSteps)
        else:
            directions = [None] * interSteps

        # loop over the target positions, orientations and directions
        for (target, ori, direc) in zip(targetPositions, orientations, directions):
            x_refs = self.inverseKinematics(endEffector, target, ori, direction=direc) #perform IK
            self.jointTargetPos = {}
            for joint in x_refs:
                self.jointTargetPos[joint] = x_refs[joint] # set the target position for each joint
            while iterCounter < maxIter:
                pltDistance.append(np.linalg.norm(self.getJointPosition(endEffector) - targetPosition))
                self.tick()
                if np.linalg.norm(self.getJointPosition(endEffector) - target) < 2*threshold: # if we are sufficiently close to the target
                    break
                iterCounter += 1

        print("precisely matching target")
        # we perform more ticks to get closer to the target using velocity control and a tight threshold
        x_refs = self.inverseKinematics(endEffector, targetPosition, orientation) # perform IK
        self.jointTargetPos = {}
        for joint in x_refs:
            self.jointTargetPos[joint] = x_refs[joint]  # set the target position for each joint
        while iterCounter < maxIter:
            pltDistance.append(np.linalg.norm(self.getJointPosition(endEffector) - targetPosition))
            self.tick()
            x_real = self.getJointPoses(self.joints) # get current poses (necessary for myGetJointVel function)
            if np.linalg.norm(self.getJointPosition(endEffector) - targetPosition) < threshold and (not velocityControl or abs(
                    self.myGetJointVel(endEffector, x_real)) < 2.6e-5):
                print("reached") # we indicate that we reached the target
                break
            iterCounter += 1


        pltTime = np.linspace(0, len(pltDistance)*self.dt, len(pltDistance))
        return pltTime, pltDistance
        pass


    def move_2_EE_with_PD(self, jointNames, targetPositions, targetOrientations, maxIter=3000):
        # Moves two end effectors to the target positions and orientations using PD control

        # Initialization
        interSteps = 50
        iterCounter = 0
        (joint1, joint2) = jointNames
        (tP1, tP2) = targetPositions
        (tO1, tO2) = targetOrientations
        # initialize the linspaces
        lTP1 = np.linspace(self.getJointPosition(joint1), tP1, interSteps)
        lTP2 = np.linspace(self.getJointPosition(joint2), tP2, interSteps)
        lTO1 = np.linspace(self.getJointOrientation(joint1, ref=self.jointRotationAxis[joint1]), tO1, interSteps)
        lTO2 = np.linspace(self.getJointOrientation(joint2, ref=self.jointRotationAxis[joint2]), tO2, interSteps)

        # we iterate through the linspaces and move the end effectors
        i1, i2 = 0, 0
        xref1 = self.inverseKinematics(joint1, lTP1[0], lTO1[0]) # perform IK
        xref2 = self.inverseKinematics(joint2, lTP2[0], lTO2[0]) # perform IK
        for joint in xref1:
            self.jointTargetPos[joint] = xref1[joint] # set the target positions
        for joint in xref2:
            self.jointTargetPos[joint] = xref2[joint] # set the target positions
        # for every joint, move to the target position
        while i1 < interSteps or i2 < interSteps:
            if i1 < interSteps and np.linalg.norm(self.getJointPosition(joint1) - lTP1[i1]) < 0.01: # if we are close enough to the target position, move to the next one
                i1 += 1
                if i1 < interSteps: # if we are not at the end of the linspaces
                    xref1 = self.inverseKinematics(joint1, lTP1[i1], lTO1[i1]) # perform IK
                    for joint in xref1:
                        self.jointTargetPos[joint] = xref1[joint] # set the target positions
            if i2 < interSteps and np.linalg.norm(self.getJointPosition(joint2) - lTP2[i2]) < 0.01: # if we are close enough to the target position, move to the next one
                i2 += 1
                if i2 < interSteps: # if we are not at the end of the linspaces
                    xref2 = self.inverseKinematics(joint2, lTP2[i2], lTO2[i2]) # perform IK
                    for joint in xref2:
                        self.jointTargetPos[joint] = xref2[joint] # set the target positions


            self.tick()
            iterCounter += 1
            if iterCounter > maxIter:
                break


    def tick(self):
        """Ticks one step of simulation using PD control."""
        # Iterate through all joints and update joint states using PD control.
        x_real = self.getJointPoses(self.joints)
        if (not self.oldPositions):  # check if oldPositions is empty (used for velocity control)
            self.oldPositions = x_real

        for joint in self.jointRotationAxis.keys(): # iterate through all joints

            # skip dummy joints and base_to_waist
            if joint == 'base_to_dummy':
                continue
            if joint == 'base_to_waist':
                continue
            jointController = self.jointControllers[joint]
            if jointController == 'SKIP_THIS_JOINT':
                continue
            # disable joint velocity controller before apply a torque
            self.disableVelocityController(joint)

            # loads your PID gains
            kp = self.ctrlConfig[jointController]['pid']['p']
            ki = self.ctrlConfig[jointController]['pid']['i']
            kd = self.ctrlConfig[jointController]['pid']['d']

            # Here, we calculate the torque using PID control.
            x_ref = self.jointTargetPos.get(joint, self.getJointPos(joint))
            dx_ref = self.jointTargetVels.get(joint, 0)
            dx_real = self.myGetJointVel(joint, x_real)
            self.errorIntegral[joint] = self.errorIntegral.get(joint, 0) + (x_ref - x_real[joint]) * self.dt
            torque = self.calculateTorque(x_ref, x_real[joint], dx_ref, dx_real, self.errorIntegral[joint], kp, ki, kd)

            self.p.setJointMotorControl2(
                bodyIndex=self.robot,
                jointIndex=self.jointIds[joint],
                controlMode=self.p.TORQUE_CONTROL,
                force=torque
            )

            # Gravity compensation
            # A naive gravitiy compensation is provided for you
            # If you have embeded a better compensation, feel free to modify
            # We didn't alter the gravity compensation
            compensation = self.jointGravCompensation[joint]
            self.p.applyExternalForce(
                objectUniqueId=self.robot,
                linkIndex=self.jointIds[joint],
                forceObj=[0, 0, -compensation],
                posObj=self.getLinkCoM(joint),
                flags=self.p.WORLD_FRAME
            )
            # Gravity compensation ends here
        self.oldPositions = x_real # update oldPositions
        self.p.stepSimulation()
        self.drawDebugLines()
        time.sleep(self.dt)

### END