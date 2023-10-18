from geometry_msgs.msg import Pose, PoseArray, Quaternion
from . pf_base import PFLocaliserBase
import math
import rospy

import numpy as np

from . util import rotateQuaternion, getHeading
from random import Random, gauss, randint, uniform, vonmisesvariate
from statistics import mean, stdev


from time import time


class PFLocaliser(PFLocaliserBase):


    def __init__(self):

        # ----- Assigns an initial number of cloud points
        self.CLOUD_POINTS = 400
        self.TOP_WEIGHT_PERCENTAGE = 0.75
        
        super(PFLocaliser, self).__init__()
        
        # ----- Set motion model parameters
        self.ODOM_ROTATION_NOISE = 0.015 # Odometry model rotation noise
        self.ODOM_TRANSLATION_NOISE = 0.030 # Odometry model x axis (forward) noise
        self.ODOM_DRIFT_NOISE = 0.022 # Odometry model y axis (side-to-side) noise
 
        # ----- Sensor model parameters
        self.NUMBER_PREDICTED_READINGS = 50 # Number of readings to predict

        # ----- Maybe add something for clusters or kidnapped robot.
        self.MAP_RESOLUTION = self.occupancy_map.info.resolution
        self.WIDTH = self.occupancy_map.info.width
        self.HEIGHT = self.occupancy_map.info.height

    def distance(self, pose1, pose2):
        # Define a distance function between two poses
        dx = pose1.position.x - pose2.position.x
        dy = pose1.position.y - pose2.position.y
        return np.sqrt(dx * dx + dy * dy)

    def euclidean_function(self, particle):
        #euclideanDistance = math.sqrt(math.pow(particle.position.x - self.estimatedpose.pose.pose.position.x, 2) + math.pow(particle.position.y - self.estimatedpose.pose.pose.position.y,2))
        euclideanDistance = math.sqrt(math.pow(particle.position.x, 2) + math.pow(particle.position.y,2))
        return euclideanDistance

    def dbscan(self, Poses, eps, min_samples):
    # Perform DBSCAN clustering on a list of Pose messages
        labels = [-1] * len(Poses)
        cluster_id = 0

        for i in range(len(Poses)):
            if labels[i] != -1:
                continue

            neighbors = []

            for j in range(len(Poses)):
                if i == j:
                    continue

                if self.distance(Poses[i], Poses[j]) < eps:
                    neighbors.append(j)

            if len(neighbors) < min_samples:
                labels[i] = -1  # Mark as noise
            else:
                cluster_id += 1
            labels[i] = cluster_id

            for j in neighbors:
                if labels[j] == -1:
                    labels[j] = cluster_id
                    sub_neighbors = []

                    for k in range(len(Poses)):
                        if k in neighbors:# avoid repeat in subneighbors
                            continue

                        if self.distance(Poses[j], Poses[k]) < eps:
                            sub_neighbors.append(k)

                    if len(sub_neighbors) >= min_samples:
                        neighbors.extend(sub_neighbors)
        return labels
    
    def estimate_aft_dbscan(self, Poses, labels):
        cluster_counts = {}
        for label in labels:
            if label not in cluster_counts:
                cluster_counts[label] = 1
            else:
                cluster_counts[label] += 1

        # Find the most dense cluster
        most_dense_cluster_id = max(cluster_counts, key=cluster_counts.get)
        label_num = cluster_counts[most_dense_cluster_id] # in terms of varing particle number overtime
        self.CLOUD_POINTS = int(1.5*label_num)
        cluster_indices = [i for i, label in enumerate(labels) if label == most_dense_cluster_id]
        cluster_poses = [Poses[i] for i in cluster_indices]

        # Calculate the estimated pose based on the mean position of the most dense cluster
        estimated_pose = Pose()
        if cluster_poses:
            estimated_pose.position.x = np.mean([pose.position.x for pose in cluster_poses])
            estimated_pose.position.y = np.mean([pose.position.y for pose in cluster_poses])
            estimated_pose.orientation.w = np.mean([pose.orientation.w for pose in cluster_poses])
            estimated_pose.orientation.z = np.mean([pose.orientation.z for pose in cluster_poses])
        return estimated_pose

    def particle_clustering(self, particlePoses):
        
        #Using distance with the initial point to cluster
        
        euclideanDistances = [] #Initializes a new array to store euclidean distances.

        for pose in particlePoses: #Takes each individual particle in the particle poses array
            euclideanDistances.append(self.euclidean_function(pose)) #sends each pose to the euclidean function to determine each particle's euclidean distance
                                                                     #and adds to array
        
        meanDistance = mean(euclideanDistances) #Calculates the mean of the distances to use as the center of gaussian
        standardDeviation = stdev(euclideanDistances) #Determines the standard deviation from the mean
        upper = meanDistance + standardDeviation
        lower = meanDistance - standardDeviation
        if standardDeviation > 3: #If standard deviation its greater than this, don't consider the particle
            remainingParticlesPoses = [pose for pose in particlePoses if (lower <self.euclidean_function(pose) < upper)] #creates an array for the new particles
            return self.particle_clustering(remainingParticlesPoses) #recursive array with new particles, until it returns the robotEstimatedPose std < 3

        else:
            estimated_pose = Pose() #Creates Robot estimated poses

            estimated_pose.position.x = np.mean([pose.position.x for pose in particlePoses])
            estimated_pose.position.y = np.mean([pose.position.y for pose in particlePoses])
            estimated_pose.orientation.w = np.mean([pose.orientation.w for pose in particlePoses])
            estimated_pose.orientation.z = np.mean([pose.orientation.z for pose in particlePoses])
            return estimated_pose


    def initialise_particle_cloud(self, initialpose):
        """
        Set particle cloud to initialpose plus noise. Called whenever an initialpose message is received (to change the
        starting location of the robot), or a new occupancy_map is received.
        self.particlecloud can be initialised here. Initial pose of the robot is also set here.     
        :Args:
            | initialpose: the initial pose estimate
        :Return:
            | (geometry_msgs.msg.PoseArray) poses of the particles
        """
        
        """ Things to initialize """
        arrayPoses = PoseArray() # The Array we are going to return for the cloud
        width = self.occupancy_map.info.width # The width of the map, so particles only span inside of the width
        height = self.occupancy_map.info.height # The height of the map so particle only span inside the heigh of the map
        resolution = self.occupancy_map.info.resolution #gives the resolution of the map
        cloudPoints = self.CLOUD_POINTS  # Is the number of cloud points we are calculating
        appendedParticles = 0 # To check that the @NUMBER cloudpoints have been added to the array

        while appendedParticles < cloudPoints:
            myPose = Pose() #creates a pose variable, once per cycle to not cause problem when appending
            random_angle = vonmisesvariate(0,0) # generates a random angle between 0 to 2pi
            #random_x = randint(0,width-1)# generates a random position around center of map
            random_x = int(gauss(math.floor(initialpose.pose.pose.position.x/resolution), width/8))
            #random_y = randint(0,height-1) # generates a random position around center of map
            random_y = int(gauss(math.floor(initialpose.pose.pose.position.y/resolution), height/8))
            myPose.position.x = random_x * resolution #Multiplies by the resolution of the map so the correct x position is obtained
            myPose.position.y = random_y * resolution #Multiplies by the resolution of the map so the correct y position is obtained
            myPose.orientation = rotateQuaternion(Quaternion(w=1.0),random_angle) #rotates from default quaternion into new angle

            if random_x < width - 1 and random_y < height - 1:
                if self.occupancy_map.data[random_x + random_y * width] == 0: # Verifies that the particle is created in a white space
                    arrayPoses.poses.append(myPose) #Adds the particle to an array.
                    appendedParticles += 1 #Ready to append the next particle

        # print(appendedParticles)
        return arrayPoses #Returns the array so they are added to the particle cloud

     
    def update_particle_cloud(self, scan):
        """
        This should use the supplied laser scan to update the current
        particle cloud. i.e. self.particlecloud should be updated.
        
        :Args:
            | scan (sensor_msgs.msg.LaserScan): laser scan to use for update

         """

        "Initialize Variables"
        cloudPoints = self.CLOUD_POINTS  # Is the # of cloud points we have
        weights = [] # Array for storing the weights of each particle
        

        "Scan the weights of each particle"
        for eachParticle in self.particlecloud.poses:
            # myPose = Pose() # A pose for each of the particles in the particle cloud
            # myPose = eachParticle #assigns that particle to the pose
            weights.append((eachParticle, self.sensor_model.get_weight(scan, eachParticle))) # Creates a tuple with the particle position and weights
        
        sortedWeights = sorted(weights, key=lambda higherWeights: higherWeights[1], reverse=True) # Puts higher weight particles at top of array to guarantee copies
        heaviestParticles = sortedWeights[0:int(self.CLOUD_POINTS * self.TOP_WEIGHT_PERCENTAGE)] #Takes the % heaviest particles in a new array

        weightSum = sum(higherWeights[1] for higherWeights in heaviestParticles) #Does the sum of weights for top particles


        " Particles to add via new random position generation "
        remainingWeightPoses = PoseArray() #Make up for those have been droped
        width = self.occupancy_map.info.width 
        height = self.occupancy_map.info.height 
        resolution = self.occupancy_map.info.resolution
        remainingCloudPoints = int(cloudPoints * (1-self.TOP_WEIGHT_PERCENTAGE)) # Is the number of cloud points we are now randomly determining
        appendedParticles = 0 # To check that the remaining cloudpoints have been added

        while appendedParticles < remainingCloudPoints:
            NewPose = Pose() #creates a pose variable, once per cycle to not cause problem when appending
            random_angle = vonmisesvariate(0,0) # generates a random angle between 0 to 2pi
            random_x = randint(0,width-1)# generates a random position around center of map 
            random_y = randint(0,height-1) # generates a random position around center of map
            NewPose.position.x = random_x * resolution #Multiplies by the resolution of the map so the correct x position is obtained
            NewPose.position.y = random_y * resolution #Multiplies by the resolution of the map so the correct y position is obtained
            NewPose.orientation = rotateQuaternion(Quaternion(w=1.0),random_angle) #rotates from default quaternion into new angle

            if self.occupancy_map.data[random_x + random_y * width] == 0: # Verifies that the particle is created in a white space
                remainingWeightPoses.poses.append(NewPose) #Adds the particle to an array.
                appendedParticles += 1 #Ready to append the next particle
            



        #Resampling from topParticles
        # ------ Cumulative Distribution initialization make a list from lower value to high value
        cumulative_weights = np.cumsum([weight/weightSum for (particle, weight) in heaviestParticles])# normlize the weight
        threshold = uniform(0, 1)
        #threshold = uniform(0,math.pow(len(heaviestParticles),-1)) #Creates uniform distribution for the threshold to update particles
        cycleNum = 0 # variable for while
        ResampledPoses = PoseArray() #creates an array for resampled pose
        for _ in range(len(heaviestParticles)):
            threshold = math.pow(len(heaviestParticles),-1)
            for i in range(len(heaviestParticles)):
                if threshold < cumulative_weights[i]:
                    CurrentPose = Pose() #stores the new pose
                    CurrentPose.position.x = heaviestParticles[i][0].position.x + gauss(0,self.ODOM_TRANSLATION_NOISE ) #stores position x
                    CurrentPose.position.y = heaviestParticles[i][0].position.y + gauss(0,self.ODOM_DRIFT_NOISE ) #stores position y
                    CurrentPose.orientation = rotateQuaternion(Quaternion(w=1.0), getHeading(heaviestParticles[i][0].orientation) + 
                            gauss(0, self.ODOM_ROTATION_NOISE)) #stores orientation
                    ResampledPoses.poses.append(CurrentPose) #appends the pose
                    threshold += math.pow(len(heaviestParticles),-1)
                    break


        LatestPosesArray = ResampledPoses #stores the array in the modified array
        LatestPosesArray.poses = LatestPosesArray.poses + remainingWeightPoses.poses #combines both array poses

        self.particlecloud = LatestPosesArray #updates particle cloud


    def estimate_pose(self):
        """
        This should calculate and return an updated robot pose estimate based
        on the particle cloud (self.particlecloud).
        
        Create new estimated pose, given particle cloud
        E.g. just average the location and orientation values of each of
        the particles and return this.
        
        Better approximations could be made by doing some simple clustering,
        e.g. taking the average location of half the particles after 
        throwing away any which are outliers

        :Return:
            | (geometry_msgs.msg.Pose) robot's estimated pose.
         """
        # labels = self.dbscan(self.particlecloud.poses, 5, 15) #set the parameter
        # robotEstimatedPose = self.estimate_aft_dbscan(self.particlecloud.poses, labels) 

        robotEstimatedPose  = self.particle_clustering(self.particlecloud.poses) # the second way to cluster

        return robotEstimatedPose #returns the new estimated pose
