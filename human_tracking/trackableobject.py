class TrackableObject:
    def __init__(self, objectID, centroid):
        # store the object ID, then initialize a list of centroids
        # using the current centroid
        self.objectID = objectID                   #给每个人label id
        self.centroids = [centroid]                #record每个人的centroid， 包括他们的history centroid
        self.counted = False                 # initialize a boolean used to indicate if the object has already been counted or not
