from SuperPointPretrainedNetwork.demo_superpoint import * 

def superpoint_generator():
        # This should be outside of trajectory esetimator and should be inside Superpoint 
        """Use superpoint to extract features in the image
        Return:
            superpoint_feature - N*2 numpy array (u,v)
        """

        # Refer to /SuperPointPretrainedNetwork/demo_superpoint for more information about the parameters
        fe = SuperPointFrontend(weights_path='SuperPointPretrainedNetwork/superpoint_v1.pth',
                          nms_dist=4,
                          conf_thresh=0.015,
                          nn_thresh=0.7,
                          cuda=False)
        # superpoints, descriptors, _ = fe.run(image)
        # return superpoints, descriptors
        return 0

if __name__ == "__main__":
    test = superpoint_generator()
