
import cv2
import sys
sys.path.append(".")
from libLocalDesc import *
from acc_test_library import *
import argparse

from matplotlib import pyplot as plt
plt.switch_backend('agg')

CosProxThres = 0.4
SignAlingThres = 4000

def main():
    ch1 = ['cos', 'sign']
    ch2 = ['aid', 'root']

    parser = argparse.ArgumentParser('Small script to'
                                     ' run sift-aid')

    parser.add_argument('--im0', '-i0', required=True)
    parser.add_argument('--im1', '-i1', required=True)
    parser.add_argument('--type', '-t', choices=ch2, 
                        required=False, default='aid'
                       )
    parser.add_argument('--thresh', '-th', choices=ch1, 
                        required=False, default='sign'
                       )
    args = parser.parse_args()

    try:
        img1 = cv2.cvtColor(cv2.imread(args.im0),
                        cv2.COLOR_BGR2GRAY) # queryImage

        img2 = cv2.cvtColor(cv2.imread(args.im1),
                        cv2.COLOR_BGR2GRAY) # trainImage
    except Exception as e:
        print(e)
    else:

        if args.type == 'root':
            total, good_HC, ET_KP, ET_M = RootSIFT(img1,
                                               img2, 
                                               MatchingThres = 0, 
                                               knn_num = 2, 
                                               Rooted = True, 
                                               GFilter=1,
                                               Visual=True)

        elif args.type == 'aid':
            total, good_HC, ET_KP, ET_M = siftAID(
                                              img1,
                                              img2,
                                              MatchingThres = SignAlingThres if args.thresh == 'sign' else CosProxThres,
                                              Simi = 'SignProx' if args.thresh == 'sign' else 'CosProx',
                                              Visual=True
                                             )
        else:
            print('hic sunt dracones!')


        print('estimated time for calculating key points ', ET_KP)
        print('estimated time for matching  ', ET_M)
if __name__ == "__main__":
    main()
