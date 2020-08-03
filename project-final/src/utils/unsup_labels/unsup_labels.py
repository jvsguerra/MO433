import os
import zipfile
import cv2
from pyimgsaliency import get_saliency_rbd, get_saliency_ft, get_saliency_mbd
from SaliencyRC import get_saliency_hc, get_saliency_rc

if __name__ == '__main__':

    # Unsupervised saliency methods adapted from: https://github.com/yhenon/pyimgsaliency
    #
    # The following algorithms are implemented for calculating saliency maps in the above repository:
    # 
    # mbd: Jianming Zhang, Stan Sclaroff, Zhe Lin, Xiaohui Shen, Brian Price and Radomir Mech. "Minimum Barrier Salient Object Detection at 80 FPS." 
    # rbd: Saliency Optimization from Robust Background Detection, Wangjiang Zhu, Shuang Liang, Yichen Wei and Jian Sun, IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014
    # ft: R. Achanta, S. Hemami, F. Estrada and S. Susstrunk, Frequency-tuned Salient Region Detection, IEEE International Conference on Computer Vision and Pattern Recognition (CVPR 2009), pp. 1597 - 1604, 2009
    #
    # Unsupervised saliency method adapted from: https://github.com/congve1/SaliencyRC
    #
    # The following algorithms are implemented for calculating saliency maps in the above repository:
    #
    # hc (Histogram based contrast) and rc (Region based constrast) : Ming-Ming Cheng, Niloy J. Mitra, Xiaolei Huang, Philip H. S. Torr, and Shi-Min Hu, IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, pp. 409 - 416, 2011
 
    shouldGetUnsupLabel = False 
    if shouldGetUnsupLabel:
        try:
            os.mkdir('./input/unsup_labels/')
            os.mkdir('./input/unsup_labels/rdb')
            os.mkdir('./input/unsup_labels/ft')
            os.mkdir('./input/unsup_labels/mbd')
            os.mkdir('./input/unsup_labels/hc')
            os.mkdir('./input/unsup_labels/rc')
        except:
            pass

        for filename in os.listdir('./input/dataset/MSRA-B'):
            if filename.endswith('.jpg'):
                # Get RBD saliency map
                try:
                    # Unknown error handler
                    rbd = get_saliency_rbd('./input/dataset/MSRA-B/' + filename).astype('uint8')
                    cv2.imwrite('./input/unsup_labels/rdb/' + filename[:-4] + '_ngt.png', rbd)
                except:
                    with open('output/failed.txt', 'a+') as f:
                        f.write(filename + '\n')

                # Get FT saliency map
                ft = get_saliency_ft('./input/dataset/MSRA-B/' + filename).astype('uint8')
                cv2.imwrite('./input/unsup_labels/ft/' + filename[:-4] + '_ngt.png', ft)

                # Get MBD saliency map
                mbd = get_saliency_mbd('./input/dataset/MSRA-B/' + filename).astype('uint8')
                cv2.imwrite('./input/unsup_labels/mbd/' + filename[:-4] + '_ngt.png', mbd)
                
                # Get HC saliency map
                hc = get_saliency_hc('./input/dataset/MSRA-B/' + filename).astype('uint8')
                cv2.imwrite('./input/unsup_labels/hc/' + filename[:-4] + '_ngt.png', hc)

                # Get RC saliency map
                rc = get_saliency_rc('./input/dataset/MSRA-B/' + filename).astype('uint8')
                cv2.imwrite('./input/unsup_labels/rc/' + filename[:-4] + '_ngt.png', rc)
    else:
        path = "input/"
        dataset_name = "unsup_labels"
        if not os.path.isdir(path + dataset_name):
            print("[==> Extracting unsupervised labels ...")
            with zipfile.ZipFile(path + dataset_name + ".zip", 'r') as zip_ref:
                zip_ref.extractall(path)
            print("Unsupervised labels extracted!")