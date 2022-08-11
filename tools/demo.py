from mmdet.apis import init_detector, inference_detector, show_result_pyplot
# import mmcv
from mmdet3d.models import build_model


def main():
    config_file = 'projects/configs/bevformer/bevformer_tiny.py'
    checkpoint_file = 'work_dirs/bevformer_tiny/epoch_33.pth'

# build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
    img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
    result = inference_detector(model, img)
# visualize the results in a new window
    show_result_pyplot(img, result, model.CLASSES)
# or save the visualization results to image files
    show_result_pyplot(img, result, model.CLASSES, out_file='result.jpg')




if __name__ == '__main__':
    main()
