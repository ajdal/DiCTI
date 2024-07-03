python run_densepose.py \
       dump \
       configs/densepose_rcnn_R_101_FPN_s1x.yaml \
       weights/densepose_rcnn_R_101_FPN_s1x.pkl \
       sample_data/images \
       --output output/densepose/sample_data.pkl \
       -v

# python run_densepose.py show \
#       configs/densepose_rcnn_R_101_FPN_s1x.yaml \
#       weights/densepose_rcnn_R_101_FPN_s1x.pkl \
#       sample_data/images dp_segm \
#       --output output/sample_data.png