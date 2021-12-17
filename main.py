#from  util.model_processor import ModelProcessor
from acl_resource import AclResource
from acl_model import Model
import numpy as np
import argparse
MODEL_PATH = './model/model_pos_v3.om'


def execute(model, output_dir, vid_name):
    # prepare dataset.
    from common.custom_dataset import CustomDataset
    dataset = CustomDataset('./data/data_2d_custom_myvideos.npz')

    # preprocess the input data
    keypoints = np.load('./data/data_2d_custom_myvideos.npz', allow_pickle=True)
    keypoints_metadata = keypoints['metadata'].item()
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
    keypoints = keypoints['positions_2d'].item()

    # normalize data points
    from common.camera import normalize_screen_coordinates
    for subject in dataset.subjects():
        assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
        for action in dataset[subject].keys():
            assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
            if 'positions_3d' not in dataset[subject][action]:
                continue
                
            for cam_idx in range(len(keypoints[subject][action])):
                
                # We check for >= instead of == because some videos in H3.6M contain extra frames
                mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length
                
                if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                    # Shorten sequence
                    keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

            assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])
            
    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                # Normalize camera frame
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = kps

    from common.generators import UnchunkedGenerator
    input_keypoints = keypoints[vid_name]['custom'][0].copy()

    gen = UnchunkedGenerator(None, None, [input_keypoints], pad=121, causal_shift=0, augment=True, kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)

    ret = np.array([])
    predicted_3d_pos = None

    for _, batch, batch_2d in gen.next_epoch():

        for i in range(0, batch_2d.shape[1] - 242):
            
            inputs_2d = batch_2d[:,i:i+243,:,:].astype('float32')
            
            predicted_3d_pos = model.execute([inputs_2d])[0]

            if gen.augment_enabled():
                # unflip
                predicted_3d_pos[1, :, :, 0] *= -1
                predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
                predicted_3d_pos = np.mean(predicted_3d_pos, axis=0, keepdims=True)
            
            output = predicted_3d_pos.squeeze(axis=0)
            
            if (i == 0):
                ret = output
            else:
                ret = np.concatenate((ret, output), axis=0)

    # visualization
    cam = dataset.cameras()[vid_name][0]
    for subject in dataset.cameras():
        if 'orientation' in dataset.cameras()[subject][0]:
            rot = dataset.cameras()[subject][0]['orientation']
            break
    from common.camera import camera_to_world
    prediction = camera_to_world(ret, R=rot, t=0)

    prediction[:, :, 2] -= np.min(prediction[:, :, 2])

    anim_output = {'Reconstruction': prediction}

    from common.camera import image_coordinates
    input_keypoints = image_coordinates(input_keypoints[..., :2], w=cam['res_w'], h=cam['res_h'])

    from common.visualization import render_animation
    render_animation(input_keypoints, keypoints_metadata, anim_output, dataset.skeleton(), dataset.fps(), 3000, cam['azimuth'], "{}/output.mp4".format(output_dir), limit=-1, downsample=1, size=6, input_video_path="./input/{}".format(vid_name), viewport=(cam['res_w'], cam['res_h']), input_video_skip=0)



# main method
if __name__ == '__main__':
    # parse argument from stdin
    description = 'Loading input npz file for 3D Lifting'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--input_vid_name', type=str, default='left_turn.mov', help="Input npz file directory")
    parser.add_argument('--output_dir', type=str, default='./output', help="Output Path")

    args = parser.parse_args()

    # loading acl_resource
    acl_resource = AclResource()
    acl_resource.init()

    # import model
    model = Model(acl_resource, MODEL_PATH)
    # predict final result and visualization
    execute(model, args.output_dir, args.input_vid_name)
