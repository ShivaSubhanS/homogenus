# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# Expressive Body Capture: 3D Hands, Face, and Body from a Single Image <https://arxiv.org/abs/1904.05866>
#
# Code Developed by: Nima Ghorbani <https://www.linkedin.com/in/nghorbani/>
# 2018.11.07

import tensorflow as tf
import numpy as np
import os, glob
import sys

# Add homogenus directory to sys.path to locate homogenus.tools
#sys.path.append('/home/repos/homogenus')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

class Homogenus_infer(object):

    def __init__(self, trained_model_dir):
        '''
        :param trained_model_dir: the directory where you have put the homogenus TF trained models
        '''
        best_model_fname = sorted(glob.glob(os.path.join(trained_model_dir, '*.ckpt.index')), key=os.path.getmtime)
        if len(best_model_fname):
            self.best_model_fname = best_model_fname[-1].replace('.index', '')
        else:
            raise ValueError('Could not find TF trained model in the provided directory --trained_model_dir=%s. Make sure you have downloaded them there.' % trained_model_dir)

        # Use TensorFlow 2.x compatibility mode to load checkpoint
        print('Loading checkpoint %s..' % self.best_model_fname)
        tf.compat.v1.disable_eager_execution()  # Disable eager execution for TF1.x compatibility
        self.graph = tf.compat.v1.get_default_graph()
        self.sess = tf.compat.v1.Session()
        self.saver = tf.compat.v1.train.import_meta_graph(self.best_model_fname + '.meta')
        self.prepare()
        self.input_tensor = self.graph.get_tensor_by_name('input_images:0')
        self.output_tensor = self.graph.get_tensor_by_name('probs_op:0')

    def prepare(self):
        # Restore the checkpoint weights
        self.saver.restore(self.sess, self.best_model_fname)
        print('Model loaded and ready for inference.')

    def predict_genders(self, images_indir=None, openpose_indir=None, images_outdir=None, openpose_outdir=None, pose_format='openpose', video_file=None):
        '''
        Given a directory with images and another directory with corresponding pose keypoint jsons will
        augment pose jsons with gender labels.

        :param images_indir: Input directory of images with common extensions (optional if video_file provided)
        :param openpose_indir: Input directory of pose keypoint jsons (OpenPose or AlphaPose format)
        :param images_outdir: If given will overlay the detected gender on detected humans that pass the criteria
        :param openpose_outdir: If given will dump the gendered pose files in this directory. If None, won't save files
        :param pose_format: Either 'openpose' or 'alphapose' to specify the keypoint format
        :param video_file: Path to video file. If provided, will extract frame 1 and process it
        :return: Dictionary mapping image filenames to list of gender predictions
        '''
        import os, sys
        import json
        import tempfile
        import shutil
        from homogenus.tools.image_tools import put_text_in_image, fontColors, read_prep_image, save_images
        from homogenus.tools.body_cropper import cropout_openpose, should_accept_pose, should_accept_pose_alphapose
        import cv2
        from homogenus.tools.omni_tools import makepath

        # Handle video input
        temp_dir = None
        video_frame_mapping = None  # Maps extracted frame name to original video frame number
        
        if video_file is not None:
            if not os.path.exists(video_file):
                raise ValueError(f"Video file not found: {video_file}")

            # Create temporary directory for extracted frame
            temp_dir = tempfile.mkdtemp(prefix="homogenus_video_")
            images_indir = temp_dir

            # Extract frame 1 from video
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_file}")

            # Read first frame
            ret, frame = cap.read()
            if not ret:
                cap.release()
                shutil.rmtree(temp_dir)
                raise ValueError(f"Could not read first frame from video: {video_file}")

            # Save frame as image - use the same naming as AlphaPose would (0.jpg for first frame)
            frame_path = os.path.join(temp_dir, "0.jpg")
            cv2.imwrite(frame_path, frame)
            cap.release()

            sys.stdout.write(f'Extracted first frame from video (frame 0): {frame_path}\n')

        if images_indir is None:
            raise ValueError("Either images_indir or video_file must be provided")

        sys.stdout.write('\nRunning homogenus on --images_indir=%s --openpose_indir=%s --pose_format=%s --video_file=%s\n' % (images_indir, openpose_indir, pose_format, video_file))

        im_fnames = []
        for img_ext in ['png', 'jpg', 'jpeg', 'bmp']:
            im_fnames.extend(glob.glob(os.path.join(images_indir, '*.%s' % img_ext)))

        if len(im_fnames):
            sys.stdout.write('Found %d images\n' % len(im_fnames))
        else:
            raise ValueError('No images could be found in %s' % images_indir)

        crop_margin = 0.08

        if images_outdir is not None:
            makepath(images_outdir)

        # Only create output directory if specified
        if openpose_outdir is not None:
            makepath(openpose_outdir)

        # Dictionary to store gender predictions for each image
        gender_results = {}

        for im_fname in im_fnames:
            im_basename = os.path.basename(im_fname)
            img_ext = im_basename.split('.')[-1]
            
            # Initialize gender list for this image
            gender_results[im_basename] = []
            
            openpose_in_fname = os.path.join(openpose_indir, im_basename.replace('.%s' % img_ext, '_keypoints.json'))
            
            # For AlphaPose format, the JSON file might be named differently
            if pose_format == 'alphapose' and not os.path.exists(openpose_in_fname):
                # Try alphapose-results.json format
                openpose_in_fname = os.path.join(openpose_indir, 'alphapose-results.json')
            
            if not os.path.exists(openpose_in_fname):
                sys.stdout.write('Warning: No keypoint file found for %s, skipping...\n' % im_basename)
                continue
                
            with open(openpose_in_fname, 'r') as f:
                pose_data = json.load(f)

            # Handle different JSON structures
            if pose_format == 'alphapose':
                # AlphaPose format: list of detections with image_id
                if isinstance(pose_data, list):
                    # Filter detections for this specific image
                    people_data = [p for p in pose_data if p.get('image_id') == im_basename]
                    if not people_data:
                        sys.stdout.write('Warning: No detections found for %s, skipping...\n' % im_basename)
                        continue
                else:
                    people_data = [pose_data]  # Single detection
            else:
                # OpenPose format: dict with 'people' key
                people_data = pose_data.get('people', [])

            im_orig = cv2.imread(im_fname, 3)[:, :, ::-1].copy()
            for opnpose_pIdx in range(len(people_data)):
                # Extract keypoints based on format
                if pose_format == 'alphapose':
                    keypoints = people_data[opnpose_pIdx].get('keypoints', [])
                    pose = np.asarray(keypoints).reshape(-1, 3)
                    # Use AlphaPose-specific validation
                    if not should_accept_pose_alphapose(pose, human_prob_thr=0.5):
                        continue
                else:
                    pose = np.asarray(people_data[opnpose_pIdx]['pose_keypoints_2d']).reshape(-1, 3)
                    if not should_accept_pose(pose, human_prob_thr=0.5):
                        continue

                crop_info = cropout_openpose(im_fname, pose, want_image=True, crop_margin=crop_margin)
                cropped_image = crop_info['cropped_image']
                if cropped_image.shape[0] < 200 or cropped_image.shape[1] < 200:
                    continue

                img = read_prep_image(cropped_image)[np.newaxis]

                # TensorFlow 1.x-style inference using compatibility mode
                probs_ob = self.sess.run(self.output_tensor, feed_dict={self.input_tensor: img})[0]

                gender_id = np.argmax(probs_ob, axis=0)
                gender_prob = probs_ob[gender_id]
                gender_pd = 'male' if gender_id == 0 else 'female'

                color = 'green'
                text = 'pred:%s[%.3f]' % (gender_pd, gender_prob)

                x1 = crop_info['crop_boundary']['offset_width']
                y1 = crop_info['crop_boundary']['offset_height']
                x2 = crop_info['crop_boundary']['target_width'] + x1
                y2 = crop_info['crop_boundary']['target_height'] + y1
                im_orig = cv2.rectangle(im_orig, (x1, y1), (x2, y2), fontColors[color], 2)
                im_orig = put_text_in_image(im_orig, [text], color, (x1, y1))[0]

                # Store gender prediction in the appropriate format
                if pose_format == 'alphapose':
                    people_data[opnpose_pIdx]['gender_pd'] = gender_pd
                else:
                    pose_data['people'][opnpose_pIdx]['gender_pd'] = gender_pd

                # Add gender to results dictionary
                gender_results[im_basename].append({
                    'person_id': opnpose_pIdx,
                    'gender': gender_pd,
                    'crop_boundary': crop_info['crop_boundary']
                })

                sys.stdout.write('%s -- person_id %d --> %s\n' % (im_fname, opnpose_pIdx, gender_pd))

            if images_outdir is not None:
                save_images(im_orig, images_outdir, [os.path.basename(im_fname)])
            
            # Save the updated pose data only if openpose_outdir is specified
            if openpose_outdir is not None:
                openpose_out_fname = os.path.join(openpose_outdir, im_basename.replace('.%s' % img_ext, '_keypoints.json'))
                if pose_format == 'alphapose':
                    # For AlphaPose, we need to update the original list
                    # This is a simplified approach - in practice, you may want to write to a separate file
                    openpose_out_fname = os.path.join(openpose_outdir, 'alphapose-results_gendered.json')
                    # Load existing results if they exist, otherwise start fresh
                    if os.path.exists(openpose_out_fname):
                        with open(openpose_out_fname, 'r') as f:
                            existing_data = json.load(f)
                        # Update or append people_data
                        existing_data.extend(people_data)
                        with open(openpose_out_fname, 'w') as f:
                            json.dump(existing_data, f)
                    else:
                        with open(openpose_out_fname, 'w') as f:
                            json.dump(people_data, f)
                else:
                    with open(openpose_out_fname, 'w') as f:
                        json.dump(pose_data, f)

        # Cleanup temporary directory if video was processed
        if temp_dir is not None:
            shutil.rmtree(temp_dir)
            sys.stdout.write(f'Cleaned up temporary directory: {temp_dir}\n')

        if images_outdir is not None:
            sys.stdout.write('Dumped overlayed images at %s\n' % images_outdir)
        if openpose_outdir is not None:
            sys.stdout.write('Dumped gendered openpose keypoints at %s\n' % openpose_outdir)

        # Return gender predictions
        return gender_results

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-tm", "--trained_model_dir", default="./homogenus/trained_models/tf/", help="The path to the directory holding homogenus trained models in TF.")
    parser.add_argument("-ii", "--images_indir", default=None, help="Directory of the input images. Optional if --video_file is provided.")
    parser.add_argument("-oi", "--openpose_indir", required=True, help="Directory of pose keypoints, e.g. json files (OpenPose or AlphaPose format).")
    parser.add_argument("-io", "--images_outdir", default=None, help="Directory to put predicted gender overlays. If not given, wont produce any overlays.")
    parser.add_argument("-oo", "--openpose_outdir", default=None, help="Directory to put the pose gendered keypoints. If not given, it will augment the original pose json files.")
    parser.add_argument("-pf", "--pose_format", default="openpose", choices=['openpose', 'alphapose'], help="Format of the pose keypoints: 'openpose' or 'alphapose'.")
    parser.add_argument("-vf", "--video_file", default=None, help="Path to video file. If provided, will extract frame 1 and process it. Overrides --images_indir.")

    ps = parser.parse_args()

    # Validate arguments
    if ps.video_file is None and ps.images_indir is None:
        parser.error("Either --images_indir or --video_file must be provided")

    hg = Homogenus_infer(trained_model_dir=ps.trained_model_dir)
    results = hg.predict_genders(images_indir=ps.images_indir, openpose_indir=ps.openpose_indir,
                      images_outdir=ps.images_outdir, openpose_outdir=ps.openpose_outdir,
                      pose_format=ps.pose_format, video_file=ps.video_file)
    
    # Print summary of results
    print("\n" + "=" * 60)
    print("Gender Prediction Results Summary")
    print("=" * 60)
    for image_name, genders in results.items():
        if genders:
            print(f"\n{image_name}:")
            for person in genders:
                print(f"  Person {person['person_id']}: {person['gender']}")
        else:
            print(f"\n{image_name}: No valid detections")
    print("=" * 60)
