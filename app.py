import cv2
import imageio
import numpy as np
import argparse
import os
import open3d as o3d
from point_utils import project_pcd, get_depth_map, unproject_pcd, covisibility_mask, mask_pcd_2d

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, default="demo.png")
parser.add_argument("--wo_sam", action="store_true")
parser.add_argument("--save_path", type=str, default="output/")

parser.add_argument("--dataset_path", type=str, default="")
parser.add_argument("--dataset_split", type=str, default="test")
parser.add_argument("--dataset_skip", type=int, default=10)

parser.add_argument("--pcd_path", type=str, default="")
parser.add_argument("--mesh_path", type=str, default="")

args = parser.parse_args()
args.use_sam = not args.wo_sam

cv2.namedWindow("2D Annotator")

vis = None
pnt_w = None
pnt_frame_buffer = []
pnt_frame_mask = None
pnt_mask = None

# Set SAM Predictor
if args.use_sam:
    import sys
    sys.path.append("./segment-anything")
    from segment_anything import sam_model_registry, SamPredictor
    sam_checkpoint = "segment-anything/checkpoints/sam_vit_h_4b8939.pth"
    # sam_checkpoint = "segment-anything/checkpoints/sam_vit_b_01ec64.pth"
    sam = sam_model_registry['vit_h'](checkpoint=sam_checkpoint)
    sam.to("cuda")

    predictor = SamPredictor(sam)
else:
    predictor = None

obj_size = 4
obj_mode = 'pcd'
# Set 3D point cloud
if args.pcd_path != "":
    pcd_t = o3d.t.io.read_point_cloud(args.pcd_path)
    pcd = pcd_t.to_legacy()
    pnt_w = np.asarray(pcd.points)
    color_ori = np.asarray(pcd.colors).copy()
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox_size = bbox.get_max_bound() - bbox.get_min_bound()
    obj_size = np.max(bbox_size)
    obj = pcd
elif args.mesh_path != "": # pcd has higher priority
    obj_mode = 'mesh'
    mesh_t = o3d.t.io.read_triangle_mesh(args.mesh_path)
    mesh = mesh_t.to_legacy()
    mesh.compute_vertex_normals()
    pnt_w = np.asarray(mesh.vertices)
    color_ori = np.asarray(mesh.vertex_colors).copy()
    if color_ori.shape[0] == 0:
        color_ori = np.ones_like(pnt_w)
    bbox = mesh.get_axis_aligned_bounding_box()
    bbox_size = bbox.get_max_bound() - bbox.get_min_bound()
    obj_size = np.max(bbox_size)
    obj = mesh

vis = o3d.visualization.Visualizer()
vis.create_window("3D Visualizer", 800, 800)
vis.add_geometry(obj)




# Initialize the list of keypoints
keypoints = []

# Colors for different modes
colors = [(0, 0, 255), (0, 255, 0)]
pnt_sel_color = np.array([0, 0, 255])
pnt_sel_color_global = np.array([255, 0, 0])
pnt_mask_idx = -1
# Initialize the mode
mode = 1
sel_mode = 1 # 0: single frame, 1: multi frame

depth_frames = []
image_idx = 0
if args.dataset_path != "":
    from nerf_synthetic import NeRFSynthetic
    data = NeRFSynthetic(args.dataset_path, split=args.dataset_split, testskip=args.dataset_skip)
    n_images = len(data)
    original_image_rgb, c2w, image_path = data[image_idx]
    if pnt_w is not None:
        uv_cam, pnt_cam, depth = project_pcd(pnt_w, data.K, c2w)
        depth_map, index = get_depth_map(uv_cam, depth, *original_image_rgb.shape[:2], scale=3)

else:
    # Load the image
    image_path = args.image
    # original_image = cv2.imread(image_path)
    # original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_rgb = imageio.imread(image_path)
    if original_image_rgb.shape[-1] == 4:
        original_image_rgb = original_image_rgb / 255.
        original_image_rgb = original_image_rgb[:,:,:3] * original_image_rgb[:,:,3:4] + (1 - original_image_rgb[:,:,3:4])
        original_image_rgb = (original_image_rgb.clip(0, 1) * 255).astype(np.uint8)

original_image = cv2.cvtColor(original_image_rgb, cv2.COLOR_RGB2BGR)
if predictor is not None:
    predictor.set_image(original_image_rgb)
image = original_image.copy()
logits = None
mask = None
print("Image loaded")

# Mouse callback function
def annotate_keypoints(event, x, y, flags, param):
    global keypoints, mode, image, logits, mask

    if event == cv2.EVENT_LBUTTONDOWN:
        # Add the keypoint and mode to the list
        keypoints.append((x, y, mode))
        # print("Keypoint added:", (x, y, mode))
        if predictor is not None:
            # Run SAM
            input_point = np.array([pts[:2] for pts in keypoints])
            input_label = np.array([pts[2] for pts in keypoints])
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                mask_input=logits,
                multimask_output=False,
            )
            mask = masks[0]

            color_mask = (np.random.random(3) * 255).astype(np.uint8)
            colored_mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2) * color_mask
            image = cv2.addWeighted(original_image, 0.5, colored_mask, 0.5, 0)
        else:
            image = original_image.copy()

        # Draw a circle at the keypoint position with the corresponding color
        for x, y, m in keypoints:
            cv2.circle(image, (x, y), 3, colors[m], -1)
            cv2.imshow("2D Annotator", image)

# Initialize the mask transparency
depth_ratio = 100

# Trackbar callback function
def on_trackbar(val):
    global depth_ratio
    depth_ratio = val
    # update_image()
    cv2.imshow("2D Annotator", image)


# Create a window and set the mouse callback function
cv2.setMouseCallback("2D Annotator", annotate_keypoints)

# Create a trackbar (slider) to control the depth_ratio of the mask
cv2.createTrackbar("Depth Percentage", "2D Annotator", depth_ratio, 100, on_trackbar)

print("Start annotating keypoints")
while True:
    cv2.imshow("2D Annotator", image)
    key = cv2.waitKey(1) & 0xFF

    # Press 'm' to toggle between modes
    if key == ord("m"):
        mode = (mode + 1) % 2

    # Press 'u' to undo the last keypoint
    if key == ord("z"):
        if keypoints:
            # Remove the last keypoint
            keypoints.pop()

            # Redraw the keypoints
            image = original_image.copy()
            for x, y, m in keypoints:
                cv2.circle(image, (x, y), 3, colors[m], -1)
            cv2.imshow("2D Annotator", image)

    # Press 's' to save the mask and keypoints
    if key == ord("s"):
        image_name = os.path.basename(image_path)
        if mask is not None:
            os.makedirs(args.save_path, exist_ok=True)
            mask_path = os.path.join(args.save_path, image_name)
            imageio.imwrite(mask_path + '.png', (mask[..., None]*255).astype(np.uint8))
            np.savetxt(mask_path + ".txt", keypoints, fmt="%d")

    # Press 'n' to go to the next image
    if key == ord("n"):
        image_idx = (image_idx + 1) % n_images
        if args.dataset_path != "":
            original_image_rgb, c2w, image_path = data[image_idx]
            original_image = cv2.cvtColor(original_image_rgb, cv2.COLOR_RGB2BGR)
            if pnt_w is not None:
                uv_cam, pnt_cam, depth = project_pcd(pnt_w, data.K, c2w)
                depth_map, index = get_depth_map(uv_cam, depth, *original_image_rgb.shape[:2], scale=3)

            if predictor is not None:
                predictor.set_image(original_image_rgb)
            image = original_image.copy()
            keypoints = []
            logits = None
            mask = None

    # Press 'p' to go to the previous image
    if key == ord("p"):
        image_idx = (image_idx - 1) % n_images
        if args.dataset_path != "":
            original_image_rgb, c2w, image_path = data[image_idx]
            original_image = cv2.cvtColor(original_image_rgb, cv2.COLOR_RGB2BGR)
            if pnt_w is not None:
                uv_cam, pnt_cam, depth = project_pcd(pnt_w, data.K, c2w)
                depth_map, index = get_depth_map(uv_cam, depth, *original_image_rgb.shape[:2], scale=3)

            if predictor is not None:
                predictor.set_image(original_image_rgb)
            image = original_image.copy()
            keypoints = []
            logits = None
            mask = None

    # Press 'r' to reset the image
    if key == ord("r"):
        image = original_image.copy()
        keypoints = []
        logits = None
        mask = None
        cv2.imshow("2D Annotator", image)

    # Press 'c' to crop the point cloud
    if key == ord("c") and pnt_w is not None and mask is not None:
        if depth_ratio == 100:
            pnt_frame_mask = mask_pcd_2d(uv_cam, mask)[..., None]
        else:
            depth_thresh = obj_size * depth_ratio / 100
            pnt_frame_mask = mask_pcd_2d(uv_cam, mask, 0.5, depth_map, depth, depth_thresh)[..., None]
        pnt_mask_idx = image_idx
    
        color = (~pnt_frame_mask)  * color_ori + pnt_frame_mask * pnt_sel_color / 225.
        if obj_mode == 'pcd':
            obj.colors = o3d.utility.Vector3dVector(color)
        else:
            obj.vertex_colors = o3d.utility.Vector3dVector(color)
        vis.update_geometry(obj)

    # Press 'u' to union the point cloud
    if key == ord("u") and pnt_w is not None and mask is not None:
        if pnt_mask is None:
            pnt_mask = pnt_frame_mask.copy()
        else:
            pnt_mask = np.logical_or(pnt_mask, pnt_frame_mask)
        color = (~pnt_mask)  * color_ori + pnt_mask * pnt_sel_color_global / 225.
        if obj_mode == 'pcd':
            obj.colors = o3d.utility.Vector3dVector(color)
        else:
            obj.vertex_colors = o3d.utility.Vector3dVector(color)
        vis.update_geometry(obj)
    
    # Press 'x' to intersect the point cloud
    if key == ord("x") and pnt_w is not None and mask is not None:
        if pnt_mask is None:
            pnt_mask = pnt_frame_mask.copy()
        else:
            pnt_mask = np.logical_and(pnt_mask, pnt_frame_mask)
        color = (~pnt_mask)  * color_ori + pnt_mask * pnt_sel_color_global / 225.
        if obj_mode == 'pcd':
            obj.colors = o3d.utility.Vector3dVector(color)
        else:
            obj.vertex_colors = o3d.utility.Vector3dVector(color)
        vis.update_geometry(obj)
    
    # Press 'k' to switch the sel mode
    if key == ord("k"):
        sel_mode = (sel_mode + 1) % 2
        print("sel_mode:", 'single frame' if sel_mode == 0 else 'multi frame')

    # Press 'a' to add pnt_frame_mask for multi frame selection
    if key == ord("a") and pnt_w is not None and pnt_frame_mask is not None:
        pnt_frame_buffer.append((pnt_frame_mask, c2w, depth_map, uv_cam))
        print('Add pnt_frame_mask to buffer, buffer size:', len(pnt_frame_buffer))

    # Press 'e' to export the masked point cloud
    if key == ord("e") and pnt_w is not None and pnt_mask is not None:
        if obj_mode == 'pcd': # 'mesh' is not supported yet
            pcd_t.point.flags = (pnt_mask * 32).astype(np.int32)
            pcd_name = os.path.basename(args.pcd_path)[:-4] + '_mask.ply'
            os.makedirs(args.save_path, exist_ok=True)
            o3d.t.io.write_point_cloud(os.path.join(args.save_path, pcd_name), pcd_t)
            print('Export masked point cloud to', os.path.join(args.save_path, pcd_name))
                                    
    # Press 'q' to exit
    if key == ord("q"):
        break
    
    if vis is not None:
        vis.poll_events()
        vis.update_renderer()

# Close all windows
cv2.destroyAllWindows()
if vis is not None:
    vis.destroy_window()

# Print the annotated keypoints
print("Annotated keypoints:", keypoints)