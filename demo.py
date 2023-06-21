import os
import os.path as osp
import smplx
import numpy as np
import torch
import trimesh
import pyrender
import pyrender.constants


def key1_func(viewer, val):
    val[0] = 0
    viewer._message_text = 'Setting height'
    viewer._message_opac = 1.0 + viewer._ticks_till_fade

def key2_func(viewer, val):
    val[0] = 1
    viewer._message_text = 'Setting inseam height'
    viewer._message_opac = 1.0 + viewer._ticks_till_fade

def key3_func(viewer, val):
    val[0] = 2
    viewer._message_text = 'Setting arm span'
    viewer._message_opac = 1.0 + viewer._ticks_till_fade

def key4_func(viewer, val):
    val[0] = 3
    viewer._message_text = 'Setting chest'
    viewer._message_opac = 1.0 + viewer._ticks_till_fade

def key5_func(viewer, val):
    val[0] = 4
    viewer._message_text = 'Setting waist'
    viewer._message_opac = 1.0 + viewer._ticks_till_fade

def key6_func(viewer, val):
    val[0] = 5
    viewer._message_text = 'Setting hip'
    viewer._message_opac = 1.0 + viewer._ticks_till_fade

def main():
    device = 'cuda:0'

    model_params = dict(model_path='data/body_models',
                        model_type='smplx',
                        num_betas=10,
                        num_pca_comps=12,
                        flat_hand_mean=False,
                        ext='npz')
    model = smplx.create(gender='male', **model_params).to(device)

    if model_params['model_type'] == 'smplx':
        left_hand_pose = torch.linalg.lstsq(model.left_hand_components.T, -model.left_hand_mean)[0][None].to(device)
        right_hand_pose = torch.linalg.lstsq(model.right_hand_components.T, -model.right_hand_mean)[0][None].to(device)
    else:
        left_hand_pose = None
        right_hand_pose = None

    base = [torch.randn(1, 10)]
    print(base[0])
    base[0] = base[0].to(device)

    hip = torch.from_numpy(np.load('data/smplx_params/smplx_params_hip_1cm_b10.npz')['betas']).to(device)
    height = torch.from_numpy(np.load('data/smplx_params/smplx_params_height_1cm_b10.npz')['betas']).to(device)
    leg = torch.from_numpy(np.load('data/smplx_params/smplx_params_leg_1cm_b10.npz')['betas']).to(device)
    arm_span = torch.from_numpy(np.load('data/smplx_params/smplx_params_arm_span_1cm_b10.npz')['betas']).to(device)
    chest = torch.from_numpy(np.load('data/smplx_params/smplx_params_chest_1cm_b10.npz')['betas']).to(device)
    waist = torch.from_numpy(np.load('data/smplx_params/smplx_params_waist_1cm_b10.npz')['betas']).to(device)

    # modify here
    height = height / 0.992

    weights = [0.0] * 6
    curr_idx = [0]

    def calc_betas(base_):
        return base_ + weights[0] * height + weights[1] * leg + weights[2] * arm_span  + \
            weights[3] * chest + weights[4] * waist + weights[5] * hip

    betas = calc_betas(base[0])

    scene = pyrender.Scene()
    scene.bg_color = np.zeros(3)

    pc = pyrender.PerspectiveCamera(yfov=np.pi / 2, aspectRatio=1.0)
    camera = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1200.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    scene.add(pc, pose=camera)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
    scene.add(light, pose=camera)

    smpl_output = [model(betas=betas,
                         left_hand_pose=left_hand_pose,
                         right_hand_pose=right_hand_pose)]
    shaped_vertices = smpl_output[0]['v_shaped' if model_params['model_type'] == 'smplx' else 'vertices']
    tm = trimesh.Trimesh(vertices=shaped_vertices.detach().cpu().numpy().squeeze() * 1000.0,
                         faces=model.faces,
                         vertex_colors=np.array([183.0, 183.0, 237.0]) if model.gender == 'male'
                                       else np.array([237.0, 183.0, 183.0]))
    vmin = np.min(tm.vertices, axis=0)
    tm.vertices -= (np.array([[0.0, vmin[1] + 1000.0, 0.0]]))

    mesh = pyrender.Mesh.from_trimesh(tm, smooth=True, wireframe=False)
    mesh_node = [scene.add(mesh, 'mesh')]

    name_list = ['height', 'inseam height', 'arm span', 'shoulder height', 'chest', 'waist', 'hip']

    def add_value(viewer_, weights_, idx_, mesh_node_, value_):
        weights_[idx_[0]] = weights_[idx_[0]] + value_
        viewer_._message_text = name_list[idx_[0]] + ' + ' + str(weights_[idx_[0]]) + 'cm'
        viewer_._message_opac = 1.0 + viewer_._ticks_till_fade
        smpl_output[0] = model(betas=calc_betas(base[0]),
                               left_hand_pose=left_hand_pose,
                               right_hand_pose=right_hand_pose)
        shaped_vertices_ = smpl_output[0]['v_shaped' if model_params['model_type'] == 'smplx' else 'vertices']

        tm_ = trimesh.Trimesh(vertices=shaped_vertices_.detach().cpu().numpy().squeeze() * 1000.0,
                              faces=model.faces,
                              vertex_colors=np.array([183.0, 183.0, 237.0]) if model.gender == 'male'
                              else np.array([237.0, 183.0, 183.0]))
        vmin_ = np.min(tm_.vertices, axis=0)
        vmax_ = np.max(tm_.vertices, axis=0)
        center_ = (vmin_ + vmax_) / 2.0
        tm_.vertices -= (np.array([[0.0, vmin[1] + 1000.0, 0.0]]))

        mesh_ = pyrender.Mesh.from_trimesh(tm_, smooth=True, wireframe=False)
        scene.remove_node(mesh_node_[0])
        mesh_node_[0] = scene.add(mesh_, 'mesh')



    def random_smpl_shape(viewer_, base_, mesh_node_):
        base_[0] = torch.randn(1, 10)
        print(base[0])
        base_[0] = base_[0].to(device)
        smpl_output[0] = model(betas=calc_betas(base_[0]),
                               left_hand_pose=left_hand_pose,
                               right_hand_pose=right_hand_pose)
        shaped_vertices_ = smpl_output[0]['v_shaped' if model_params['model_type'] == 'smplx' else 'vertices']

        tm_ = trimesh.Trimesh(vertices=shaped_vertices_.detach().cpu().numpy().squeeze() * 1000.0,
                              faces=model.faces,
                              vertex_colors=np.array([183.0, 183.0, 237.0]) if model.gender == 'male'
                              else np.array([237.0, 183.0, 183.0]))
        vmin_ = np.min(tm_.vertices, axis=0)
        tm_.vertices -= (np.array([[0.0, vmin[1] + 1000.0, 0.0]]))

        mesh_ = pyrender.Mesh.from_trimesh(tm_, smooth=True, wireframe=False)
        scene.remove_node(mesh_node_[0])
        mesh_node_[0] = scene.add(mesh_, 'mesh')


        viewer_._message_text = 'New random SMPL(-X) shape'
        viewer_._message_opac = 1.0 + viewer_._ticks_till_fade

    key_mapping={
        '1': (key1_func, [curr_idx]),
        '2': (key2_func, [curr_idx]),
        '3': (key3_func, [curr_idx]),
        '4': (key4_func, [curr_idx]),
        '5': (key5_func, [curr_idx]),
        '6': (key6_func, [curr_idx]),
        '-': (add_value, [weights, curr_idx, mesh_node, -0.5]),
        '=': (add_value, [weights, curr_idx, mesh_node, 0.5]),
        ' ': (random_smpl_shape, [base, mesh_node])
    }

    r = pyrender.Viewer(scene, viewport_size=[800, 800], use_raymond_lighting=True, registered_keys=key_mapping)


if __name__ == '__main__':
    main()
