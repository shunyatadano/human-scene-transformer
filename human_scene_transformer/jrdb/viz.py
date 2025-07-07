# Copyright 2024 The human_scene_transformer Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Visualization functions for JRDB dataset."""

import os
import imageio
import itertools
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import tensorflow as tf
import gin
import tempfile
import shutil

from absl import app
from absl import flags
from absl import logging

from human_scene_transformer.jrdb import dataset_params as jrdb_dp
from human_scene_transformer.jrdb import input_fn as jrdb_input_fn
from human_scene_transformer.model import model as hst_model
from human_scene_transformer.model import model_params as mp

# Path to the downloaded font
FONT_PATH = 'jrdb/fonts/ipagp00303/ipagp.ttf'


def create_prediction_gif(model, dataset_sample, output_dir, num_history_steps, num_prediction_steps, sample_index, font_prop):
    """
    Generates a GIF of the predicted trajectory against the ground truth.

    Args:
        model: The trained Human Scene Transformer model.
        dataset_sample: A single unbatched dataset sample.
        output_dir: The directory to save the output GIF.
        num_history_steps: Number of historical steps in the input.
        num_prediction_steps: Number of prediction steps.
        sample_index: The index of the sample for naming the output file.
        font_prop: Font properties for Japanese characters.
    """
    batched_sample = {k: tf.expand_dims(v, axis=0) for k, v in dataset_sample.items()}
    predictions, _ = model(batched_sample, training=False)

    all_agent_positions = batched_sample['agents/position'][0]
    predicted_positions = predictions['agents/position'][0]

    agent_idx = 0
    history_positions = all_agent_positions[agent_idx, :num_history_steps, :2]
    gt_future_positions_agent = all_agent_positions[agent_idx, num_history_steps:, :2]

    # Select the first mode for visualization
    predicted_full_trajectory_agent = predicted_positions[0, agent_idx, :, :]
    predicted_future_positions_agent = predicted_full_trajectory_agent[num_history_steps:, :]

    # Calculate the plotting range to encompass all trajectories
    all_points = np.vstack((history_positions, gt_future_positions_agent, predicted_future_positions_agent))
    x_min, y_min = np.min(all_points, axis=0)
    x_max, y_max = np.max(all_points, axis=0)

    if not np.all(np.isfinite([x_min, y_min, x_max, y_max])):
        logging.warning(f'Skipping sample {sample_index} due to invalid data.')
        return

    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    max_range = max(x_max - x_min, y_max - y_min)
    plot_half_range = max_range * 0.6

    plot_xlim = (x_center - plot_half_range, x_center + plot_half_range)
    plot_ylim = (y_center - plot_half_range, y_center + plot_half_range)

    temp_dir = tempfile.mkdtemp()
    frame_files = []

    # Frame 1: History
    plt.figure(figsize=(8, 8))
    plt.plot(history_positions[:, 0], history_positions[:, 1], 'bo-', label='観測軌道 (正解)')
    plt.xlabel('X座標', fontproperties=font_prop)
    plt.ylabel('Y座標', fontproperties=font_prop)
    plt.title(f'エージェント {agent_idx} 軌道予測 (サンプル {sample_index})', fontproperties=font_prop)
    plt.legend(prop=font_prop)
    plt.grid(True)
    plt.xlim(plot_xlim)
    plt.ylim(plot_ylim)
    frame_path = os.path.join(temp_dir, 'frame_0.png')
    plt.savefig(frame_path)
    frame_files.append(frame_path)
    plt.close()

    # Frame 2: Ground Truth Future
    plt.figure(figsize=(8, 8))
    plt.plot(history_positions[:, 0], history_positions[:, 1], 'bo-', label='観測軌道 (正解)')
    plt.plot(gt_future_positions_agent[:, 0], gt_future_positions_agent[:, 1], 'go-', label='未来軌道 (正解)')
    plt.xlabel('X座標', fontproperties=font_prop)
    plt.ylabel('Y座標', fontproperties=font_prop)
    plt.title(f'エージェント {agent_idx} 軌道予測 (サンプル {sample_index})', fontproperties=font_prop)
    plt.legend(prop=font_prop)
    plt.grid(True)
    plt.xlim(plot_xlim)
    plt.ylim(plot_ylim)
    frame_path = os.path.join(temp_dir, 'frame_1.png')
    plt.savefig(frame_path)
    frame_files.append(frame_path)
    plt.close()

    # Frame 3: Predicted Future
    plt.figure(figsize=(8, 8))
    plt.plot(history_positions[:, 0], history_positions[:, 1], 'bo-', label='観測軌道 (正解)')
    plt.plot(gt_future_positions_agent[:, 0], gt_future_positions_agent[:, 1], 'go-', label='未来軌道 (正解)')
    plt.plot(predicted_future_positions_agent[:, 0], predicted_future_positions_agent[:, 1], 'ro--', label='予測未来軌道')
    plt.xlabel('X座標', fontproperties=font_prop)
    plt.ylabel('Y座標', fontproperties=font_prop)
    plt.title(f'エージェント {agent_idx} 軌道予測 (サンプル {sample_index})', fontproperties=font_prop)
    plt.legend(prop=font_prop)
    plt.grid(True)
    plt.xlim(plot_xlim)
    plt.ylim(plot_ylim)
    frame_path = os.path.join(temp_dir, 'frame_2.png')
    plt.savefig(frame_path)
    frame_files.append(frame_path)
    plt.close()

    # Create GIF
    output_filename = os.path.join(output_dir, f'predicted_trajectory_sample_{sample_index}.gif')
    with imageio.get_writer(output_filename, mode='I', duration=1.0) as writer:
        for filename in frame_files:
            image = imageio.imread(filename)
            writer.append_data(image)
    
    shutil.rmtree(temp_dir)
    print(f"GIF saved to {output_filename}")


_MODEL_PATH = flags.DEFINE_string(
    'model_path',
    None,
    'Path to model directory (containing params/operative_config.gin and ckpts).',
)

_CKPT_PATH = flags.DEFINE_string(
    'checkpoint_path',
    None,
    'Path to model checkpoint.',
)

_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    'jrdb/plot',
    'Directory to save the output plots.',
)

def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    if not _MODEL_PATH.value or not _CKPT_PATH.value:
        raise app.UsageError('Both --model_path and --checkpoint_path must be specified.')

    # Load gin config
    gin.parse_config_files_and_bindings(
        [os.path.join(_MODEL_PATH.value, 'params', 'operative_config.gin')],
        None,
        skip_unknown=True
    )
    logging.info('Actual gin config used:\n%s', gin.config_str())

    d_params = jrdb_dp.JRDBDatasetParams(num_agents=None)
    model_p = mp.ModelParams()
    model = hst_model.HumanTrajectorySceneTransformer(model_p)

    checkpoint_mngr = tf.train.Checkpoint(model=model)
    checkpoint_mngr.restore(_CKPT_PATH.value).expect_partial()
    logging.info('Restored checkpoint: %s', _CKPT_PATH.value)

    eval_dataset = jrdb_input_fn.load_dataset(
        d_params,
        d_params.eval_scenes,
        augment=False,
        shuffle=False,
        allow_parallel=False,
        evaluation=True,
        repeat=False,
        keep_subsamples=True,
    )

    num_history_steps = d_params.num_history_steps
    num_prediction_steps = d_params.num_steps - d_params.num_history_steps

    # Set up Japanese font
    font_prop = fm.FontProperties(fname=FONT_PATH)

    # Ensure output directory exists
    os.makedirs(_OUTPUT_DIR.value, exist_ok=True)

    for i, single_sample in enumerate(itertools.islice(eval_dataset, 5)):
        create_prediction_gif(
            model,
            single_sample,
            _OUTPUT_DIR.value,
            num_history_steps,
            num_prediction_steps,
            i,
            font_prop
        )

flags.mark_flags_as_required(['model_path', 'checkpoint_path'])
if __name__ == '__main__':
    app.run(main)