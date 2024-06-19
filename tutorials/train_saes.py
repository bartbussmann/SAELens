import torch
import os
import sys
import wandb
from typing import Any, cast

sys.path.append("../../MATS_sprint")
sys.path.append("/workspace/SAELens")


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB__SERVICE_WAIT"] = "300"

from sae_lens.training.config import LanguageModelSAERunnerConfig
from sae_lens.training.lm_runner import SAETrainingRunner
from sae_lens import SparseAutoencoder


def get_base_sae(wandb_run):
    artifact = wandb_run.use_artifact(
        f"jbloom/mats_sae_training_gpt2_feature_splitting_experiment/sparse_autoencoder_gpt2-small_blocks.8.hook_resid_pre_{2**0 * 768}:v9",
        type="model",
    )
    artifact_dir = artifact.download()
    model_file = os.listdir(artifact_dir)[0]
    return SparseAutoencoder.load_from_pretrained_legacy(
        os.path.join(artifact_dir, model_file)
    )

model_name = "pythia-410m-deduped"
if model_name == "gpt2-small":
    d_in = 768
    dataset_path = "Skylion007/openwebtext"
elif model_name == "pythia-410m-deduped":
    d_in = 1024
    dataset_path = "EleutherAI/the_pile_deduplicated"

for l1_coefficient in [0.001]:
    for task in ["classic"]:

        expansion_factor = 16

        cfg = LanguageModelSAERunnerConfig(
            # Data Generating Function (Model + Training Distibuion)
            model_name=model_name,
            hook_point=f"blocks.3.hook_resid_pre",
            hook_point_layer=3,
            d_in=d_in,
            dataset_path=dataset_path,
            is_dataset_tokenized=False,
            reconstruct_or_error_target=task,
            # SAE Parameters
            expansion_factor=expansion_factor,  # determines the dimension of the SAE.
            b_dec_init_method="geometric_median",
            # Training Parameters
            lr=0.0004,
            l1_coefficient=l1_coefficient,
            lr_scheduler_name="cosineannealing",
            train_batch_size_tokens=4096,
            context_size=128,
            lr_warm_up_steps=0,
            # Activation Store Parameters
            n_batches_in_buffer=128,
            training_tokens=1_000_000 * 300,  # 200M tokens seems doable overnight.
            store_batch_size_prompts=32,
            # Resampling protocol
            # feature_sampling_method = 'anthropic',
            use_ghost_grads=True,
            # feature_sampling_method = None,
            # feature_sampling_window = 1000,
            # feature_reinit_scale = 0.2,
            # resample_batches=1028,
            # dead_feature_window=5000,
            # dead_feature_window=50000,
            dead_feature_threshold = 1e-7,
            # WANDB
            log_to_wandb=True,
            wandb_project="test",
            wandb_entity="mats-sprint",
            wandb_log_frequency=100,
            eval_every_n_wandb_logs=100000000000,
            # Misc
            device="cuda",
            seed=42,
            n_checkpoints=10,
            checkpoint_path="checkpoints",
            dtype=torch.float32,
        )

        run = wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            config=cast(Any, cfg),
            name=cfg.run_name,
            id=cfg.wandb_id,
        )

        base_sae = get_base_sae(run)

        sparse_autoencoder = SAETrainingRunner(cfg, base_sae).run()