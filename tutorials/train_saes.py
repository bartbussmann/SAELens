#%%
import torch
import os 
import sys 
import wandb

# sys.path.append("..")
sys.path.append("/workspace/MATS_sprint")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB__SERVICE_WAIT"] = "300"

from sae_lens.training.config import LanguageModelSAERunnerConfig
from sae_lens.training.lm_runner import SAETrainingRunner
from sae_lens.training.session_loader import LMSparseAutoencoderSessionloader
from sae_lens.training.utils import BackwardsCompatiblePickleClass
from transformer_lens import HookedSAE, HookedSAEConfig
from sae_lens import SparseAutoencoder

# base_sae = SparseAutoencoder.from_pretrained(
#     release = "jbloom/mats_sae_training_gpt2_feature_splitting_experiment/sparse_autoencoder_gpt2-small", # see other options in sae_lens/pretrained_saes.yaml
#     sae_id = "blocks.8.hook_resid_pre_768:v0" # won't always be a hook point
# )

# base_path = "jbloom/mats_sae_training_gpt2_feature_splitting_experiment/sparse_autoencoder_gpt2-small_blocks.8.hook_resid_pre_768:v0"

from pathlib import Path
def folder_to_file(folder):
    folder = Path(folder)
    files = list(folder.glob("*"))
    files = [str(f) for f in files]
    return files[0] if len(files) == 1 else files


run = wandb.init()

n = 0
artifact = run.use_artifact(
    f"jbloom/mats_sae_training_gpt2_feature_splitting_experiment/sparse_autoencoder_gpt2-small_blocks.8.hook_resid_pre_{2**n * 768}:v0",
    type="model",
)
artifact_dir = artifact.download()
file = folder_to_file(artifact_dir)
blob = torch.load(file, pickle_module=BackwardsCompatiblePickleClass)
config_dict = blob["cfg"].__dict__
state_dict = blob["state_dict"]
cfg = HookedSAEConfig(
    d_sae=config_dict["d_sae"],
    d_in=config_dict["d_in"],
    hook_name=config_dict["hook_point"],
    use_error_term=True,
    dtype=torch.float32,
    seed=None,
    device="cuda",
)
print(cfg)
sae = HookedSAE(cfg)
sae.load_state_dict(state_dict)
base_sae = sae
# model, base_sae, activations_store = LMSparseAutoencoderSessionloader.load_pretrained_sae("/workspace/config.yaml")



cfg = LanguageModelSAERunnerConfig(

    # Data Generating Function (Model + Training Distibuion)
    model_name = "gpt2-small",

    hook_point = f"blocks.8.hook_resid_pre",
    hook_point_layer = 8,
    d_in = 768,
    dataset_path = "Skylion007/openwebtext",
    is_dataset_tokenized=False,

    reconstruct_or_error_target = "reconstruction",
    
    # SAE Parameters
    expansion_factor = 1, # determines the dimension of the SAE.
    b_dec_init_method = "geometric_median",
    
    # Training Parameters
    lr = 0.0004,
    l1_coefficient = 0.00008,
    # lr_scheduler_name=None,
    train_batch_size_tokens = 4096,
    context_size = 128,
    lr_warm_up_steps=0,
    
    # Activation Store Parameters
    n_batches_in_buffer = 128,
    training_tokens = 1_000_000 * 300, # 200M tokens seems doable overnight.
    store_batch_size_prompts = 32,
    
    # Resampling protocol
    # feature_sampling_method = 'anthropic',
    use_ghost_grads=True,
    # feature_sampling_method = None,
    # feature_sampling_window = 1000,
    # feature_reinit_scale = 0.2,
    # resample_batches=1028,
    # dead_feature_window=5000,
    # dead_feature_window=50000,
    # dead_feature_threshold = 1e-8,
    
    # WANDB
    log_to_wandb = True,
    wandb_project= "mats_sae_training_gpt2_feature_splitting_experiment",
    wandb_entity = None,
    wandb_log_frequency=100,
    
    # Misc
    device = "cuda",
    seed = 42,
    n_checkpoints = 10,
    checkpoint_path = "checkpoints",
    dtype = torch.float32,
    )

sparse_autoencoder = SAETrainingRunner(cfg, base_sae).run()

# %%
