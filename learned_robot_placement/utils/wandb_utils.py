import numpy as np
import wandb

class wandbLogger:
    def __init__(self, exp_config, run_name, group_name, disabled=False):
        if disabled:
            mode="disabled"
        else:
            mode=None
        wandb.init(project="xyz", entity="irosa-ias", group=group_name, id=run_name, config=exp_config,
                reinit=True, resume="allow", settings=wandb.Settings(start_method="fork"), mode=mode)

    def run_log_wandb(self,success_rate, J, R, E, avg_episode_length, q_loss):
        wandb.log({"success_rate": success_rate, "J": J, "R": R, "entropy": E, 
                                         "avg_episode_length": avg_episode_length, "q_loss": q_loss})

    def vid_log_wandb(self,img_dataset):
        np_video = np.expand_dims(np.moveaxis(img_dataset[0][0],-1,0), axis=0)
        for tup in img_dataset:
            np_video = np.vstack((np_video, np.expand_dims(np.moveaxis(tup[0],-1,0), axis=0)))        
        wandb.log({"video": wandb.Video(np_video, fps=60, format="gif")})