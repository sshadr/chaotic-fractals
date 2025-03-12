import os, sys
import numpy as np
from system.renderer import GaussianSplatter, SuperSampleRender
from system.model import Model
from system.model_inference import ModelInfer, batch_point_gen
import torch
from tqdm import tqdm

from cf.images.image_io import save_image, create_video_tiled_gt
from cf.images.colorization import apply_signed_colormap
from cf.images.conversions import tensor_to_image, image_to_tensor

from utils.losses import moments_loss
from utils.optim_utils import sample_patch, prepare_output_and_logger, create_name, plot_loss_curve, write_to_json

from easydict import EasyDict as edict
from utils.notebook import dump_notebook
import time

class FractalTrainerMoments():
    def __init__(self, args, gt_img):
        super().__init__()
        '''
        args: Parser class with the entire json content
        renderer: load json contents in the trainer
        '''
        self.start_time = time.time()
        self.time_logger = edict()
        self.args = args
        self.logger = prepare_output_and_logger(self.args)
        self.writer = self.logger.tb_writer
        
        save_path = os.path.join(self.logger.output_path, 'gt.png')
        self.gt_img = gt_img
        save_image(self.gt_img, save_path)

        # prepare for optimization
        self.gt_img = image_to_tensor(self.gt_img, to_cuda=True)  

        self.cfg = self.args.get_optimizer_config()

        # Init renderer (use it for both gt and optim renders using update_points)
        self.r_config = self.args.get_render_config()
        self.dummy_point = torch.tensor([0, 0, 1], dtype=torch.float32).unsqueeze(0)
        
        self.renderer = GaussianSplatter(self.dummy_point, self.r_config)

        self.model = Model(self.args, self.logger)
        torch.save(self.model.ifs_code,f"{self.logger.init_path}/init_ifs_code.pth")

       

    def train(self):
        training_iters = self.cfg['training_iters']  # number of training iterations
        learning_rate = float(self.cfg['learning_rate'])  # "learning rate"

        def initialize_same_optimizers(model_params, lr, include_opacities=False):
            params1, params2, params3, _ = model_params

            if not include_opacities:
                optim_params1 = [
                    {'params': params1 + params2, 'lr': lr}
                ]
            else:
                optim_params1 = [
                    {'params': params1 + params2 + params3, 'lr': lr}
                ]

            optimizer1 = torch.optim.Adam(optim_params1)

            return optimizer1
        
        loss_values = []
        iter = 1
        best_loss = 100000
        best_ifs_code = self.model.ifs_code
        
        optimize_opacities = False

        optimizer1 = initialize_same_optimizers(self.model.get_learnable_params(), learning_rate, include_opacities=optimize_opacities) 

        for iter in tqdm(range(iter, training_iters+1)):            
            optimizer1.zero_grad()
            
            stime = time.time()
            points = self.model.forward() # generate points
            self.time_logger.point_gen_pass = time.time() - stime
            
            self.renderer.update_points(points)            
        
            full_view = sample_patch(1)
            stime = time.time()
            self.init_img = self.renderer.render(full_view) # only the full view is a self
            self.time_logger.render_pass = time.time() - stime

            # -------- Compute Loss ----------
            moments = moments_loss(self.gt_img, self.init_img)                        
            loss = moments
                    
            loss_values.append(loss.item())

            # -------- Compute Loss ----------

            # save after forward prop:
            if iter % 50 == 0 or iter == 1:
                torch.save(best_ifs_code, f"{self.logger.output_path}/optimized_ifs_code_{iter}.pth")

            with torch.no_grad():                
                self.save_during_training(iter, training_iters)
                    
            stime = time.time()
            loss.backward(retain_graph=True)
            self.time_logger.backward_pass = time.time() - stime

            optimizer1.step()
        
            with torch.no_grad():                        
                if loss.item() < best_loss:
                    best_ifs_code = self.model.ifs_code
                    torch.save(self.model.ifs_code, f"{self.logger.output_path}/best_optimized_ifs_code.pth")
                    
                    best_loss = loss.item()


            self.writer.add_scalar("Moments", moments.item(), iter)          
            
            tqdm.write(
                "Iteration: {}, Loss: {}".format(
                    iter, loss.item()
                )
            )   
                
        self.post_training(loss_values, best_ifs_code)
        s = time.time()
        self.val()
        print("Eval time: ", time.time() - s)

    def post_training(self, loss_values, best_ifs_code=None):
        self.time_logger.train_time = time.time() - self.start_time
        print("Training time: ", self.time_logger.train_time)

        plot_loss_curve(loss_values, "Loss", self.logger.output_path)
        best_code = best_ifs_code
        
        torch.save(best_code, f"{self.logger.output_path}/optimized_ifs_code.pth")
        torch.save(self.model.foreground_color, f"{self.logger.output_path}/optimized_color.pth")
        
        out_dict = self.model.create_stats_dict("opt", ifs=best_code)
        out_dict.optimized_color = self.model.foreground_color.detach().cpu().flatten().tolist()
        write_to_json(out_dict, f"{self.logger.output_path}/optimized_ifs_code.json")

        create_video_tiled_gt(self.logger.batch_final_imgs_path, self.logger.batch_final_diff_path, tensor_to_image(self.gt_img.clone()), os.path.join(self.logger.output_path, "sequence.mp4"))


    def save_during_training(self, iter, training_iters):
        if iter == training_iters:
            self.renderer.save_render(f"{self.logger.output_path}/ckpt_image.png")

        if (iter % 50 == 0) or (iter == 1):
            str_name = create_name(iter)
            self.renderer.save_render(f"{self.logger.batch_final_imgs_path}/{str_name}.png")
            diff_img = tensor_to_image((self.gt_img - self.init_img).clone())
            diff_img = apply_signed_colormap(diff_img)
            diff_img = np.array(diff_img, dtype=np.float32)
            save_image(diff_img, f"{self.logger.batch_final_diff_path}/{str_name}.png")

        if iter == 1: # to save init image
            self.renderer.save_render(f"{self.logger.init_path}/{str_name}_opt_res.png")
        

    def val(self):
        '''
        takes the ifs_code generated during training to generate fractals
        '''
        model = ModelInfer(f"{self.logger.output_path}/best_optimized_ifs_code.pth", lf=False, naive=False)
        points = model.forward()

        os.makedirs(os.path.join(self.logger.output_path, 'val'), exist_ok=True)

        stime = time.time()
        points = batch_point_gen(model, int(1e8/2), None)
        print('Time taken to generate 1e8 points:', time.time()-stime)

        base_res = 1024
        factor = 8
        color = self.model.foreground_color.detach().cpu().numpy()
        bg_color = self.r_config.background_color
        fractal = SuperSampleRender(points, base_res, factor, color, bg_color)

        # check path and test locally for 10 iters before sending to git
        super_path = os.path.join(self.logger.output_path, "val", f'supersampled.png')
        save_image(fractal, super_path)

        print("Finished rendering using optimized IFS Code")
        # Unbind only when the gt_renderer is not being used anymore
        self.renderer.unbind()
        dump_notebook(self.logger.log_dir, self.time_logger)
