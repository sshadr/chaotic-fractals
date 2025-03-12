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

from utils.losses import constrain_condition_number, lipschitz_constraint_sigma, ssim, constrain_bias, recon_loss
from utils.optim_utils import sample_patch, prepare_output_and_logger, create_name, plot_loss_curve, write_to_json

from easydict import EasyDict as edict
from utils.notebook import dump_notebook
import time
import lpips

class FractalTrainerSA():
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
        parameterization = self.cfg['parameterization']
        training_iters = self.cfg['training_iters']  # number of training iterations        
        
        loss_values = []
        iter = 1
        best_loss = 100000
        best_ifs_code = self.model.ifs_code
        new_view = None

        simulated_anneal_opacity = False
        temp_vals = []
        energy_vals = []
        metropolis_vals = []
        
        lpips_loss_fn = lpips.LPIPS(net='alex').cuda()
        
        with torch.no_grad():
            for iter in tqdm(range(iter, training_iters+1)):                
                linearly_decreasing_temperature_with_iter = (training_iters - iter)/training_iters
                ssim_loss = 0
                
                stime = time.time()
                points = self.model.forward() # generate points
                self.time_logger.point_gen_pass = time.time() - stime

                self.renderer.update_points(points, self.model.foreground_color)         

                full_view = sample_patch(1)
                stime = time.time()
                self.init_img = self.renderer.render(full_view) # only the full view is a self
                self.time_logger.render_pass = time.time() - stime

                # -------- Compute Loss ----------
                mse_loss = recon_loss(self.gt_img, self.init_img)
                ssim_loss = 1 - ssim(self.init_img, self.gt_img)
                lpips_loss = lpips_loss_fn(self.init_img, self.gt_img)

                # ----------------- 
                # regularizers
                singular_penalty = lipschitz_constraint_sigma(self.model.get_singular_values())
                bias_penalty = constrain_bias(self.model.get_processed_biases())
                condition_number_penalty = constrain_condition_number(self.model.get_singular_values())
                
                
                lreg = 1e-2 * (singular_penalty + bias_penalty) + 1e-1 * condition_number_penalty

                loss =  10 * mse_loss + (2 * lpips_loss + 1 * ssim_loss) +\
                        lreg
                        
                loss_values.append(loss.item())

                # -------- Compute Loss ----------
                
                self.time_logger.backward_pass = 0

                # save after forward prop:
                if iter % 50 == 0 or iter == 1:
                    torch.save(best_ifs_code, f"{self.logger.output_path}/optimized_ifs_code_{iter}.pth")
             
                self.save_during_training(iter, training_iters)
                
                if True: # Simulated annealing
                    sa_temp =  linearly_decreasing_temperature_with_iter 
                    annealing_noise = sa_temp * 0.2 # less random states are explored as optimization progresses

                    # store original weights before changing
                    self.model.save_current_state(save_opacity=simulated_anneal_opacity)

                    if parameterization == 'naive':
                        best_matrices_sa = self.model.current_matrices
                        best_lbias_sa = self.model.current_lbiases

                    if parameterization == 'mlp':
                        best_mlp_weights = self.model.current_mlp_params
                        best_init_tensor = self.model.current_init_tensor

                    if parameterization == 'mat_exp':
                        best_sigmas_sa = self.model.current_singular_values
                        best_uv_sa = self.model.current_uv_values
                        best_lbias_sa = self.model.current_lbiases
                    
                    if parameterization == 'angle':
                        best_sigmas_sa = self.model.current_singular_values
                        best_angles_sa = self.model.current_angles
                        best_lbias_sa = self.model.current_lbiases

                    if parameterization == 'qr':
                        best_sigmas_sa = self.model.current_singular_values
                        best_au_sa = self.model.current_au_values
                        best_av_sa = self.model.current_av_values
                        best_lbias_sa = self.model.current_lbiases

                        if simulated_anneal_opacity:
                            best_opacity_sa = self.model.current_opacities
                        
                    search_loss = loss


                    # one iteration of Simulated annealing
                    for search in range(1):
                        ## new candidate search ----------------
                        if parameterization == 'naive':
                            self.model.replace_all_params_naive(self.model.current_matrices, self.model.current_lbiases, noise=annealing_noise)

                        if parameterization == 'mat_exp':
                            self.model.replace_all_params_mat_exp(self.model.current_singular_values, self.model.current_uv_values, self.model.current_lbiases, noise=annealing_noise)
                        
                        if parameterization == 'angle':
                            self.model.replace_all_params_angle_param(self.model.current_singular_values, self.model.current_angles, self.model.current_lbiases, noise=annealing_noise)
                        
                        if parameterization == 'qr' and not simulated_anneal_opacity:
                            self.model.replace_all_params_qr(self.model.current_singular_values, self.model.current_au_values, self.model.current_av_values, self.model.current_lbiases, noise=annealing_noise)
                                                
                        if parameterization == 'qr' and simulated_anneal_opacity: #if opacities has to be SA'd
                            self.model.replace_all_params_qr(self.model.current_singular_values, self.model.current_au_values, self.model.current_av_values, self.model.current_lbiases, self.model.current_opacities, noise=annealing_noise)
            
                        points = self.model.forward()
                        new_view = sample_patch(1)
                        self.renderer.update_points(points, self.model.foreground_color)
                        init_img = self.renderer.render(new_view) 

                        # rendered new candidate ----------------
                        # evaluate new candidate
                        mse_loss = recon_loss(self.gt_img, init_img)
                        ssim_loss = 1 - ssim(init_img, self.gt_img)
                        lpips_loss = lpips_loss_fn(init_img, self.gt_img)

                        # ----------------- 
                        # regularizers
                        singular_penalty = lipschitz_constraint_sigma(self.model.get_singular_values())
                        bias_penalty = constrain_bias(self.model.get_processed_biases())
                        condition_number_penalty = constrain_condition_number(self.model.get_singular_values())
                
                
                        clreg = 1e-2 * (singular_penalty + bias_penalty) + 1e-1 * condition_number_penalty

                        candidate_loss =  10 * mse_loss + (2 * lpips_loss + 1 * ssim_loss) +\
                                clreg

                        energy_state = candidate_loss - search_loss

                        metropolis_criterion = torch.exp(-10 * energy_state /(sa_temp))
                        
                        metropolis_vals.append(metropolis_criterion.item())
                        temp_vals.append(sa_temp)
                        energy_vals.append(energy_state.item())

                        if energy_state.item() <= 0.0 or (torch.rand(1) < metropolis_criterion.cpu()):
                            # store new best candidate:
                            if parameterization == 'naive':
                                best_matrices_sa, best_lbias_sa = self.model.get_all_params_naive()

                            if parameterization == 'mat_exp':
                                best_sigmas_sa, best_uv_sa, best_lbias_sa = self.model.get_all_params_mat_exp()
                            
                            if parameterization == 'angle':
                                best_sigmas_sa, best_angles_sa, best_lbias_sa = self.model.get_all_params_angle_param()
                            
                            if parameterization == 'qr' and not simulated_anneal_opacity:
                                best_sigmas_sa, best_au_sa, best_av_sa, best_lbias_sa = self.model.get_all_params_qr(return_opacity=False)
                            
                            if parameterization == 'qr' and simulated_anneal_opacity: #if opacities has to be SA'd
                                best_sigmas_sa, best_au_sa, best_av_sa, best_lbias_sa, best_opacity_sa = self.model.get_all_params_qr(return_opacity=True)
                            
                            search_loss = candidate_loss
                            
                    # ----- replace weights with the best weights found using simulated annealing
                    if parameterization == 'naive':
                        self.model.replace_all_params_naive(best_matrices_sa, best_lbias_sa, noise=0.0)

                    if parameterization == 'mat_exp':
                        self.model.replace_all_params_mat_exp(best_sigmas_sa, best_uv_sa, best_lbias_sa, noise=0.0)
                    
                    if parameterization == 'angle':
                        self.model.replace_all_params_angle_param(best_sigmas_sa, best_angles_sa, best_lbias_sa, noise=0.0)
                    
                    if parameterization == 'qr' and not simulated_anneal_opacity:
                        self.model.replace_all_params_qr(best_sigmas_sa, best_au_sa, best_av_sa, best_lbias_sa, noise=0.0)
                    
                    if parameterization == 'qr' and simulated_anneal_opacity: #if opacities has to be SA'd
                        self.model.replace_all_params_qr(best_sigmas_sa, best_au_sa, best_av_sa, best_lbias_sa, best_opacity_sa, noise=0.0)
                   
                
                if loss.item() < best_loss:
                    best_ifs_code = self.model.ifs_code
                    torch.save(self.model.ifs_code, f"{self.logger.output_path}/best_optimized_ifs_code.pth")
                    
                    best_loss = loss.item()


            self.writer.add_scalar("Loss", loss.item(), iter)

            # print current loss
            tqdm.write(
                "Iteration: {}, Loss: {}".format(
                    iter, loss.item()
                )
            )
            
            
        before_op_dict = self.model.create_stats_dict("pre_opt")

        write_to_json(before_op_dict, f"{self.logger.output_path}/pre_optimized_ifs_code.json")
        
        # returns ifs_code after post_processing in model
        plot_loss_curve(metropolis_vals, "Metropolis_criterion", self.logger.output_path, log_plot=False)
        plot_loss_curve(temp_vals, "Temperature", self.logger.output_path, log_plot=False)
        plot_loss_curve(energy_vals, "Energy State", self.logger.output_path, log_plot=False)

        
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
        points = self.model.forward() # generate points
        self.renderer.update_points(points, self.model.foreground_color)
        full_view = sample_patch(1)
        img = self.renderer.render(view_matrix=full_view)

        if iter == training_iters:
            self.renderer.save_render(f"{self.logger.output_path}/ckpt_image.png")

        if (iter % 50 == 0) or (iter == 1):
            str_name = create_name(iter)
            self.renderer.save_render(f"{self.logger.batch_final_imgs_path}/{str_name}.png")
            diff_img = tensor_to_image((self.gt_img - img).clone())
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
