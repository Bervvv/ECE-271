%% Bolin He, PID: A53316428, Hw03
% Nov 27,2019

clear all;
close all;
clc;

Dataset = load('TrainingSamplesDCT_subsets_8.mat');
Alpha = load('Alpha.mat');
Alpha = Alpha.alpha;

for strategy = 1:2
    
    if strategy == 1
        strategy = load('Prior_1.mat');
    elseif strategy == 2
        strategy = load('Prior_2.mat');
    end
    
    for dataset = 1:4
        if dataset == 1
            BG = Dataset.D1_BG;
            FG = Dataset.D1_FG;
        elseif dataset == 2
            BG = Dataset.D2_BG;
            FG = Dataset.D2_FG;
        elseif dataset == 3
            BG = Dataset.D3_BG;
            FG = Dataset.D3_FG;
        elseif dataset == 4
            BG = Dataset.D4_BG;
            FG = Dataset.D4_FG;
        end
        
        bayes_error = [];
        mle_error = [];
        map_error = [];
        n_FG = length(FG);
        n_BG = length(BG);

        % Initialization
        for alpha_idx = 1:length(Alpha)
            cov_0 = zeros(64,64);
            for idx = 1:64
               cov_0(idx,idx) = Alpha(alpha_idx)*strategy.W0(idx); 
            end

            % FG
            FG_cov = cov(FG);
            t = inv(cov_0 + (1/n_FG)*FG_cov);
            mu_1_FG = cov_0 * t * transpose(mean(FG)) + (1/n_FG) * FG_cov * t * transpose(strategy.mu0_FG);
            cov_1_FG = cov_0 * t * (1/n_FG) * FG_cov;
            
            % predictive distribution of FG
            mu_pred_FG = mu_1_FG;
            cov_pred_FG = FG_cov + cov_1_FG;

            % BG
            BG_cov = cov(BG);
            t2 = inv(cov_0 + (1/n_BG)*BG_cov);
            mu_1_BG = cov_0 * t2 * transpose(mean(BG)) + (1/n_BG) * BG_cov * t2 * transpose(strategy.mu0_BG);
            cov_1_BG = cov_0 * t2 * (1/n_BG) * BG_cov;
            
            % predictive distribution of BG
            mu_pred_BG = mu_1_BG;
            cov_pred_BG = BG_cov + cov_1_BG;

            % Prior
            num_FG = length(FG);
            num_BG = length(BG);
            prior_FG = num_FG / (num_FG + num_BG);
            prior_BG = num_BG / (num_FG + num_BG);
        
            % Bayes
            I = imread('cheetah.bmp');
            I = im2double(I);
            
            % Paddling
            I = [zeros(size(I,1),2) I];
            I = [zeros(2, size(I,2)); I];
            I = [I zeros(size(I,1),5)];
            I = [I; zeros(5, size(I,2))];

            % DCT
            [m,n] = size(I);
            Blocks = ones(m-7,n-7);
            det_cov_FG = det(cov_pred_FG);
            det_cov_BG = det(cov_pred_BG);
            ave_tmp_FG = transpose(mu_pred_FG);
            ave_tmp_BG = transpose(mu_pred_BG);
            inv_tmp_FG = inv(cov_pred_FG);
            inv_tmp_BG = inv(cov_pred_BG);

            % predict
            const_FG = ave_tmp_FG*inv_tmp_FG*transpose(ave_tmp_FG) + log(det_cov_FG) - 2*log(prior_FG);
            const_BG = ave_tmp_BG*inv_tmp_BG*transpose(ave_tmp_BG) + log(det_cov_BG) - 2*log(prior_BG);
            
            for i=1:m-7
                for j=1:n-7
                    DCT = dct2(I(i:i+7,j:j+7));
                    zigzag_order = zigzag(DCT);
                    feature = zigzag_order;
                    g_cheetah = 0;
                    g_grass = 0;
                    % cheetah
                    g_cheetah = g_cheetah + feature*inv_tmp_FG*transpose(feature);
                    g_cheetah = g_cheetah - 2*feature*inv_tmp_FG*transpose(ave_tmp_FG);
                    g_cheetah = g_cheetah + const_FG;
                    % grass
                    g_grass = g_grass + feature*inv_tmp_BG*transpose(feature);
                    g_grass = g_grass - 2*feature*inv_tmp_BG*transpose(ave_tmp_BG);
                    g_grass = g_grass + const_BG;
                    if g_cheetah >= g_grass
                        Blocks(i,j) = 0;
                    end
                end
            end

            % save 
            ground_truth = imread('cheetah_mask.bmp')/255;
            [x,y] = size(ground_truth);
            count1 = 0;
            count2 = 0;
            count_cheetah_truth = 0;
            count_grass_truth = 0;
            for i=1:x
                for j=1:y
                    if prediction(i,j) > ground_truth(i,j)
                        count2 = count2 + 1;
                        count_grass_truth = count_grass_truth + 1;
                    elseif prediction(i,j) < ground_truth(i,j)
                        count1 = count1 + 1;
                        count_cheetah_truth = count_cheetah_truth + 1;
                    elseif ground_truth(i,j) >0
                        count_cheetah_truth = count_cheetah_truth + 1;
                    else
                        count_grass_truth = count_grass_truth + 1;
                    end
                end
            end
            error1_64 = (count1/count_cheetah_truth) * prior_FG;
            error2_64 = (count2/count_grass_truth) * prior_BG;
            total_error_64 = error1_64 + error2_64;
            bayes_error = [bayes_error total_error_64];
            % ML
            % DCT
            [m,n] = size(I);
            Blocks = ones(m-7,n-7);
            mean_FG = mean(FG);
            mean_BG = mean(BG);
            ave_tmp_FG = mean_FG;
            ave_tmp_BG = mean_BG;
            inv_covFG = inv(FG_cov);
            inv_covBG = inv(BG_cov);
            DcovFG = det(FG_cov);
            DcovBG = det(BG_cov);

            % predict
            const_FG = ave_tmp_FG*inv_covFG*transpose(ave_tmp_FG) + log(DcovFG) - 2*log(prior_FG);
            const_BG = ave_tmp_BG*inv_covBG*transpose(ave_tmp_BG) + log(DcovBG) - 2*log(prior_BG);
            
            for i=1:m-7
                for j=1:n-7
                    DCT = dct2(I(i:i+7,j:j+7));
                    zigzag_order = zigzag(DCT);
                    feature = zigzag_order;
                    g_cheetah = 0;
                    g_grass = 0;
                    % cheetah
                    g_cheetah = g_cheetah + feature*inv_covFG*transpose(feature);
                    g_cheetah = g_cheetah - 2*feature*inv_covFG*transpose(ave_tmp_FG);
                    g_cheetah = g_cheetah + const_FG;
                    % grass
                    g_grass = g_grass + feature*inv_covBG*transpose(feature);
                    g_grass = g_grass - 2*feature*inv_covBG*transpose(ave_tmp_BG);
                    g_grass = g_grass + const_BG;
                    if g_cheetah >= g_grass
                        Blocks(i,j) = 0;
                    end
                end
            end

            % save 
            count1 = 0;
            count2 = 0;
            count_cheetah_truth = 0;
            count_grass_truth = 0;
            for i=1:x
                for j=1:y
                    if prediction(i,j) > ground_truth(i,j)
                        count2 = count2 + 1;
                        count_grass_truth = count_grass_truth + 1;
                    elseif prediction(i,j) < ground_truth(i,j)
                        count1 = count1 + 1;
                        count_cheetah_truth = count_cheetah_truth + 1;
                    elseif ground_truth(i,j) >0
                        count_cheetah_truth = count_cheetah_truth + 1;
                    else
                        count_grass_truth = count_grass_truth + 1;
                    end
                end
            end
            error1_64 = (count1/count_cheetah_truth) * prior_FG;
            error2_64 = (count2/count_grass_truth) * prior_BG;
            total_error_64 = error1_64 + error2_64;
            mle_error = [mle_error total_error_64];

            % MAP            
            % DCT
            [m,n] = size(I);
            Blocks = ones(m-7,n-7);
            det_cov_FG = det(FG_cov);
            det_cov_BG = det(BG_cov);
            ave_tmp_FG = transpose(mu_pred_FG);
            ave_tmp_BG = transpose(mu_pred_BG);

            % predict
            const_FG = ave_tmp_FG*inv_covFG*transpose(ave_tmp_FG) + log(det_cov_FG) - 2*log(prior_FG);
            const_BG = ave_tmp_BG*inv_covBG*transpose(ave_tmp_BG) + log(det_cov_BG) - 2*log(prior_BG);
            
            for i=1:m-7
                for j=1:n-7
                    DCT = dct2(I(i:i+7,j:j+7));
                    zigzag_order = zigzag(DCT);
                    feature = zigzag_order;
                    g_cheetah = 0;
                    g_grass = 0;
                    % cheetah
                    g_cheetah = g_cheetah + feature*inv_covFG*transpose(feature);
                    g_cheetah = g_cheetah - 2*feature*inv_covFG*transpose(ave_tmp_FG);
                    g_cheetah = g_cheetah + const_FG;
                    % grass
                    g_grass = g_grass + feature*inv_covBG*transpose(feature);
                    g_grass = g_grass - 2*feature*inv_covBG*transpose(ave_tmp_BG);
                    g_grass = g_grass + const_BG;
                    if g_cheetah >= g_grass
                        Blocks(i,j) = 0;
                    end
                end
            end
            
            % save
            count1 = 0;
            count2 = 0;
            count_cheetah_truth = 0;
            count_grass_truth = 0;
            for i=1:x
                for j=1:y
                    if prediction(i,j) > ground_truth(i,j)
                        count2 = count2 + 1;
                        count_grass_truth = count_grass_truth + 1;
                    elseif prediction(i,j) < ground_truth(i,j)
                        count1 = count1 + 1;
                        count_cheetah_truth = count_cheetah_truth + 1;
                    elseif ground_truth(i,j) >0
                        count_cheetah_truth = count_cheetah_truth + 1;
                    else
                        count_grass_truth = count_grass_truth + 1;
                    end
                end
            end
            error1_64 = (count1/count_cheetah_truth) * prior_FG;
            error2_64 = (count2/count_grass_truth) * prior_BG;
            total_error_64 = error1_64 + error2_64;
            map_error = [map_error total_error_64];
        end
        
        % plot
        plot(Alpha,bayes_error,Alpha,mle_error,Alpha,map_error);
        legend('Predict','ML','MAP');
        set(gca, 'XScale', 'log');
        xlabel('Alpha');
        ylabel('PoE');
        title('PoE vs Alpha');
        saveas(gcf,['Strategy_' int2str(strategy) '_dataset_' int2str(dataset) '_PoEvsAlpha.png']);
    end
end
