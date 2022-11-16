CUDA_VISIBLE_DEVICES=0 python main.py --model_name_or_path hustvl/yolos-tiny --data_dir dataset_masks \
                            --train train --val eval --test test \
                            --do_train --do_eval --do_predict \
                            --train_args train_args.json --random_seed 42