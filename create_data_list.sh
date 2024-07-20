
python scripts/make_file_list.py \
			--img_folder data/OpenImage/ \
			--val_size 5000 \
			--save_folder ./datalist/OpenImage/

python scripts/make_file_list.py \
			--img_folder data/CelebAHQ_train_512/ \
			--val_size 1000 \
			--save_folder ./datalist/CelebAHQ/

