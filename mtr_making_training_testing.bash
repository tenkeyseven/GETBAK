# python making_poisoned_datasets.py
# python training_backdoor_model.py
# python test_attack_new.py

# # clean-label baseline
python create_clean_label_datasets.py
python training_backdoor_model.py
python test_attack_new.py

# # Global Random Noise Trigger  
# #python random_noise_generator.py  
# python create_random_global_trigger_datasets.py
# python training_backdoor_model.py
# python test_attack_new.py