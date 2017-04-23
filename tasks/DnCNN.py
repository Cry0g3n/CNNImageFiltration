from models.DnCNN import DnCNN

net = DnCNN()
net.create_model(height=64, width=64, channels=1)
net.load_data('gauss_noise_patches-size-64.pickle')
net.save_data('noise_patch', 'clear_patch')
net.fit(batch_size=32, nb_epochs=1)
