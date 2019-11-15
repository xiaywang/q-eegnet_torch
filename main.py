from eegnet_controller import train_subject_specific
from utils.plot_results import plot_loss_accuracy

for subject in range(1, 10):
    _model, loss, accuracy = train_subject_specific(subject, epochs=500)
    plot_loss_accuracy(subject, loss, accuracy)
