from eegnet_controller import train_subject_specific
from utils.plot_results import plot_loss_accuracy


sum_accuracy = 0.0

for subject in range(1, 10):
    _model, loss, accuracy = train_subject_specific(subject, epochs=500)
    plot_loss_accuracy(subject, loss, accuracy)
    sum_accuracy += accuracy[1, -1]

sum_accuracy = sum_accuracy / 9

print(f"\nAverage Accuracy: {sum_accuracy}")
